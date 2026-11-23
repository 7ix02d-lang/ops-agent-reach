import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

os.environ["PATH"] = "/home/runner/workspace/.local/share/../bin:" + os.environ.get("PATH", "")

app = FastAPI()

API_KEY = os.environ.get("OPS_SEARCH_KEY")
ALLOWED_PLATFORMS = {p.strip() for p in os.environ.get("ALLOWED_PLATFORMS", "youtube").split(",") if p.strip()}
CHANNEL_SAMPLE = 15
MIN_SAMPLE = 3
CACHE_TTL = 6 * 3600
CHANNEL_ID_RE = re.compile(r"^UC[\w-]{20,}$")
_channel_cache: dict = {}
_yt_slots = threading.BoundedSemaphore(3)

# الإعدادات الجديدة
TARGET_VALID_VIDEOS = 5          # نحاول نوصل لـ 5
MIN_VALID_VIDEOS = 3
HOOK_SECONDS = 10
CTA_SECONDS = 15
SAMPLE_CHARS = 900


def require_key(authorization: str = Header(default="")):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    platform: str = "youtube"
    limit: int = Field(default=12, ge=1, le=20)  # نرفع الـ limit عشان نقدر نفلتر
    outliers: bool = True


def parse_json_output(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def channel_views(channel_id: str):
    now = time.time()
    hit = _channel_cache.get(channel_id)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    with _yt_slots:
        try:
            proc = subprocess.run(
                [
                    "yt-dlp", "--flat-playlist", "--dump-json",
                    "--playlist-end", str(CHANNEL_SAMPLE),
                    f"https://www.youtube.com/channel/{channel_id}/videos",
                ],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            return None
    views = []
    for line in proc.stdout.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        v = d.get("view_count")
        if isinstance(v, (int, float)):
            views.append((d.get("id"), v))
    if views:
        _channel_cache[channel_id] = (now, views)
    return views or None


def enrich(item: dict):
    raw = item.get("raw") or {}
    cid, vid, views = raw.get("channel_id"), raw.get("id"), raw.get("view_count")
    if not (isinstance(cid, str) and CHANNEL_ID_RE.match(cid) and isinstance(views, (int, float))):
        return item
    sample = channel_views(cid)
    if not sample:
        return item
    others = [v for (i, v) in sample if i != vid]
    if len(others) < MIN_SAMPLE:
        return item
    avg = sum(others) / len(others)
    if avg > 0:
        item["channel_avg_views"] = round(avg)
        item["channel_sample_size"] = len(others)
        item["outlier_ratio"] = round(views / avg, 2)
    return item


def extract_video_id(item: dict) -> Optional[str]:
    """استخراج video_id من نتائج agent-reach"""
    # أشكال محتملة
    vid = item.get("id") or item.get("video_id")
    if isinstance(vid, str) and len(vid) == 11:
        return vid

    raw = item.get("raw") or {}
    vid = raw.get("id") or raw.get("video_id")
    if isinstance(vid, str) and len(vid) == 11:
        return vid

    url = item.get("url") or item.get("link") or raw.get("url") or ""
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)

    return None


def get_transcript_segments(video_id: str) -> Optional[list]:
    """جلب الترانسكريبت (يفضل عربي، وإلا auto)"""
    try:
        # نحاول عربي أولاً ثم أي لغة متاحة
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # أولوية: عربي يدوي → عربي auto → أي لغة
        try:
            transcript = transcript_list.find_transcript(["ar", "ar-SA", "ar-EG"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["ar", "ar-SA", "ar-EG"])
            except Exception:
                # أي ترانسكريبت متاح
                available = list(transcript_list)
                if not available:
                    return None
                transcript = available[0]

        return transcript.fetch()
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, Exception):
        return None


def split_spoken_parts(segments: list, duration_hint: Optional[float] = None) -> dict:
    """
    segments: قائمة dict فيها start, duration, text
    نرجع spoken_hook, spoken_cta, transcript_sample
    """
    if not segments:
        return {
            "spoken_hook": "",
            "spoken_cta": "",
            "transcript_sample": "",
        }

    # نحول لـ list مرتبة
    segs = sorted(segments, key=lambda x: x.get("start", 0))

    # حساب المدة التقريبية
    last = segs[-1]
    total_duration = last.get("start", 0) + last.get("duration", 0)
    if duration_hint and duration_hint > total_duration:
        total_duration = duration_hint

    # --- Hook: 0 → 10 ثواني ---
    hook_parts = []
    for s in segs:
        start = s.get("start", 0)
        if start > HOOK_SECONDS:
            break
        text = (s.get("text") or "").strip()
        if text:
            hook_parts.append(text)
    spoken_hook = " ".join(hook_parts).strip()

    # --- CTA: آخر 15 ثانية ---
    cta_start = max(0, total_duration - CTA_SECONDS)
    cta_parts = []
    for s in segs:
        start = s.get("start", 0)
        if start >= cta_start:
            text = (s.get("text") or "").strip()
            if text:
                cta_parts.append(text)
    spoken_cta = " ".join(cta_parts).strip()

    # --- Sample من الوسط (حوالي 800-1000 حرف) ---
    mid_start = total_duration * 0.25
    mid_end = total_duration * 0.75
    sample_parts = []
    current_len = 0
    for s in segs:
        start = s.get("start", 0)
        if start < mid_start:
            continue
        if start > mid_end and current_len >= 600:
            break
        text = (s.get("text") or "").strip()
        if text:
            sample_parts.append(text)
            current_len += len(text) + 1
            if current_len >= SAMPLE_CHARS:
                break

    transcript_sample = " ".join(sample_parts).strip()
    if len(transcript_sample) > SAMPLE_CHARS + 100:
        transcript_sample = transcript_sample[:SAMPLE_CHARS] + "..."

    return {
        "spoken_hook": spoken_hook,
        "spoken_cta": spoken_cta,
        "transcript_sample": transcript_sample,
    }


def process_video(item: dict) -> Optional[dict]:
    """معالجة فيديو واحد: جلب الترانسكريبت + استخراج الأجزاء"""
    video_id = extract_video_id(item)
    if not video_id:
        return None

    segments = get_transcript_segments(video_id)
    if not segments:
        return None  # تخطي حسب القرار المعتمد

    # مدة تقريبية من البيانات إن وجدت
    duration = None
    raw = item.get("raw") or {}
    if isinstance(raw.get("duration"), (int, float)):
        duration = float(raw["duration"])
    elif isinstance(item.get("duration"), (int, float)):
        duration = float(item["duration"])

    spoken = split_spoken_parts(segments, duration)

    # نبني العنصر النهائي
    result = {
        **item,  # نحافظ على البيانات الأصلية (title, views, outlier...)
        "video_id": video_id,
        "spoken_hook": spoken["spoken_hook"],
        "spoken_cta": spoken["spoken_cta"],
        "transcript_sample": spoken["transcript_sample"],
        "has_transcript": True,
    }
    return result


@app.post("/search", dependencies=[Depends(require_key)])
def search(data: SearchRequest):
    platform = data.platform.strip().lower()
    if platform not in ALLOWED_PLATFORMS:
        return {"ok": False, "items": [], "error": f"platform_not_allowed: {platform}"}

    query = data.query.strip().lstrip("-").strip()
    if not query:
        return {"ok": False, "items": [], "error": "empty_query"}

    t0 = time.time()

    # 1. بحث أولي
    try:
        proc = subprocess.run(
            ["agent-reach", platform, "search", query],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "items": [], "error": "search_timeout"}
    except Exception as e:
        return {"ok": False, "items": [], "error": f"search_failed: {e}"}

    body = parse_json_output(proc.stdout)
    if not isinstance(body, dict):
        return {"ok": False, "items": [], "error": "bad_output"}

    raw_items = body.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    # 2. outlier enrichment (كما كان)
    if data.outliers and platform == "youtube" and raw_items:
        with ThreadPoolExecutor(max_workers=3) as pool:
            raw_items = list(pool.map(enrich, raw_items))

    # 3. فلترة + جلب الترانسكريبت (الجديد)
    valid_items = []
    # نرتب حسب outlier_ratio تنازلياً إن وجد
    sorted_items = sorted(
        raw_items,
        key=lambda x: x.get("outlier_ratio") or 0,
        reverse=True,
    )

    for item in sorted_items:
        if len(valid_items) >= TARGET_VALID_VIDEOS:
            break
        processed = process_video(item)
        if processed:
            valid_items.append(processed)

    # لو ما وصلنا للحد الأدنى نرجع اللي لقيناه
    if len(valid_items) < MIN_VALID_VIDEOS:
        # ممكن نرجع تحذير، لكن نرجع النتائج الموجودة
        pass

    body["items"] = valid_items
    body.setdefault("ok", True)
    body["meta"] = {
        **(body.get("meta") or {}),
        "total_ms": int((time.time() - t0) * 1000),
        "valid_with_transcript": len(valid_items),
        "requested_target": TARGET_VALID_VIDEOS,
    }
    return body


@app.get("/diag", dependencies=[Depends(require_key)])
def diag():
    return {
        "yt_dlp": shutil.which("yt-dlp"),
        "agent_reach": shutil.which("agent-reach"),
        "platforms": sorted(ALLOWED_PLATFORMS),
        "auth_enabled": bool(API_KEY),
        "transcript_api": True,
    }


@app.get("/")
def home():
    return {"status": "Agent Reach is working"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
