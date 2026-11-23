import json
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi

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

TARGET_VALID_VIDEOS = 5
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
    limit: int = Field(default=12, ge=1, le=20)
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
    cid = raw.get("channel_id")
    vid = raw.get("id")
    views = raw.get("view_count")

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
    for key in ["id", "video_id"]:
        vid = item.get(key)
        if isinstance(vid, str) and len(vid) == 11:
            return vid

    raw = item.get("raw") or {}
    for key in ["id", "video_id"]:
        vid = raw.get(key)
        if isinstance(vid, str) and len(vid) == 11:
            return vid

    url = item.get("url") or item.get("link") or raw.get("url") or raw.get("webpage_url") or ""
    match = re.search(r"(?:v=|youtu\.be/|shorts/|/embed/)([a-zA-Z0-9_-]{11})", str(url))
    if match:
        return match.group(1)
    return None


def get_transcript_via_api(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """الطريقة الأولى: youtube-transcript-api"""
    try:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(video_id, languages=["ar", "ar-SA", "ar-EG", "en", "en-US"])
            return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
        except Exception:
            pass

        try:
            transcript_list = api.list(video_id)
            for lang_group in [["ar", "ar-SA", "ar-EG"], ["en", "en-US"]]:
                try:
                    t = transcript_list.find_transcript(lang_group)
                    fetched = t.fetch()
                    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
                except Exception:
                    continue
            # أي لغة
            for t in transcript_list:
                try:
                    fetched = t.fetch()
                    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
                except Exception:
                    continue
        except Exception:
            pass
    except Exception:
        pass
    return None


def parse_vtt(content: str) -> List[Dict[str, Any]]:
    """تحويل ملف VTT إلى segments"""
    segments = []
    lines = content.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            try:
                times = line.split("-->")
                start_str = times[0].strip().replace(",", ".")
                # تحويل الوقت إلى ثواني
                def to_seconds(t):
                    parts = t.split(":")
                    if len(parts) == 3:
                        h, m, s = parts
                        return int(h) * 3600 + int(m) * 60 + float(s)
                    elif len(parts) == 2:
                        m, s = parts
                        return int(m) * 60 + float(s)
                    return float(t)
                start = to_seconds(start_str)
                # جمع النص
                text_parts = []
                i += 1
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    text_parts.append(lines[i].strip())
                    i += 1
                text = " ".join(text_parts).strip()
                if text:
                    segments.append({"text": text, "start": start, "duration": 2.0})
                continue
            except Exception:
                pass
        i += 1
    return segments


def get_transcript_via_ytdlp(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """الطريقة الثانية: yt-dlp (أقوى ضد الحظر)"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = subprocess.run(
                [
                    "yt-dlp",
                    "--skip-download",
                    "--write-auto-subs",
                    "--sub-langs", "ar.*,en.*",
                    "--convert-subs", "vtt",
                    "--output", f"{tmpdir}/%(id)s",
                    "--quiet",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
        except Exception:
            return None

        # نبحث عن أي ملف vtt
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            return None

        # نفضل العربي
        chosen = None
        for f in vtt_files:
            if "ar" in f.name.lower():
                chosen = f
                break
        if not chosen:
            chosen = vtt_files[0]

        try:
            content = chosen.read_text(encoding="utf-8", errors="ignore")
            segments = parse_vtt(content)
            return segments if segments else None
        except Exception:
            return None


def get_transcript_segments(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """يجرب الـ API أولاً، ثم yt-dlp كخطة بديلة"""
    # 1. جرب المكتبة الرسمية
    segments = get_transcript_via_api(video_id)
    if segments:
        return segments

    # 2. خطة بديلة قوية
    segments = get_transcript_via_ytdlp(video_id)
    if segments:
        return segments

    return None


def split_spoken_parts(segments: list, duration_hint: Optional[float] = None) -> dict:
    if not segments:
        return {"spoken_hook": "", "spoken_cta": "", "transcript_sample": ""}

    segs = sorted(segments, key=lambda x: x.get("start", 0))
    last = segs[-1]
    total_duration = last.get("start", 0) + last.get("duration", 2)
    if duration_hint and duration_hint > total_duration:
        total_duration = duration_hint

    # Hook
    hook_parts = []
    for s in segs:
        if s.get("start", 0) > HOOK_SECONDS:
            break
        text = (s.get("text") or "").strip()
        if text:
            hook_parts.append(text)
    spoken_hook = " ".join(hook_parts).strip()

    # CTA
    cta_start = max(0, total_duration - CTA_SECONDS)
    cta_parts = []
    for s in segs:
        if s.get("start", 0) >= cta_start:
            text = (s.get("text") or "").strip()
            if text:
                cta_parts.append(text)
    spoken_cta = " ".join(cta_parts).strip()

    # Sample
    mid_start = total_duration * 0.2
    mid_end = total_duration * 0.8
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
    video_id = extract_video_id(item)
    if not video_id:
        return None

    segments = get_transcript_segments(video_id)
    if not segments:
        return None

    duration = None
    raw = item.get("raw") or {}
    if isinstance(raw.get("duration"), (int, float)):
        duration = float(raw["duration"])
    elif isinstance(item.get("duration"), (int, float)):
        duration = float(item["duration"])

    spoken = split_spoken_parts(segments, duration)

    return {
        **item,
        "video_id": video_id,
        "spoken_hook": spoken["spoken_hook"],
        "spoken_cta": spoken["spoken_cta"],
        "transcript_sample": spoken["transcript_sample"],
        "has_transcript": True,
    }


@app.post("/search", dependencies=[Depends(require_key)])
def search(data: SearchRequest):
    platform = data.platform.strip().lower()
    if platform not in ALLOWED_PLATFORMS:
        return {"ok": False, "items": [], "error": f"platform_not_allowed: {platform}"}

    query = data.query.strip().lstrip("-").strip()
    if not query:
        return {"ok": False, "items": [], "error": "empty_query"}

    t0 = time.time()

    try:
        proc = subprocess.run(
            ["agent-reach", platform, "search", query],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "items": [], "error": "search_timeout"}
    except Exception as e:
        return {"ok": False, "items": [], "error": f"search_failed: {str(e)}"}

    body = parse_json_output(proc.stdout)
    if not isinstance(body, dict):
        return {"ok": False, "items": [], "error": "bad_output"}

    raw_items = body.get("items") if isinstance(body.get("items"), list) else []

    if data.outliers and platform == "youtube" and raw_items:
        with ThreadPoolExecutor(max_workers=3) as pool:
            raw_items = list(pool.map(enrich, raw_items))

    valid_items = []
    sorted_items = sorted(raw_items, key=lambda x: x.get("outlier_ratio") or 0, reverse=True)

    for item in sorted_items:
        if len(valid_items) >= TARGET_VALID_VIDEOS:
            break
        processed = process_video(item)
        if processed:
            valid_items.append(processed)

    body["items"] = valid_items
    body.setdefault("ok", True)
    body["meta"] = {
        **(body.get("meta") or {}),
        "total_ms": int((time.time() - t0) * 1000),
        "valid_with_transcript": len(valid_items),
        "requested_target": TARGET_VALID_VIDEOS,
        "raw_found": len(raw_items),
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
        "fallback": "yt-dlp",
    }


@app.get("/")
def home():
    return {"status": "Agent Reach is working", "version": "2.1-ytdlp-fallback"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
