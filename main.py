import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

os.environ["PATH"] = "/home/runner/workspace/.local/share/../bin:" + os.environ.get("PATH", "")

app = FastAPI()

API_KEY = os.environ.get("OPS_SEARCH_KEY")  # لو ما انحط، السيرفر يشتغل مفتوح مثل قبل
ALLOWED_PLATFORMS = {p.strip() for p in os.environ.get("ALLOWED_PLATFORMS", "youtube").split(",") if p.strip()}
CHANNEL_SAMPLE = 15
MIN_SAMPLE = 3
CACHE_TTL = 6 * 3600
CHANNEL_ID_RE = re.compile(r"^UC[\w-]{20,}$")
_channel_cache: dict = {}
_yt_slots = threading.BoundedSemaphore(3)  # حد لعمليات yt-dlp المتزامنة (الخطة المجانية ضعيفة)


def require_key(authorization: str = Header(default="")):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="unauthorized")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    platform: str = "youtube"
    limit: int = Field(default=8, ge=1, le=15)
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
        return {"ok": False, "items": [], "error": f"search_failed: {e}"}

    body = parse_json_output(proc.stdout)
    if not isinstance(body, dict):
        return {"ok": False, "items": [], "error": "bad_output"}

    items = body.get("items")
    items = items[: data.limit] if isinstance(items, list) else []
    if data.outliers and platform == "youtube" and items:
        with ThreadPoolExecutor(max_workers=3) as pool:
            items = list(pool.map(enrich, items))
    body["items"] = items
    body.setdefault("ok", True)
    body["meta"] = {**(body.get("meta") or {}), "total_ms": int((time.time() - t0) * 1000)}
    return body


@app.get("/diag", dependencies=[Depends(require_key)])
def diag():
    return {
        "yt_dlp": shutil.which("yt-dlp"),
        "agent_reach": shutil.which("agent-reach"),
        "platforms": sorted(ALLOWED_PLATFORMS),
        "auth_enabled": bool(API_KEY),
    }


@app.get("/")
def home():
    return {"status": "Agent Reach is working"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
