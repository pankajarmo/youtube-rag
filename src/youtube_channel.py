from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from yt_dlp import YoutubeDL


def normalize_listing_url(url: str) -> str:
    """
    Bare youtube.com/@handle links resolve to the channel home tab in yt-dlp, which
    yields channel metadata rows (UC… ids), not watch-page video ids. Append /videos
    when the path is only /@handle so listing returns real uploads.
    """
    u = url.strip()
    parsed = urlparse(u)
    host = (parsed.hostname or "").lower()
    if host not in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        return u
    path = (parsed.path or "/").rstrip("/") or "/"
    if re.fullmatch(r"/@[\w.-]+", path, re.IGNORECASE):
        new_path = path + "/videos"
        return urlunparse(
            (
                parsed.scheme or "https",
                parsed.netloc,
                new_path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    return u


def get_channel_video_entries(
    channel_url: str,
    playlist_max: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch video entries from a channel or playlist URL using yt-dlp (flat, no download).

    Returns list of dicts with at least 'id' (video id); may include 'title'.
    """
    channel_url = normalize_listing_url(channel_url)
    ydl_opts: dict[str, Any] = {
        "extract_flat": True,
        "quiet": True,
        "playlistend": playlist_max,
        "ignoreerrors": True,
    }
    entries: list[dict[str, Any]] = []
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    if not info:
        return entries
    raw_entries = info.get("entries")
    if not raw_entries:
        return entries
    for entry in raw_entries:
        if not entry or not isinstance(entry, dict):
            continue
        vid = entry.get("id")
        if not vid or not isinstance(vid, str):
            continue
        if vid.startswith("http"):
            continue
        title = entry.get("title") or entry.get("fulltitle")
        entries.append({"id": vid, "title": title if isinstance(title, str) else None})
    return entries


def get_channel_video_ids(channel_url: str, playlist_max: int = 100) -> list[str]:
    """Backward-compatible list of video IDs only."""
    return [e["id"] for e in get_channel_video_entries(channel_url, playlist_max)]
