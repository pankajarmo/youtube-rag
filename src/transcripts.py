from __future__ import annotations

import re
import time
from dataclasses import dataclass
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import TYPE_CHECKING

from requests import Session
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

if TYPE_CHECKING:
    from youtube_transcript_api.proxies import ProxyConfig

_PREFERRED_LANGUAGES = ("en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU")
_RETRYABLE = (IpBlocked, RequestBlocked)
_MAX_RETRIES = 3
_RETRY_DELAYS = (2.0, 5.0, 10.0)


@dataclass(frozen=True)
class TranscriptResult:
    text: str | None
    reason: str | None = None  # e.g. ip_blocked, disabled, not_found


def is_youtube_video_id(video_id: str) -> bool:
    """True for 11-char watch IDs; excludes channel/playlist IDs (UC…, PL…)."""
    if not video_id or video_id.startswith("http"):
        return False
    if video_id.startswith(("UC", "PL", "UU", "FL", "RD", "LL", "VL")):
        return False
    return bool(re.fullmatch(r"[\w-]{11}", video_id))


def _session_with_cookies(cookies_path: str) -> Session:
    path = Path(cookies_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"YOUTUBE_COOKIES_PATH not found: {path}")
    jar = MozillaCookieJar(str(path))
    jar.load(ignore_discard=True, ignore_expires=True)
    session = Session()
    session.cookies = jar
    session.headers.update({"Accept-Language": "en-US"})
    return session


def build_transcript_api(
    *,
    proxy_config: ProxyConfig | None = None,
    cookies_path: str | None = None,
) -> YouTubeTranscriptApi:
    http_client = _session_with_cookies(cookies_path) if cookies_path else None
    return YouTubeTranscriptApi(proxy_config=proxy_config, http_client=http_client)


def _joined_text(fetched) -> str:
    return " ".join(snippet.text for snippet in fetched).strip()


def fetch_transcript(
    video_id: str,
    *,
    api: YouTubeTranscriptApi | None = None,
) -> TranscriptResult:
    """
    Fetch transcript text for a video. Returns reason codes when unavailable
    so callers can distinguish IP blocks from missing captions.
    """
    if not is_youtube_video_id(video_id):
        return TranscriptResult(None, "invalid_video_id")

    client = api or build_transcript_api()
    last_reason = "not_found"

    for attempt in range(_MAX_RETRIES):
        try:
            transcript_list = client.list(video_id)
            try:
                transcript = transcript_list.find_transcript(list(_PREFERRED_LANGUAGES))
            except NoTranscriptFound:
                transcript = next(iter(transcript_list), None)
                if transcript is None:
                    return TranscriptResult(None, "not_found")
            fetched = transcript.fetch()
            text = _joined_text(fetched)
            return TranscriptResult(text or None, None if text else "empty")
        except TranscriptsDisabled:
            return TranscriptResult(None, "disabled")
        except VideoUnavailable:
            return TranscriptResult(None, "unavailable")
        except NoTranscriptFound:
            return TranscriptResult(None, "not_found")
        except _RETRYABLE:
            last_reason = "ip_blocked"
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            return TranscriptResult(None, last_reason)
        except Exception:
            return TranscriptResult(None, "error")

    return TranscriptResult(None, last_reason)


def get_transcript_text(
    video_id: str,
    *,
    api: YouTubeTranscriptApi | None = None,
) -> str | None:
    """Backward-compatible: transcript text or None."""
    return fetch_transcript(video_id, api=api).text
