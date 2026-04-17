from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript_text(video_id: str) -> str | None:
    """
    Fetch full transcript text for a video, or None if unavailable.
    """
    try:
        fetched = YouTubeTranscriptApi().fetch(
            video_id,
            languages=("en", "en-US", "en-GB"),
        )
        return " ".join(snippet.text for snippet in fetched)
    except Exception:
        return None
