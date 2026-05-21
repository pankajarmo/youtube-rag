from src.transcripts import is_youtube_video_id
from src.youtube_channel import normalize_listing_url


def test_normalize_handle_appends_videos():
    assert (
        normalize_listing_url("https://www.youtube.com/@MrBeast")
        == "https://www.youtube.com/@MrBeast/videos"
    )


def test_normalize_handle_trailing_slash():
    assert (
        normalize_listing_url("https://www.youtube.com/@MrBeast/")
        == "https://www.youtube.com/@MrBeast/videos"
    )


def test_normalize_channel_uc_appends_videos():
    assert normalize_listing_url(
        "https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA"
    ) == ("https://www.youtube.com/channel/UCX6OQ3DkcsbYNE6H8uQQuVA/videos")


def test_normalize_leaves_playlist_urls():
    url = "https://www.youtube.com/playlist?list=PLabc123"
    assert normalize_listing_url(url) == url


def test_is_youtube_video_id():
    assert is_youtube_video_id("AaMdXZMvT3w")
    assert not is_youtube_video_id("UCX6OQ3DkcsbYNE6H8uQQuVA")
    assert not is_youtube_video_id("")
