"""
YouTube transcript downloader.

Pulls transcripts from YouTube videos and saves them as markdown files
in a speaker's corpus directory.

Usage:
    python -m src.corpus.youtube --speaker jane_smith --url "https://youtube.com/watch?v=..."
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # bare video ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def fetch_transcript(video_id: str) -> str:
    """Fetch and format a YouTube transcript."""
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    # Join all snippet texts into flowing prose
    lines = [snippet.text for snippet in transcript]
    return " ".join(lines)


def save_transcript(
    speaker_id: str,
    video_id: str,
    transcript: str,
    title: str | None = None,
    category: str = "speeches",
) -> Path:
    """Save a transcript as a markdown file in the speaker's corpus."""
    speaker_dir = Path(f"data/speakers/{speaker_id}/{category}")
    speaker_dir.mkdir(parents=True, exist_ok=True)

    filename = f"youtube_{video_id}.md"
    filepath = speaker_dir / filename

    header = f"# {title or f'YouTube Video {video_id}'}\n\n"
    header += f"- **Source**: https://youtube.com/watch?v={video_id}\n"
    header += f"- **Type**: YouTube transcript\n\n---\n\n"

    filepath.write_text(header + transcript, encoding="utf-8")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Download YouTube transcript for a speaker")
    parser.add_argument("--speaker", required=True, help="Speaker ID (folder name)")
    parser.add_argument("--url", required=True, help="YouTube video URL or ID")
    parser.add_argument("--title", help="Optional title for the transcript")
    parser.add_argument(
        "--category",
        default="speeches",
        choices=["speeches", "interviews", "writings"],
        help="Which subfolder to save to",
    )
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    print(f"📥 Fetching transcript for {video_id}...")

    transcript = fetch_transcript(video_id)
    print(f"   Got {len(transcript.split())} words")

    filepath = save_transcript(
        speaker_id=args.speaker,
        video_id=video_id,
        transcript=transcript,
        title=args.title,
        category=args.category,
    )
    print(f"   ✅ Saved to {filepath}")


if __name__ == "__main__":
    main()

