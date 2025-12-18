"""YouTube video downloader using yt-dlp with progress tracking."""
import logging
import os
import re
import socket
from pathlib import Path
from typing import Callable, Optional

import certifi
import yt_dlp

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Handle YouTube video downloads with progress tracking."""

    # Maximum quality: 720p
    MAX_HEIGHT = 720

    # Format selection: video + audio combined, prefer h264/aac for broad compatibility
    FORMAT_STRING = (
        f"bestvideo[height<=?{MAX_HEIGHT}][vcodec^=avc]"
        "+bestaudio[acodec^=aac]/"
        f"bestvideo[height<=?{MAX_HEIGHT}]"
        "+bestaudio/"
        f"best[height<=?{MAX_HEIGHT}]"
    )

    @staticmethod
    def validate_youtube_url(url: str) -> bool:
        """Validate if URL is a valid YouTube URL.

        Args:
            url: URL to validate

        Returns:
            True if valid YouTube URL, False otherwise
        """
        if not url:
            return False
        youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        return bool(re.match(youtube_regex, url))

    @staticmethod
    def download(
        url: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        timeout_seconds: int = 600,
    ) -> str:
        """Download YouTube video at max 720p resolution.

        Args:
            url: YouTube URL
            output_path: Directory where video will be saved
            progress_callback: Function(pct: float, detail: str) for progress updates
            timeout_seconds: Maximum time allowed for download (default 10 minutes)

        Returns:
            Path to downloaded video file

        Raises:
            ValueError: If URL is invalid
            Exception: If download fails
        """
        # Validate URL
        if not YouTubeDownloader.validate_youtube_url(url):
            raise ValueError(f"Invalid YouTube URL: {url}")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Setup progress tracking
        def progress_hook(d):
            """Progress hook for yt-dlp."""
            if progress_callback is None:
                return

            try:
                status = d.get("status")
                if status == "downloading":
                    total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate")
                    downloaded_bytes = d.get("downloaded_bytes", 0)

                    if total_bytes:
                        pct = downloaded_bytes / total_bytes
                    else:
                        pct = 0.0

                    speed = d.get("speed")
                    eta = d.get("eta")

                    speed_str = ""
                    if speed:
                        speed_mb = speed / (1024 * 1024)
                        speed_str = f" @ {speed_mb:.1f} MB/s"

                    eta_str = ""
                    if eta and eta > 0:
                        minutes, seconds = divmod(int(eta), 60)
                        if minutes > 0:
                            eta_str = f" (~{minutes}m {seconds}s)"
                        else:
                            eta_str = f" (~{seconds}s)"

                    detail = f"Downloading{speed_str}{eta_str}"
                    pct_display = int(pct * 100)
                    progress_callback(pct, f"{pct_display}% {detail}")

                elif status == "finished":
                    progress_callback(0.95, "Finalizing video...")

            except Exception as e:
                logger.warning(f"Error in progress hook: {e}")

        # Configure yt-dlp options
        ydl_opts = {
            "format": YouTubeDownloader.FORMAT_STRING,
            "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
            "progress_hooks": [progress_hook],
            # Network and Connectivity optimizations
            "source_address": "0.0.0.0",  # Force IPv4
            "geo_bypass": True,
            "nocheckcertificate": True,   # Disables SSL certificate verification
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Sec-Fetch-Mode": "navigate",
            },
            # Security and compatibility
            "restrictfilenames": True,
            "noplaylist": True,  # Don't download playlists, just the video
            "quiet": False,
            "no_warnings": False,
            "socket_timeout": timeout_seconds,
        }

        try:
            # Ensure Python/yt-dlp uses a valid CA bundle (fixes macOS cert issues)
            try:
                # Only set if not already configured in the environment
                if not os.environ.get("SSL_CERT_FILE"):
                    os.environ["SSL_CERT_FILE"] = certifi.where()
            except Exception:
                # Best-effort: if certifi isn't available or setting env fails, continue
                pass
            print(f"🎥 Downloading YouTube video: {url}")
            print(f"📁 Output directory: {output_path}")
            print(f"📺 Format: Max {YouTubeDownloader.MAX_HEIGHT}p")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Attempt to determine filename
                try:
                    filename = ydl.prepare_filename(info)
                except Exception:
                    filename = None

                video_path = None
                if filename and os.path.exists(filename):
                    video_path = filename

                if not video_path:
                    # Search for a recent video file in the output directory
                    video_files = [
                        f for f in os.listdir(output_path)
                        if f.endswith((".mp4", ".mkv", ".webm", ".mov", ".m4v"))
                    ]
                    if video_files:
                        video_files = sorted(video_files)
                        video_path = os.path.join(output_path, video_files[-1])

                if not video_path or not os.path.exists(video_path):
                    raise Exception("Downloaded file not found")

                print(f"✅ Video downloaded: {video_path}")
                print(f"📊 Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")

                if progress_callback:
                    progress_callback(1.0, "Download complete")

                return video_path

        except yt_dlp.utils.DownloadError as e:
            # Inspect the exception chain for name resolution / DNS errors
            cause = e
            dns_issue = False
            while cause:
                try:
                    if isinstance(cause, socket.gaierror):
                        dns_issue = True
                        break
                except Exception:
                    pass
                if "Failed to resolve" in str(cause) or "NameResolutionError" in str(cause):
                    dns_issue = True
                    break
                cause = getattr(cause, "__cause__", None)

            if dns_issue:
                enhanced = (
                    "YouTube download error: name resolution failed (DNS). "
                    "This environment may block outbound network/DNS lookups. "
                    "Workarounds: upload the video file via the UI, set `HTTP_PROXY`/`HTTPS_PROXY` in your Space settings, "
                    "or enable outbound network access for the Hugging Face Space."
                )
                print(f"❌ {enhanced}")
                raise Exception(enhanced) from e

            error_msg = f"YouTube download error: {e}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to download video: {e}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg) from e
