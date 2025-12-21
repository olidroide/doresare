import asyncio
import glob
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

try:
    import certifi
except ImportError:
    certifi = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Base exception for all download-related errors."""


class RequiresFFmpegError(DownloadError):
    """Raised when yt-dlp requires FFmpeg but it is not available."""


class VideoDownloader:
    """Robust video downloader optimized for YouTube, mirroring backend patterns.
    
    Ensures high compatibility, reliable SSL handling, and guaranteed video output.
    """

    def __init__(self, max_height: int = 720):
        self.max_height = max_height
        # Robust format selection:
        # 1. Prefer AVC (H.264) + AAC for maximum compatibility.
        # 2. Fallback to any high-quality video/audio pair.
        # 3. Last resort: any format that definitely contains video [vcodec!=none].
        self.format_string = (
            f"bestvideo[height<=?{max_height}][vcodec^=avc]+bestaudio[acodec^=aac]/"
            f"bestvideo[height<=?{max_height}]+bestaudio/"
            f"best[height<=?{max_height}][vcodec!=none]/"
            "best[vcodec!=none]"
        )

        # Ensure SSL certificates are available
        if certifi and not os.environ.get("SSL_CERT_FILE"):
            os.environ["SSL_CERT_FILE"] = certifi.where()

    def _get_opts(self, outtmpl: str, hook: Any) -> dict[str, Any]:
        return {
            "format": self.format_string,
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,  # Silences the PO Token and SABR warnings
            "progress_hooks": [hook],
            "nocheckcertificate": True,
            "restrictfilenames": True,
            "merge_output_format": "mp4",
            "source_address": "0.0.0.0",
            "geo_bypass": True,
            "socket_timeout": 600,
            # Balanced options: allow remote scripts for signatures, 
            # and use clients that are proven to work for this video.
            "allow_remote_scripts": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web", "tv"],
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Sec-Fetch-Mode": "navigate",
            },
        }

    def _download_sync(self, url: str, outtmpl: str, hook: Any, temp_base: str) -> Path:
        if yt_dlp is None:
            raise DownloadError("yt-dlp is not available")

        opts = self._get_opts(outtmpl, hook)
        print(f"📥 Starting YouTube download: {url}")
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # We execute download directly. Redundant extract_info(download=False) calls
                # often trigger false-positive "format not available" errors.
                info = ydl.extract_info(url, download=True)
                
                # Resolve the final filename (handling potential merges to mp4/webm/mkv)
                try:
                    filename = ydl.prepare_filename(info)
                    base, _ = os.path.splitext(filename)
                    # Check for common extensions in case of merger or conversion
                    for ext in [".mp4", ".mkv", ".webm", ".mov", ".m4v"]:
                        if os.path.exists(base + ext):
                            filename = base + ext
                            break
                except Exception:
                    filename = None

                if filename and os.path.exists(filename):
                    video_path = Path(os.path.abspath(filename))
                else:
                    # Fallback to globbing if prepare_filename fails
                    matches = glob.glob(temp_base + ".*")
                    if not matches:
                        raise DownloadError("Downloaded file not found on disk.")
                    
                    # Prioritize common video extensions
                    video_exts = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
                    v_matches = [m for m in matches if Path(m).suffix.lower() in video_exts]
                    res = v_matches[0] if v_matches else matches[0]
                    video_path = Path(os.path.abspath(res))

                # Final validation: the backend REQUIRES video content
                ext = video_path.suffix.lower()
                if ext in [".m4a", ".mp3", ".wav", ".aac"]:
                    raise DownloadError(f"Format mismatch: downloaded audio ({ext}) instead of video.")

                print(f"✅ Download complete: {video_path.name} ({os.path.getsize(video_path) // 1024} KB)")
                return video_path

        except Exception as e:
            print(f"❌ Download failed: {e}")
            msg = str(e).lower()
            if "ffmpeg" in msg or "ffprobe" in msg:
                raise RequiresFFmpegError(str(e)) from e
            raise DownloadError(str(e)) from e

    async def download_stream(self, url: str) -> AsyncGenerator[dict | int | Path, None]:
        """Asynchronous wrapper for the download process with progress tracking."""
        loop = asyncio.get_running_loop()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | int] = asyncio.Queue()

        def progress_hook(d: dict[str, Any]) -> None:
            try:
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate")
                    downloaded = d.get("downloaded_bytes", 0)
                    if total:
                        pct = float(downloaded * 100 / total)
                        
                        # Create rich info
                        info = {
                            "pct": pct,
                            "downloaded": downloaded,
                            "total": total,
                            "speed": d.get("speed"),
                            "eta": d.get("eta")
                        }
                        loop.call_soon_threadsafe(queue.put_nowait, info)
                elif d.get("status") == "finished":
                    loop.call_soon_threadsafe(queue.put_nowait, 100)
            except Exception:
                pass

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_base = tmp.name
        outtmpl = f"{temp_base}.%(ext)s"

        download_task = loop.run_in_executor(
            None, self._download_sync, url, outtmpl, progress_hook, temp_base
        )

        last_pct = -1
        while not download_task.done():
            try:
                pct = await asyncio.wait_for(queue.get(), timeout=0.2)
                if isinstance(pct, (int, float)):
                    if pct > last_pct:
                        yield pct
                        last_pct = pct
                elif isinstance(pct, dict):
                    # Always yield dict updates
                    yield pct
                    last_pct = pct["pct"]
            except asyncio.TimeoutError:
                continue
            except Exception: # Catch any other unexpected errors in the queue processing
                break

        result_path = await download_task
        yield result_path


async def download_video_stream(
    url: str, max_height: int = 720
) -> AsyncGenerator[dict | int | Path, None]:
    """Shortcut function to use the VideoDownloader service."""
    downloader = VideoDownloader(max_height=max_height)
    async for item in downloader.download_stream(url):
        yield item
