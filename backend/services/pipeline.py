import os
from typing import Optional

from domain.models import VideoAnalysis
from services import audio_extractor, chord_detector, video_renderer
from services.file_manager import FileManager
from services.font_manager import FontManager
from services.youtube_downloader import YouTubeDownloader


def generate_video(
    input_source: str,
    file_manager: FileManager,
    progress=None,
    cleanup: bool = True,
    font_manager: FontManager = None,
) -> str:
    """
    Orchestrates the video generation process using the VideoAnalysis aggregate.
    Supports local file paths or YouTube URLs as `input_source`.

    Args:
        input_source: Path to local video file OR YouTube URL
        file_manager: FileManager instance for file operations
        progress: Optional progress callback
        cleanup: If True, removes work directory after completion (default: True)
        font_manager: FontManager instance for font loading

    Returns:
        Path to generated video
    """
    # Enforce Dependency Injection
    if not file_manager:
        raise ValueError("file_manager is required")
    if not font_manager:
        raise ValueError("font_manager is required")

    # Create unique work directory for this pipeline run
    import time
    unique_id = int(time.time())
    work_dir = os.path.join(file_manager.base_dir, f"process_{unique_id}")
    file_manager.ensure_directory(work_dir)

    # Detect if input is a YouTube URL
    is_youtube_url = YouTubeDownloader.validate_youtube_url(input_source)

    # Handle YouTube download if needed
    if is_youtube_url:
        print(f"🌐 YouTube URL detected: {input_source}")
        if progress:
            progress(0.05, desc="Downloading YouTube video...")

        try:
            def download_progress(pct, detail):
                """Wrapper for download progress"""
                if progress:
                    # Map [0.0, 1.0] to pipeline range [0.05, 0.15]
                    pipeline_pct = 0.05 + (pct * 0.1)
                    progress(pipeline_pct, desc=f"⬇️ {detail}")

            input_file = YouTubeDownloader.download(
                input_source,
                output_path=work_dir,
                progress_callback=download_progress,
                timeout_seconds=600,
            )
            print(f"✅ Video downloaded and ready for processing")

        except Exception as e:
            error_msg = f"Failed to download YouTube video: {e}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    else:
        # Local file
        input_file = input_source

    if not os.path.exists(input_file):
        msg = f"❌ File not found: {input_file}"
        print(msg)
        raise FileNotFoundError(msg)

    # Create Aggregate Root (output_path will be set later)
    analysis = VideoAnalysis(input_path=input_file)

    # Generate Output Path in the work directory
    output_filename = file_manager.create_unique_filename(prefix="video", extension="mp4")
    analysis.output_path = file_manager.get_output_path(output_filename, directory=work_dir)

    print(f"🎬 Processing: {analysis.input_path}")
    print(f"📁 Work directory: {work_dir}")
    if progress:
        progress(0, desc="Starting...")

    # Track intermediate files for cleanup
    intermediate_files = []

    try:
        # 1. Extract audio
        print("🎵 Extracting audio...")
        if progress:
            progress(0.1, desc="Extracting audio...")

        # Use FileManager to create a temp path for audio
        audio_path = file_manager.get_output_path(
            file_manager.create_unique_filename("extracted_audio", "wav"), directory=work_dir
        )

        audio_file = audio_extractor.extract_audio_from_video(analysis.input_path, output_path=audio_path)

        if not audio_file:
            msg = "❌ Failed to extract audio"
            print(msg)
            raise Exception(msg)

        intermediate_files.append(audio_file)

        # 1.5 Audio separation (Optional but recommended)
        skip_separation = os.getenv("SKIP_AUDIO_SEPARATION", "false").lower() == "true"
        separation_timeout = int(os.getenv("AUDIO_SEPARATION_TIMEOUT", "300"))

        if not skip_separation:
            print("🧠 Separating audio (Vocals/Instrumental)...")
            if progress:
                progress(0.2, desc="Separating audio with AI...")

            # Start separation in background with a shared state
            import threading

            separation_result = {"stem_file": None, "done": False}
            shared_progress = {"pct": 0.0, "detail": ""}

            def sep_progress_callback(pct, detail=""):
                shared_progress["pct"] = pct
                if detail:
                    shared_progress["detail"] = detail

            def run_separation():
                try:
                    stem_file = audio_extractor.separate_audio_ai(
                        audio_file,
                        output_dir=work_dir,
                        progress_callback=sep_progress_callback,
                        timeout_seconds=separation_timeout,
                    )
                    separation_result["stem_file"] = stem_file
                except Exception as e:
                    print(f"❌ Exception in separation thread: {e}")
                    separation_result["stem_file"] = None
                finally:
                    separation_result["done"] = True

            sep_thread = threading.Thread(target=run_separation, daemon=True)
            sep_thread.start()

            elapsed = 0
            poll_interval = 0.5
            max_wait = separation_timeout
            last_log_time = 0

            while not separation_result["done"] and elapsed < max_wait:
                import time

                time.sleep(poll_interval)
                elapsed += poll_interval

                if elapsed - last_log_time >= 10:
                    print(f"⏳ Audio separation in progress... {elapsed:.1f}s elapsed", flush=True)
                    last_log_time = elapsed

                current_sep_progress = shared_progress["pct"]
                detail_str = shared_progress["detail"]

                if progress:
                    pipeline_progress = 0.2 + (current_sep_progress * 0.2)
                    pipeline_progress = min(0.39, pipeline_progress)
                    pct_display = int(current_sep_progress * 100)
                    
                    # Format a professional status string for SSE
                    status_base = "Separating audio with AI"
                    if detail_str:
                         desc_str = f"{status_base} | {detail_str}"
                    else:
                         desc_str = f"{status_base} ({pct_display}%)"
                         
                    progress(pipeline_progress, desc=desc_str)

            sep_thread.join(timeout=1)

            stem_file = separation_result["stem_file"]

            if stem_file:
                print(f"✅ Using separated stem: {stem_file}", flush=True)
                intermediate_files.append(stem_file)
                if "Instrumental" in stem_file:
                    vocals_file = stem_file.replace("Instrumental", "Vocals")
                    if os.path.exists(vocals_file):
                        intermediate_files.append(vocals_file)
                analysis.set_audio(stem_file)
            else:
                print("⚠️ Using original audio (separation failed or timed out)")
                analysis.set_audio(audio_file)
        else:
            print("⚠️ Audio separation skipped (SKIP_AUDIO_SEPARATION=true)")
            analysis.set_audio(audio_file)

        # 2. Detect chords
        print(f"🎸 Detecting chords...")
        if progress:
            progress(0.4, desc=f"Detecting chords...")

        if not analysis.audio_path:
            raise Exception("Audio path missing in analysis")

        chords = chord_detector.detect_chords_chroma_improved(analysis.audio_path)

        if not chords:
            msg = "⚠️ No chords detected."
            print(msg)
            raise Exception(msg)

        analysis.set_chords(chords)
        print(f"✅ Detected {len(analysis.chords)} chords.")

        # 3. Generate video
        print("🎥 Generating video with overlays...")
        if progress:
            progress(0.6, desc="Rendering video...")

        video_renderer.render_video_with_overlays(
            analysis,
            progress=progress,
            start_pct=0.6,
            end_pct=1.0,
            file_manager=file_manager,
            work_dir=work_dir,
            font_manager=font_manager,
        )

        analysis.complete()

        print("✨ Process completed!")
        if progress:
            progress(1.0, desc="Completed!")

        return analysis.output_path

    except Exception as e:
        print(f"❌ Error in generate_video: {e}")
        analysis.fail(str(e))
        raise e

    finally:
        if cleanup:
            # Schedule asynchronous cleanup
            import shutil
            import threading

            def delayed_cleanup():
                import time

                time.sleep(10)

                if os.path.exists(work_dir):
                    try:
                        shutil.rmtree(work_dir)
                        print(f"✅ [Background] Removed work directory: {work_dir}")
                    except Exception as e:
                        print(f"⚠️ [Background] Could not remove work directory: {e}")

            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()
            print(f"🕐 Scheduled cleanup for work directory: {work_dir} (in 10 seconds)")

