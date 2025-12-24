import os

from domain.models import VideoAnalysis
from services import audio_extractor, video_renderer
from services.bitwave_adapter import BitwaveAdapter, get_bitwave_analyzer
from services.chord_analyzer import ChordAnalyzer
from services.file_manager import FileManager
from services.font_manager import FontManager
from services.youtube_downloader import YouTubeDownloader

# Initialize Bitwave services (Singleton-like)
_bitwave_adapter = None
_chord_analyzer = None
_bitwave_analyzer = None


def get_bitwave_services():
    global _bitwave_adapter, _chord_analyzer, _bitwave_analyzer
    if _bitwave_adapter is None:
        _bitwave_adapter = BitwaveAdapter()
    if _chord_analyzer is None:
        _chord_analyzer = ChordAnalyzer()
    if _bitwave_analyzer is None:
        # Check if enabled via env
        enable = os.getenv("ENABLE_BITWAVE_ANALYSIS", "false").lower() == "true"
        _bitwave_analyzer = get_bitwave_analyzer(enable=enable)
    return _bitwave_adapter, _chord_analyzer, _bitwave_analyzer


def generate_video(
    input_source: str,
    file_manager: FileManager,
    font_manager: FontManager,
    progress=None,
    cleanup: bool = True,
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
    output_filename = file_manager.create_unique_filename(
        prefix="video", extension="mp4"
    )
    analysis.output_path = file_manager.get_output_path(
        output_filename, directory=work_dir
    )

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
            file_manager.create_unique_filename("extracted_audio", "wav"),
            directory=work_dir,
        )

        audio_file = audio_extractor.extract_audio_from_video(
            analysis.input_path, output_path=audio_path
        )

        if not audio_file:
            msg = "❌ Failed to extract audio"
            print(msg)
            raise Exception(msg)

        intermediate_files.append(audio_file)

        # 2. Bitwave Stem Separation & Chord Detection
        print("🧠 Separating audio with Bitwave (Bass/Other/Vocals/Drums)...")
        if progress:
            progress(0.2, desc="Separating audio with Bitwave AI...")

        import asyncio
        import threading

        bitwave_adapter, chord_analyzer, bitwave_adv = get_bitwave_services()

        bitwave_result = {"stems": None, "sr": None, "done": False, "error": None}

        def run_bitwave():
            try:
                # Create a new event loop for this thread to handle async Bitwave call
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                stems, sr = loop.run_until_complete(
                    bitwave_adapter.separate_audio(audio_file)
                )
                bitwave_result["stems"] = stems
                bitwave_result["sr"] = sr
            except Exception as e:
                print(f"❌ Exception in Bitwave thread: {e}")
                bitwave_result["error"] = e
            finally:
                bitwave_result["done"] = True

        bitwave_thread = threading.Thread(target=run_bitwave, daemon=True)
        bitwave_thread.start()

        # Wait for completion with progress updates
        elapsed = 0
        while not bitwave_result["done"]:
            import time

            time.sleep(0.5)
            elapsed += 0.5
            if progress:
                # Fake progress for now as Bitwave might not provide it easily
                p = 0.2 + min(0.19, elapsed / 60 * 0.2)
                progress(
                    p, desc=f"Separating audio with Bitwave AI... ({elapsed:.1f}s)"
                )

        bitwave_thread.join()

        if bitwave_result["error"]:
            raise bitwave_result["error"]

        stems = bitwave_result["stems"]
        sr = bitwave_result["sr"]

        print("🎸 Detecting chords from stems...")
        if progress:
            progress(0.4, desc="Detecting chords from stems...")

        chords = chord_analyzer.detect_chords_from_stems(stems, sr=sr)

        # 2.1 Advanced Bitwave analysis (Hybrid approach)
        if bitwave_adv and bitwave_adv.is_available:
            try:
                print(
                    "🔍 Performing advanced Bitwave analysis for hybrid verification..."
                )
                chords_adv = bitwave_adv.detect_chords(audio_file, sr=int(sr))
                print(f"📊 Bitwave Advanced detected {len(chords_adv)} chords.")
                # In the future, we can implement a merge function here.
                # For now, we prioritize the stem-based analysis which is more robust.
            except Exception as e:
                print(f"⚠️ Advanced Bitwave analysis failed: {e}")

        if not chords:
            msg = "⚠️ No chords detected."
            print(msg)
            raise Exception(msg)

        analysis.set_chords(chords)
        print(f"✅ Detected {len(analysis.chords)} chords using Bitwave.")

        # Set audio path for rendering
        analysis.set_audio(audio_file)

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
            print(
                f"🕐 Scheduled cleanup for work directory: {work_dir} (in 10 seconds)"
            )
