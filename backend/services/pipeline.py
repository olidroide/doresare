import os
from typing import Optional
from services import audio_extractor
from services import chord_detector
from services import video_renderer
from domain.models import VideoAnalysis
from services.font_manager import FontManager
from services.file_manager import FileManager

def generate_video(input_file: str, file_manager: FileManager, progress=None, cleanup: bool = True, font_manager: FontManager = None) -> str:
    """
    Orchestrates the video generation process using the VideoAnalysis aggregate.
    Returns the path to the generated video.
    
    Args:
        input_file: Path to the input video file
        file_manager: FileManager instance for file operations
        progress: Optional progress callback
        cleanup: If True, removes work directory after completion (default: True)
        font_manager: FontManager instance for font loading

    """
    # Enforce Dependency Injection
    if not file_manager:
        raise ValueError("file_manager is required")
    if not font_manager:
        raise ValueError("font_manager is required")
    


    if not os.path.exists(input_file):
        msg = f"❌ File not found: {input_file}"
        print(msg)
        raise FileNotFoundError(msg)
    
    # Create Aggregate Root (output_path will be set later)
    analysis = VideoAnalysis(input_path=input_file)
    
    # Create unique work directory for this pipeline run
    # This allows us to cleanup ALL files at the end
    import time
    unique_id = int(time.time())
    work_dir = os.path.join(file_manager.base_dir, f"process_{unique_id}")
    file_manager.ensure_directory(work_dir)

    # Generate Output Path in the work directory
    output_filename = file_manager.create_unique_filename(prefix="video", extension="mp4")
    analysis.output_path = file_manager.get_output_path(output_filename, directory=work_dir)

    print(f"🎬 Processing: {analysis.input_path}")
    print(f"📁 Work directory: {work_dir}")
    if progress: progress(0, desc="Starting...")

    # Track intermediate files for cleanup
    intermediate_files = []

    try:
        # 1. Extract audio
        print("🎵 Extracting audio...")
        if progress: progress(0.1, desc="Extracting audio...")
        
        # Use FileManager to create a temp path for audio
        audio_path = file_manager.get_output_path(file_manager.create_unique_filename("extracted_audio", "wav"), directory=work_dir)
        
        audio_file = audio_extractor.extract_audio_from_video(analysis.input_path, output_path=audio_path)
        
        if not audio_file:
            msg = "❌ Failed to extract audio"
            print(msg)
            raise Exception(msg)
        
        intermediate_files.append(audio_file)

        # 1.5 Audio separation (Optional but recommended)
        # Check if we should skip separation (e.g., on resource-constrained environments)
        skip_separation = os.getenv("SKIP_AUDIO_SEPARATION", "false").lower() == "true"
        separation_timeout = int(os.getenv("AUDIO_SEPARATION_TIMEOUT", "300"))  # Default 5 minutes
        
        if not skip_separation:
            print("🧠 Separating audio (Vocals/Instrumental)...")
            if progress: progress(0.2, desc="Separating audio with AI...")
            
            # Callback for fluid progress between 0.2 and 0.4
            def separation_progress(pct):
                if progress:
                    # Map 0.0-1.0 to 0.2-0.4
                    current = 0.2 + (pct * 0.2)
                    progress(current, desc=f"Separating audio with AI ({int(pct*100)}%)...")

            # Attempt separation with timeout
            stem_file = audio_extractor.separate_audio_ai(
                audio_file, 
                output_dir=work_dir,
                progress_callback=separation_progress,
                timeout_seconds=separation_timeout
            )
            
            if stem_file:
                print(f"✅ Using separated stem: {stem_file}")
                intermediate_files.append(stem_file)
                if "Instrumental" in stem_file:
                    vocals_file = stem_file.replace("Instrumental", "Vocals")
                    if os.path.exists(vocals_file):
                        intermediate_files.append(vocals_file)
                
                # Update domain entity with the best audio source
                analysis.set_audio(stem_file)
            else:
                print("⚠️ Using original audio (separation failed or timed out)")
                analysis.set_audio(audio_file)
        else:
            print("⚠️ Audio separation skipped (SKIP_AUDIO_SEPARATION=true)")
            analysis.set_audio(audio_file)

        # 2. Detect chords
        print(f"🎸 Detecting chords...")
        if progress: progress(0.4, desc=f"Detecting chords...")
        
        # We use the audio stored in the analysis entity
        if not analysis.audio_path:
             raise Exception("Audio path missing in analysis")

        # Route to appropriate detection method
        chords = chord_detector.detect_chords_chroma_improved(analysis.audio_path)
        
        if not chords:
            msg = "⚠️ No chords detected."
            print(msg)
            raise Exception(msg)

        # Update domain entity
        analysis.set_chords(chords)
        print(f"✅ Detected {len(analysis.chords)} chords.")

        # 3. Generate video
        print("🎥 Generating video with overlays...")
        if progress: progress(0.6, desc="Rendering video...")
        
        video_renderer.render_video_with_overlays(
            analysis, # Pass the whole aggregate
            progress=progress,
            start_pct=0.6,
            end_pct=1.0,
            file_manager=file_manager,
            work_dir=work_dir,
            font_manager=font_manager
        )

        
        # Final verification
        analysis.complete()
            
        print("✨ Process completed!")
        if progress: progress(1.0, desc="Completed!")
        
        return analysis.output_path

    except Exception as e:
        print(f"❌ Error in generate_video: {e}")
        analysis.fail(str(e))
        raise e

    finally:
        if cleanup:
            # Schedule asynchronous cleanup
            # We CANNOT cleanup in the finally block because Gradio needs the file
            # for postprocessing AFTER the function returns.
            # Instead, we schedule cleanup in a background thread with a delay.
            import threading
            import shutil
            
            def delayed_cleanup():
                import time
                # Wait longer to ensure Gradio has finished postprocessing and serving
                time.sleep(10)  # 10 seconds should be more than enough
                
                if os.path.exists(work_dir):
                    try:
                        shutil.rmtree(work_dir)
                        print(f"✅ [Background] Removed work directory: {work_dir}")
                    except Exception as e:
                        print(f"⚠️ [Background] Could not remove work directory: {e}")
            
            # Start cleanup in background thread
            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()
            print(f"🕐 Scheduled cleanup for work directory: {work_dir} (in 10 seconds)")

