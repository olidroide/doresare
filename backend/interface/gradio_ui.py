import gradio as gr
import os
import sys
import time

from functools import partial
from services.pipeline import generate_video
from services.file_manager import default_file_manager
from services.font_manager import FontManager

def process_video(input_video, font_manager: FontManager, progress=gr.Progress()):
    """
    Main function to be called by Gradio.
    Receives the path of the uploaded video, processes it, and returns the path of the generated video.
    """
    if not input_video:
        print("⚠️ Received empty input_video")
        return None
        
    print(f"🚀 START PROCESSING. Received video: {input_video}")
    
    # Cleanup old files (> 3 minutes)
    # Frontend downloads video immediately, so files can be cleaned up quickly
    default_file_manager.cleanup_old_files(max_age_seconds=60*3)

    try:
        # Wrapper to ensure progress is sent correctly
        def progress_wrapper(p, desc=None):
            try:
                progress(p, desc=desc)
            except Exception:
                pass

        # Call our pipeline with explicit dependency injection
        # The pipeline now handles output path generation
        output_path = generate_video(
            input_video, 
            file_manager=default_file_manager, 
            progress=progress_wrapper,
            font_manager=font_manager
        )
        
        print(f"✅ Video generated successfully: {output_path}")
        return output_path
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR in backend processing: {e}")
        import traceback
        traceback.print_exc()
        # In case of error, Gradio will show the message
        raise gr.Error(f"Internal backend error: {str(e)}")

def create_app(font_manager: FontManager):
    """
    Creates and returns the Gradio application instance.
    """
    # Inject dependencies using partial
    process_with_deps = partial(process_video, font_manager=font_manager)

    # Define Gradio interface
    iface = gr.Interface(
        fn=process_with_deps,
        inputs=gr.File(label="Upload Video (MP4)"),
        outputs=gr.File(label="Generated Video"),
        title="Ukulele Chord Video Generator (Backend)",
        description="Backend API to generate videos with ukulele chords. Internal use.",
        flagging_mode="never"
    )

    app = gr.TabbedInterface([iface], ["Generator"]).queue()
    return app

