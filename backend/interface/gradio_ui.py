
from functools import partial

import gradio as gr
from services.file_manager import default_file_manager
from services.font_manager import FontManager
from services.pipeline import generate_video


def process_video(input_video, font_manager: FontManager, progress=gr.Progress()):
    """
    Main function to be called by Gradio.
    Receives either a local video path or YouTube URL.
    """
    if not input_video:
        print("⚠️ Received empty input_video")
        return None
        
    print(f"🚀 START PROCESSING. Received input: {input_video}")
    
    # Cleanup old files (> 3 minutes)
    default_file_manager.cleanup_old_files(max_age_seconds=60*3)

    try:
        # Wrapper to ensure progress is sent correctly
        def progress_wrapper(p, desc=None):
            try:
                progress(p, desc=desc)
            except Exception:
                pass

        # Call our pipeline with explicit dependency injection
        output_path = generate_video(
            input_video,  # Can be file path or URL
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
        raise gr.Error(f"Internal backend error: {str(e)}")


def create_app(font_manager: FontManager):
    """
    Creates and returns the Gradio application instance.
    Now with YouTube URL support.
    """
    # Inject dependencies using partial
    process_with_deps = partial(process_video, font_manager=font_manager)

    # Define Gradio interface with combined input
    with gr.Blocks(title="Doresare Backend") as app:
        gr.Markdown("""
        # 🎸 Ukulele Chord Video Generator (Backend)
        
        Upload a video file or paste a YouTube URL to generate a video with chord overlays.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### File Upload")
                file_input = gr.File(label="Upload Video (MP4)", file_count="single")
            
            with gr.Column():
                gr.Markdown("### Or YouTube URL")
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )
        
        output_video = gr.File(label="Generated Video")
        submit_btn = gr.Button("Process Video", variant="primary")
        
        def process_input(file_input_val, url_input_val):
            """Determine which input to use and process it"""
            # Prioritize file upload if both are provided
            if file_input_val:
                print("Using uploaded file")
                return process_with_deps(file_input_val)
            elif url_input_val:
                print("Using YouTube URL")
                return process_with_deps(url_input_val)
            else:
                raise gr.Error("Please provide either a file or YouTube URL")
        
        submit_btn.click(
            fn=process_input,
            inputs=[file_input, url_input],
            outputs=output_video
        )
    
    app.queue()
    return app
    

