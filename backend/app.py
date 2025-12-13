import os
import sys

# Force unbuffered output for debugging in Docker
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path to allow absolute imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from interface.gradio_ui import create_app
from services import audio_extractor
from services.font_manager import FontManager

# Initialize services
font_manager = FontManager()

# Run startup checks to ensure environment is ready
print("🚀 Initializing backend services...")
try:
    font_manager.check_resources()
    audio_extractor.check_resources()
    
    # Load heavy AI models at startup
    audio_extractor.load_global_model()
    
    print("✅ All services ready.")
except Exception as e:
    print(f"❌ Startup check failed: {e}")
    # We continue even if checks fail, as services have fallbacks or will error gracefully later

# Create the app instance with injected dependencies
app = create_app(font_manager)


if __name__ == "__main__":
    # Launch server on port 7860 (HF Spaces standard)
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=share)
