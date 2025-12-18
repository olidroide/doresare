import os
import sys

# Force unbuffered output for debugging in Docker
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path to allow absolute imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from interface.gradio_ui import create_app
from services import audio_extractor
from services.font_manager import FontManager

def check_connectivity():
    """Diagnostic check for network connectivity, specifically for YouTube."""
    import socket
    target = "www.youtube.com"
    print(f"🌐 Connectivity check: Resolving '{target}'...")
    try:
        addr = socket.gethostbyname(target)
        print(f"✅ DNS Success: '{target}' resolved to {addr}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS Failure: Could not resolve '{target}': {e}")
        return False
    except Exception as e:
        print(f"❌ Connectivity Error: {e}")
        return False

# Initialize services
font_manager = FontManager()

# Run startup checks to ensure environment is ready
print("🚀 Initializing backend services...")
check_connectivity()
try:
    font_manager.check_resources()
    audio_extractor.check_resources()
    # Detect deployment environment via `ENV`.
    deploy_env = os.getenv('ENV', 'LOCAL').upper()
    print(f"🔎 Deployment environment: '{deploy_env}'")

    if deploy_env == 'HF':
        # Hugging Face: prefer lightweight model, don't preload heavy global models
        print("ℹ️ HF preset: prefer lightweight separator model and skip global preload.")
        os.environ.setdefault('AUDIO_SEPARATOR_MODEL', 'UVR_MDXNET_KARA_2.onnx')
        os.environ.setdefault('SKIP_AUDIO_SEPARATION', 'false')
        # On HF we usually don't want to preload if resources are tight
        
    elif deploy_env == 'INTEL_LOW_SERVER':
        # Intel low-end server: enable OpenVINO, preload model and allow GPU usage
        print("🖥️ INTEL_LOW_SERVER preset: enabling OpenVINO, preferring iGPU, preloading models.")
        # Inform audio_extractor and downstream libs
        os.environ.setdefault('USE_OPENVINO', 'true')
        # Request OpenVINO provider for ONNX Runtime (fallback to CPU provider is fine)
        os.environ.setdefault('ONNXRUNTIME_EXECUTION_PROVIDERS', 'OpenVINOExecutionProvider,CPUExecutionProvider')
        os.environ.setdefault('SKIP_AUDIO_SEPARATION', 'false')
        os.environ.setdefault('AUDIO_SEPARATOR_MODEL', 'UVR_MDXNET_KARA_2.onnx')
        
        # Video rendering presets for Intel J3455
        os.environ.setdefault('MOVIEPY_USE_GPU', 'true')
        os.environ.setdefault('MOVIEPY_FFMPEG_CODEC', 'h264_qsv')
        
        # Try to preload heavy model at startup to speed per-request inference
        try:
            audio_extractor.load_global_model()
        except Exception as e:
            print(f"⚠️ Preload failed in INTEL_LOW_SERVER preset: {e}")

    else:
        # Local/default: keep current behavior (preload for performance)
        audio_extractor.load_global_model()

    print(f"✅ All services ready for {deploy_env} mode.")
except Exception as e:
    print(f"❌ Startup check failed: {e}")
    # We continue even if checks fail, as services have fallbacks or will error gracefully later

# Create the app instance with injected dependencies
app = create_app(font_manager)


if __name__ == "__main__":
    # Launch server on port 7860 (HF Spaces standard)
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=share)
