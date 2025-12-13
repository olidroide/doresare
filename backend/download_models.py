import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from services.font_manager import FontManager
    from audio_separator.separator import Separator
    
    print("⬇️ Pre-downloading resources for Docker image...")

    # 1. Download Fonts
    print("🎨 Checking/Downloading fonts...")
    fm = FontManager()
    fm.check_resources()
    
    print("⬇️ Pre-downloading AI models...")
    
    # Initialize separator
    # This will download the model to the valid cache directory
    model_dir = os.getenv('AUDIO_SEPARATOR_MODEL_DIR', '/home/user/models')
    os.makedirs(model_dir, exist_ok=True)
    
    separator = Separator(log_level=logging.INFO, model_file_dir=model_dir)
    
    # Load the specific model used in the pipeline
    # Using HQ_3 model by default for better performance on low-power CPUs
    # Can be overridden with AUDIO_SEPARATOR_MODEL environment variable
    model_name = os.getenv('AUDIO_SEPARATOR_MODEL', 'UVR-MDX-NET-Inst_HQ_3.onnx')
    print(f"⬇️ Downloading model: {model_name}")
    separator.load_model(model_filename=model_name)
    
    print(f"✅ Model {model_name} downloaded and cached successfully!")
    
except Exception as e:
    print(f"❌ Error downloading models: {e}")
    sys.exit(1)
