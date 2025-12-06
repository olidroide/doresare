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
    # This will download the model to the default cache directory
    separator = Separator(log_level=logging.INFO)
    
    # Load the specific model used in the pipeline
    # Using 'Main' model by default for better performance on home servers
    model_name = 'UVR-MDX-NET-Inst_Main.onnx'
    print(f"⬇️ Downloading model: {model_name}")
    separator.load_model(model_filename=model_name)
    
    print(f"✅ Model {model_name} downloaded and cached successfully!")
    
except Exception as e:
    print(f"❌ Error downloading models: {e}")
    sys.exit(1)
