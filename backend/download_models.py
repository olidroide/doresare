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
    
    # Define models to pre-cache
    # 1. KARA_2 (Lightweight, for low-end devices)
    # 2. Inst_Main (High-quality, for rock/acoustic on HF)
    models_to_download = [
        os.getenv('AUDIO_SEPARATOR_MODEL', 'UVR_MDXNET_KARA_2.onnx'),
        'UVR-MDX-NET-Inst_Main.onnx'
    ]

    for model_name in models_to_download:
        if not model_name.endswith('.onnx'):
            model_name += '.onnx'
            
        print(f"⬇️ Downloading model: {model_name}")
        separator.load_model(model_filename=model_name)
        print(f"✅ Model {model_name} cached successfully!")
    
    print("✅ All AI models downloaded and cached successfully!")
    
except Exception as e:
    print(f"❌ Error downloading models: {e}")
    sys.exit(1)
