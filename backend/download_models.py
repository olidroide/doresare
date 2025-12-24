import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_separator.separator import Separator

    from services.font_manager import FontManager

    print("⬇️ Pre-downloading resources for Docker image...")

    # 1. Download Fonts
    print("🎨 Checking/Downloading fonts...")
    fm = FontManager()
    fm.check_resources()

    model_dir = os.getenv("AUDIO_SEPARATOR_MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)

    # 2. Download Demucs Models (htdemucs_ft, htdemucs_6s)
    print("⬇️ Pre-downloading Demucs models...")
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["TORCH_HOME"] = os.path.abspath(model_dir)

    from demucs import pretrained

    for model_name in ["htdemucs_ft", "htdemucs_6s"]:
        print(f"⬇️ Downloading Demucs model: {model_name}")
        pretrained.get_model(model_name)
        print(f"✅ Demucs model {model_name} cached successfully!")

    # 3. Download ONNX Models
    print("⬇️ Pre-downloading ONNX models...")
    separator = Separator(log_level=logging.INFO, model_file_dir=model_dir)

    # Define models to pre-cache
    models_to_download = [
        os.getenv("AUDIO_SEPARATOR_MODEL", "UVR_MDXNET_KARA_2.onnx"),
        "UVR-MDX-NET-Inst_Main.onnx",
    ]

    for model_name in models_to_download:
        if not model_name.endswith(".onnx"):
            model_name += ".onnx"

        print(f"⬇️ Downloading model: {model_name}")
        separator.load_model(model_filename=model_name)
        print(f"✅ Model {model_name} cached successfully!")

    print("✅ All AI models downloaded and cached successfully!")

except Exception as e:
    print(f"❌ Error downloading models: {e}")
    sys.exit(1)
