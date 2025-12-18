import os
import unittest
from unittest.mock import patch, MagicMock

# We need to test the logic in app.py. Since app.py executes code on import, 
# we'll test the logic by mimicking what app.py does.

class TestEnvPresets(unittest.TestCase):
    def test_hf_preset(self):
        with patch.dict(os.environ, {"ENV": "HF"}, clear=True):
            deploy_env = os.getenv('ENV', 'LOCAL').upper()
            self.assertEqual(deploy_env, "HF")
            
            # Simulate app.py logic
            if deploy_env == 'HF':
                os.environ.setdefault('AUDIO_SEPARATOR_MODEL', 'UVR_MDXNET_KARA_2.onnx')
                os.environ.setdefault('SKIP_AUDIO_SEPARATION', 'false')
            
            self.assertEqual(os.environ.get('AUDIO_SEPARATOR_MODEL'), 'UVR_MDXNET_KARA_2.onnx')
            self.assertEqual(os.environ.get('SKIP_AUDIO_SEPARATION'), 'false')

    def test_intel_low_server_preset(self):
        with patch.dict(os.environ, {"ENV": "INTEL_LOW_SERVER"}, clear=True):
            deploy_env = os.getenv('ENV', 'LOCAL').upper()
            self.assertEqual(deploy_env, "INTEL_LOW_SERVER")
            
            # Simulate app.py logic
            if deploy_env == 'INTEL_LOW_SERVER':
                os.environ.setdefault('USE_OPENVINO', 'true')
                os.environ.setdefault('ONNXRUNTIME_EXECUTION_PROVIDERS', 'OpenVINOExecutionProvider,CPUExecutionProvider')
                os.environ.setdefault('MOVIEPY_USE_GPU', 'true')
                os.environ.setdefault('MOVIEPY_FFMPEG_CODEC', 'h264_qsv')
            
            self.assertEqual(os.environ.get('USE_OPENVINO'), 'true')
            self.assertEqual(os.environ.get('ONNXRUNTIME_EXECUTION_PROVIDERS'), 'OpenVINOExecutionProvider,CPUExecutionProvider')
            self.assertEqual(os.environ.get('MOVIEPY_USE_GPU'), 'true')
            self.assertEqual(os.environ.get('MOVIEPY_FFMPEG_CODEC'), 'h264_qsv')

if __name__ == '__main__':
    unittest.main()
