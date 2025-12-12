import importlib
import sys
from pathlib import Path

import numpy as np


def make_fake_openvino(monkeypatch):
    class FakeInferRequest:
        def infer(self, _inputs):
            # Return a dict with two outputs shaped like input
            # Grab input tensor
            for v in _inputs.values():
                arr = np.asarray(v)
                break
            out = arr.copy()
            # return vocals and instrumental as zeros and original
            return {"vocals_out": np.zeros_like(out).ravel(), "inst_out": out.ravel()}

    class FakeCompiled:
        def create_infer_request(self):
            return FakeInferRequest()

    class FakeCore:
        def read_model(self, model):
            return object()

        def compile_model(self, model, device, config):
            return FakeCompiled()

    fake_mod = type(sys)('openvino')
    runtime = type(sys)('openvino.runtime')
    runtime.Core = lambda *a, **k: FakeCore()
    fake_mod.runtime = runtime
    monkeypatch.setitem(sys.modules, 'openvino', fake_mod)
    monkeypatch.setitem(sys.modules, 'openvino.runtime', runtime)


def test_openvino_separator_with_fake_runtime(tmp_path, monkeypatch):
    make_fake_openvino(monkeypatch)

    # Create a short sine audio file
    import soundfile as sf
    sr = 44100
    t = np.linspace(0, 0.1, int(sr * 0.1), False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio_file = tmp_path / "tone.wav"
    sf.write(str(audio_file), tone, sr)

    # Import separator and run
    mod = importlib.import_module('backend.infrastructure.audio_separation.openvino_separator')
    sep = mod.OpenVINOAudioSeparator(model_path='fake.onnx')
    out = sep.separate(str(audio_file), chunk_duration=1)

    assert 'vocals' in out and 'instrumental' in out
    assert Path(out['vocals']).exists()
    assert Path(out['instrumental']).exists()
