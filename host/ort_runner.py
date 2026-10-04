"""ONNX Runtime stand-in for fieldbench.runner.TrtRunner, to check pipelines on the host."""
import time
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort


class OrtRunner:
    def __init__(self, onnx_path, providers=("CPUExecutionProvider",)):
        self.sess = ort.InferenceSession(str(onnx_path), providers=list(providers))
        self.inputs = [SimpleNamespace(name=i.name, host=np.zeros(i.shape, np.float32)) for i in self.sess.get_inputs()]
        self.outputs = [SimpleNamespace(name=o.name, host=None) for o in self.sess.get_outputs()]
        self.infer()  # allocate outputs

    def infer(self):
        t = time.perf_counter()
        ys = self.sess.run(None, {i.name: i.host for i in self.inputs})
        for o, y in zip(self.outputs, ys):
            if o.host is None:
                o.host = y
            else:
                o.host[...] = y
        ms = (time.perf_counter() - t) * 1e3
        return ms, ms

    def close(self):
        pass
