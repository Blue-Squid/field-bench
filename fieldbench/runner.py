"""Minimal TensorRT executor with per-inference GPU (CUDA event) and end-to-end timing."""
import ctypes
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


def _ck(ret):
    """Unwrap cuda-python's (err, *values) return convention."""
    err, *vals = ret
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return vals[0] if len(vals) == 1 else vals


class Tensor:
    def __init__(self, name, shape, dtype, is_input):
        self.name, self.shape, self.dtype, self.is_input = name, shape, dtype, is_input
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.device = _ck(cudart.cudaMalloc(self.nbytes))
        self.host_ptr = _ck(cudart.cudaMallocHost(self.nbytes))  # pinned, for async copies
        buf = (ctypes.c_byte * self.nbytes).from_address(self.host_ptr)
        self.host = np.frombuffer(buf, dtype=dtype).reshape(shape)

    def free(self):
        cudart.cudaFree(self.device)
        cudart.cudaFreeHost(self.host_ptr)


class TrtRunner:
    def __init__(self, engine_file):
        self.engine_file = engine_file  # clones (pipeline workers) deserialize their own copy
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_file, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = _ck(cudart.cudaStreamCreate())
        self.ev_start = _ck(cudart.cudaEventCreate())
        self.ev_end = _ck(cudart.cudaEventCreate())

        self.tensors = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError(f"{name} has a dynamic shape {shape}; export static ONNX")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            t = Tensor(name, shape, dtype, is_input)
            self.context.set_tensor_address(name, t.device)
            self.tensors.append(t)
        self.inputs = [t for t in self.tensors if t.is_input]
        self.outputs = [t for t in self.tensors if not t.is_input]

    @property
    def device_memory_bytes(self):
        """Activation/scratch memory the engine needs on top of its weights."""
        return getattr(self.engine, "device_memory_size_v2", self.engine.device_memory_size)

    def fill_random(self, seed=0):
        rng = np.random.default_rng(seed)
        for t in self.inputs:
            t.host[...] = rng.standard_normal(t.shape).astype(t.dtype)

    def infer(self, upload=True):
        """One synchronous inference. Returns (gpu_ms, end_to_end_ms).

        gpu_ms covers only enqueue-to-completion of the network on the stream;
        end_to_end_ms adds host->device and device->host copies and launch overhead.
        upload=False skips the host->device input copies, for inputs already written on the
        GPU (gpuprep).
        """
        h0 = time.perf_counter()
        for t in self.inputs if upload else ():
            _ck(cudart.cudaMemcpyAsync(t.device, t.host_ptr, t.nbytes, H2D, self.stream))
        _ck(cudart.cudaEventRecord(self.ev_start, self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        _ck(cudart.cudaEventRecord(self.ev_end, self.stream))
        for t in self.outputs:
            _ck(cudart.cudaMemcpyAsync(t.host_ptr, t.device, t.nbytes, D2H, self.stream))
        _ck(cudart.cudaStreamSynchronize(self.stream))
        h1 = time.perf_counter()
        gpu_ms = _ck(cudart.cudaEventElapsedTime(self.ev_start, self.ev_end))
        return gpu_ms, (h1 - h0) * 1e3

    def close(self):
        for t in self.tensors:
            t.free()
        cudart.cudaEventDestroy(self.ev_start)
        cudart.cudaEventDestroy(self.ev_end)
        cudart.cudaStreamDestroy(self.stream)
        del self.context, self.engine, self.runtime
