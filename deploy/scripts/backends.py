"""Inference backends for the exported leg policy, auto-detected, all with the same call: `run(x[1, obs]) -> y[1, act]`.

Used by bench_policy.py, check_parity.py and verify_policy_pack.py (docs/plan/jetson-inference.md). The robot runtime
itself (humanoid_arc_bme/onboard/runtime/policies.py) still builds its own ORT CPU session; nothing here changes it.

Backends (name -> what it is):
  ort-cpu   onnxruntime CPUExecutionProvider, `threads` intra-op threads, sequential execution (always, if ORT is installed)
  ort-cuda  onnxruntime CUDAExecutionProvider (only with an onnxruntime-gpu build that has it, e.g. Jetson AI Lab jp6/cu126)
  ort-trt   onnxruntime TensorrtExecutionProvider (same build; engine cache in `cache_dir`)
  trt       TensorRT python API on a serialized engine from build_engine.sh (needs `tensorrt` 10.7.x + cuda-python)

The TensorRT version rule (exactly 10.7 on the Jetson, docs/plan/vendor/gr00t-wbc-g1-jetpack6.md) is enforced here for
`trt` and `ort-trt` by `require_trt_version`; there is no override switch on purpose.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

REQUIRED_TRT = "10.7"  # major.minor; any 10.7.x.y is accepted, nothing else


class BackendUnavailable(RuntimeError):
    pass


def trt_version_ok(version: str, required: str = REQUIRED_TRT) -> bool:
    """'10.7.0.23' -> True for required '10.7'; '10.3.0.30', '10.13.3', '10.70.1' -> False."""
    parts, req = str(version).strip().split("."), required.split(".")
    return len(parts) >= len(req) and parts[: len(req)] == req


def require_trt_version(version: str) -> None:
    if not trt_version_ok(version):
        raise BackendUnavailable(f"TensorRT {version} refused: exactly {REQUIRED_TRT}.x is required on the robot "
                                 "(docs/plan/jetson-inference.md, TensorRT version rule)")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise BackendUnavailable("onnxruntime not installed") from e
    return ort


def _tensorrt():
    try:
        import tensorrt as trt
    except ImportError as e:
        raise BackendUnavailable("tensorrt python package not installed") from e
    return trt


def detect(engine: str | None = None) -> list[str]:
    """Backends that can be constructed here (engine: path for `trt`)."""
    out = []
    try:
        provs = _ort().get_available_providers()
        out.append("ort-cpu")
        if "CUDAExecutionProvider" in provs:
            out.append("ort-cuda")
        if "TensorrtExecutionProvider" in provs:
            out.append("ort-trt")
    except BackendUnavailable:
        pass
    if engine and Path(engine).exists():
        try:
            _tensorrt()
            out.append("trt")
        except BackendUnavailable:
            pass
    return out


def versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()}
    try:
        ort = _ort()
        v["onnxruntime"] = ort.__version__
        v["ort_providers"] = ort.get_available_providers()
    except BackendUnavailable:
        v["onnxruntime"] = None
    try:
        v["tensorrt"] = _tensorrt().__version__
    except BackendUnavailable:
        v["tensorrt"] = None
    return v


class OrtRunner:
    def __init__(self, onnx_path, providers, threads: int = 1, provider_options=None, name="ort-cpu"):
        ort = _ort()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads == 1:
            # no spinning worker pool: a 50 Hz loop sleeps between calls, spin threads only burn a core
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers,
                                         provider_options=provider_options)
        inp, out = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name
        self.input_shape, self.output_shape = list(inp.shape), list(out.shape)
        self.name = name
        self.info = {"backend": name, "providers_used": self.sess.get_providers(), "threads": threads}
        # a fallback to CPU would silently measure the wrong thing
        if name != "ort-cpu" and self.sess.get_providers()[0] == "CPUExecutionProvider":
            raise BackendUnavailable(f"{name}: session fell back to CPU")

    def run(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.output_name], {self.input_name: x})[0]

    # the humanoid_arc_bme.onboard.runtime `session` protocol (OnnxLegPolicy accepts any object with these two)
    def get_inputs(self):
        return self.sess.get_inputs()


def _cudart():
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.8 layout
    except ImportError:
        try:
            from cuda import cudart  # cuda-python 12.x (Jetson AI Lab jp6/cu126 has 12.6.2)
        except ImportError as e:
            raise BackendUnavailable("cuda-python not installed (pip install cuda-python==12.6.*)") from e
    return cudart


def _ck(res):
    err, *rest = res
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


class TrtRunner:
    """Static-shape engine, one input, one output, pinned host buffers, one stream, synchronous per call.
    Written against the TensorRT 10 tensor-name API (set_tensor_address / execute_async_v3). NOT run by us: no TensorRT
    on the workstation; check_parity.py is how it is validated on the Jetson."""

    def __init__(self, engine_path, name="trt"):
        trt = _tensorrt()
        require_trt_version(trt.__version__)
        cudart = _cudart()
        self._cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise BackendUnavailable(f"could not deserialize {engine_path} (built with another TensorRT or device?)")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 1 or len(outs) != 1:
            raise BackendUnavailable(f"expected 1 input / 1 output, got {ins} / {outs}")
        self.input_name, self.output_name = ins[0], outs[0]
        self.input_shape = list(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = list(self.engine.get_tensor_shape(self.output_name))
        if any(d < 0 for d in self.input_shape + self.output_shape):
            raise BackendUnavailable("dynamic shapes are not supported here; export with a static batch of 1")
        for n in (self.input_name, self.output_name):
            if self.engine.get_tensor_dtype(n) != trt.DataType.FLOAT:
                raise BackendUnavailable(f"tensor {n} is not float32 at the engine boundary")
        self.stream = _ck(cudart.cudaStreamCreate())
        self._buf = {}
        for n, shape in ((self.input_name, self.input_shape), (self.output_name, self.output_shape)):
            nbytes = int(np.prod(shape)) * 4
            dev = _ck(cudart.cudaMalloc(nbytes))
            host_ptr = _ck(cudart.cudaMallocHost(nbytes))
            host = np.ctypeslib.as_array((ctypes.c_float * int(np.prod(shape))).from_address(int(host_ptr))).reshape(shape)
            self._buf[n] = (dev, host_ptr, host, nbytes)
            self.ctx.set_tensor_address(n, int(dev))
        self.name = name
        self.info = {"backend": name, "tensorrt": trt.__version__, "engine": str(engine_path)}

    def run(self, x: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        d_in, _, h_in, nb_in = self._buf[self.input_name]
        d_out, _, h_out, nb_out = self._buf[self.output_name]
        h_in[...] = x
        _ck(cudart.cudaMemcpyAsync(d_in, h_in.ctypes.data, nb_in, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        _ck(cudart.cudaMemcpyAsync(h_out.ctypes.data, d_out, nb_out, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                   self.stream))
        _ck(cudart.cudaStreamSynchronize(self.stream))
        return h_out.copy()

    def close(self):
        for dev, host_ptr, _, _ in self._buf.values():
            self._cudart.cudaFree(dev)
            self._cudart.cudaFreeHost(host_ptr)
        self._buf = {}



def make_runner(name: str, onnx_path, engine: str | None = None, threads: int = 1, fp16: bool = False,
                cache_dir: str | None = None):
    if name == "ort-cpu":
        return OrtRunner(onnx_path, ["CPUExecutionProvider"], threads, name=name)
    if name == "ort-cuda":
        return OrtRunner(onnx_path, ["CUDAExecutionProvider", "CPUExecutionProvider"], threads, name=name)
    if name == "ort-trt":
        trt = _tensorrt()  # the EP links libnvinfer; the python package tells us which version is on the path
        require_trt_version(trt.__version__)
        cache = Path(cache_dir or Path.home() / ".cache" / "humanoid_arc" / "ort_trt")
        cache.mkdir(parents=True, exist_ok=True)
        opts = {"trt_fp16_enable": bool(fp16), "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
                "trt_timing_cache_enable": True, "trt_timing_cache_path": str(cache)}
        r = OrtRunner(onnx_path, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"], threads,
                      provider_options=[opts, {}, {}], name=name)
        r.info["fp16"] = bool(fp16)
        return r
    if name == "trt":
        if not engine:
            raise BackendUnavailable("trt backend needs --engine")
        meta = Path(str(engine) + ".json")
        r = TrtRunner(engine, name=name)
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("onnx_sha256") and m["onnx_sha256"] != sha256(onnx_path):
                raise BackendUnavailable(f"{engine} was built from another ONNX (sha256 in {meta.name} differs)")
            r.info["precision"] = m.get("precision")
        return r
    raise ValueError(f"unknown backend {name!r}")
