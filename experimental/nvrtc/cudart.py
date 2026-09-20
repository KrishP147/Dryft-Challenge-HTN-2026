"""Runtime-compiled CUDA C++ via NVRTC + the driver API, using only ctypes and torch's bundled libs.

No nvcc, no ninja, no compiled artifacts shipped: the kernel source is a Python string, compiled to a
cubin at load time by libnvrtc (a dependency of the torch wheel) and launched with cuLaunchKernel on
torch's current stream (so CUDA-graph capture records the launches).
"""
import ctypes
import glob
import os

import torch


def _find_lib(name_glob, subdir):
    root = os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), "nvidia", subdir, "lib")
    for p in sorted(glob.glob(os.path.join(root, name_glob))):
        return p
    return None


_nvrtc = None
_cuda = None


def _libs():
    global _nvrtc, _cuda
    if _nvrtc is None:
        path = _find_lib("libnvrtc.so*", "cuda_nvrtc") or "libnvrtc.so.12"
        _nvrtc = ctypes.CDLL(path)
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _nvrtc, _cuda


def _check_nvrtc(res, what):
    if res != 0:
        raise RuntimeError(f"nvrtc {what} failed with code {res}")


def _check_cu(res, what):
    if res != 0:
        raise RuntimeError(f"cuda driver {what} failed with code {res}")


def compile_cubin(src, arch="sm_90", extra=()):
    nvrtc, _ = _libs()
    prog = ctypes.c_void_p()
    _check_nvrtc(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"kernel.cu", 0, None, None), "create")
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17", *[o.encode() for o in extra]]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    if res != 0:
        n = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("nvrtc compile failed:\n" + log.value.decode(errors="replace"))
    size = ctypes.c_size_t()
    _check_nvrtc(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "cubin size")
    buf = ctypes.create_string_buffer(size.value)
    _check_nvrtc(nvrtc.nvrtcGetCUBIN(prog, buf), "cubin")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


class Module:
    """A loaded cubin. Keep a reference: the driver unloads the module when it is destroyed."""

    def __init__(self, cubin):
        _, cuda = _libs()
        torch.zeros(1, device="cuda")  # make sure torch's primary context is current on this thread
        self._cubin = cubin
        self.mod = ctypes.c_void_p()
        _check_cu(cuda.cuModuleLoadData(ctypes.byref(self.mod), cubin), "cuModuleLoadData")
        self._fns = {}

    def function(self, name, max_dynamic_smem=0):
        _, cuda = _libs()
        fn = ctypes.c_void_p()
        _check_cu(cuda.cuModuleGetFunction(ctypes.byref(fn), self.mod, name.encode()), f"cuModuleGetFunction({name})")
        if max_dynamic_smem:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
            _check_cu(cuda.cuFuncSetAttribute(fn, 8, max_dynamic_smem), "cuFuncSetAttribute")
        return Kernel(fn)


class Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, grid, block, args, smem=0):
        """args: torch tensors (passed as device pointers), Python ints (int32) or floats (float32)."""
        _, cuda = _libs()
        keep = []
        for a in args:
            if isinstance(a, torch.Tensor):
                keep.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, bool):
                keep.append(ctypes.c_int(int(a)))
            elif isinstance(a, int):
                keep.append(ctypes.c_int(a))
            elif isinstance(a, float):
                keep.append(ctypes.c_float(a))
            else:
                keep.append(a)
        params = (ctypes.c_void_p * len(keep))(*[ctypes.cast(ctypes.pointer(k), ctypes.c_void_p) for k in keep])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check_cu(cuda.cuLaunchKernel(self.fn, *g, *b, smem, stream, params, None), "cuLaunchKernel")


_CANARY_SRC = r"""
extern "C" __global__ void canary(float* out, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v + (float)i;
}
"""


def canary():
    """Which stages work on this machine? Returned as a per-step delay (seconds) so a public run can
    read the answer back from TPOT: 2 ms = libnvrtc loads, +4 ms = compiles to a cubin,
    +8 ms = loads + launches on torch's stream and computes the right answer. 0 = nothing works."""
    delay = 0.0
    try:
        _libs()
        delay += 0.002
        cubin = compile_cubin(_CANARY_SRC, "sm_90")
        delay += 0.004
        mod = Module(cubin)
        k = mod.function("canary")
        out = torch.zeros(1024, device="cuda")
        k((4,), (256,), [out, 1024, 3.0])
        torch.cuda.synchronize()
        ok = bool(torch.equal(out, 3.0 + torch.arange(1024, device="cuda", dtype=torch.float32)))
        if ok:
            delay += 0.008
    except Exception:
        pass
    return delay
