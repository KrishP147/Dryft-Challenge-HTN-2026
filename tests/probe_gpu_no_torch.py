"""Run fused pointwise kernels on a Linux CUDA GPU without PyTorch.

Requires Triton 3.1 and the NVIDIA driver. This is a small hardware smoke test;
the full engine correctness suite still requires PyTorch and model weights.
"""
import ctypes as c
import importlib.machinery
import math
import struct
import sys
import types

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

torch_stub = types.ModuleType("torch")
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", None)
sys.modules["torch"] = torch_stub
sys.path.insert(0, "engine")
from fused import _add_rms_kernel, _silu_mul_kernel  # noqa: E402


def check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with CUDA error {code}")


cuda = c.CDLL("libcuda.so.1")


def call(name, *args):
    check(getattr(cuda, name)(*args), name)


def bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return bits >> 16


def from_bf16(bits):
    return struct.unpack("f", struct.pack("I", bits << 16))[0]


call("cuInit", 0)
device = c.c_int()
call("cuDeviceGet", c.byref(device), 0)
major, minor = c.c_int(), c.c_int()
call("cuDeviceGetAttribute", c.byref(major), 75, device)
call("cuDeviceGetAttribute", c.byref(minor), 76, device)
sm = major.value * 10 + minor.value
ctx = c.c_void_p()
call("cuCtxCreate_v2", c.byref(ctx), 0, device)

N, INTER = 13, 8
signature = {0: "*bf16", 1: "*bf16", 2: "i32", 3: "i32"}
source = ASTSource(_silu_mul_kernel, signature, {4: 1024})
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 4, "num_stages": 1})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())

gates = [[-4.0, -1.0, 0.0, 1.0, 4.0, 0.5, -0.5, 2.0],
         [2.5, -3.0, 0.25, -0.25, 1.5, 0.0, 0.0, 0.0]]
ups = [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
       [-1.0, 2.0, -3.0, 4.0, -5.0, 0.0, 0.0, 0.0]]
values = gates[0] + ups[0] + gates[1] + ups[1]
host_in = (c.c_uint16 * len(values))(*(bf16(x) for x in values))
host_out = (c.c_uint16 * N)()
gpu_in, gpu_out = c.c_uint64(), c.c_uint64()
call("cuMemAlloc_v2", c.byref(gpu_in), c.sizeof(host_in))
call("cuMemAlloc_v2", c.byref(gpu_out), c.sizeof(host_out))
call("cuMemcpyHtoD_v2", gpu_in, host_in, c.sizeof(host_in))
arg_n, arg_inter = c.c_int(N), c.c_int(INTER)
args = (c.c_void_p * 4)(
    c.addressof(gpu_in), c.addressof(gpu_out),
    c.addressof(arg_n), c.addressof(arg_inter),
)
call("cuLaunchKernel", function, 1, 1, 1, 128, 1, 1, 0, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_out, gpu_out, c.sizeof(host_out))
for i in range(N):
    row, col = divmod(i, INTER)
    gate, up = gates[row][col], ups[row][col]
    expected = from_bf16(bf16(from_bf16(bf16(gate / (1 + math.exp(-gate)))) * up))
    actual = from_bf16(host_out[i])
    assert abs(actual - expected) < 0.05, (i, actual, expected)
print(f"SiLU fused kernel: GPU correctness OK on SM{sm}")
call("cuMemFree_v2", gpu_out)
call("cuMemFree_v2", gpu_in)
call("cuModuleUnload", module)

# Add plus RMSNorm has two important bf16 round points: residual addition and
# normalized activation before weight multiplication. Exercise both on GPU.
COLS, ROWS, EPS = 7, 2, 1e-6
source = ASTSource(
    _add_rms_kernel,
    {0: "*bf16", 1: "*bf16", 2: "*bf16", 3: "*bf16", 4: "*bf16",
     5: "i32", 6: "fp32"},
    {7: True, 8: 128},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 4, "num_stages": 1})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
x = [0.25, -0.5, 1.0, -1.5, 2.0, -2.5, 0.125,
     -0.125, 0.75, -1.25, 1.75, -2.25, 2.75, -0.875]
d = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7,
     0.7, -0.6, 0.5, -0.4, 0.3, -0.2, -0.1]
w = [1.0, 0.75, 1.25, -0.5, 1.5, 0.25, -1.0]
host_x = (c.c_uint16 * len(x))(*(bf16(v) for v in x))
host_d = (c.c_uint16 * len(d))(*(bf16(v) for v in d))
host_w = (c.c_uint16 * len(w))(*(bf16(v) for v in w))
host_h = (c.c_uint16 * len(x))()
host_y = (c.c_uint16 * len(x))()
buffers = [c.c_uint64() for _ in range(5)]
for ptr, buf in zip(buffers, (host_x, host_d, host_w, host_h, host_y)):
    call("cuMemAlloc_v2", c.byref(ptr), c.sizeof(buf))
for ptr, buf in zip(buffers[:3], (host_x, host_d, host_w)):
    call("cuMemcpyHtoD_v2", ptr, buf, c.sizeof(buf))
arg_cols, arg_eps = c.c_int(COLS), c.c_float(EPS)
args = (c.c_void_p * 7)(*(c.addressof(p) for p in buffers),
                        c.addressof(arg_cols), c.addressof(arg_eps))
call("cuLaunchKernel", function, ROWS, 1, 1, 128, 1, 1, 0, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_h, buffers[3], c.sizeof(host_h))
call("cuMemcpyDtoH_v2", host_y, buffers[4], c.sizeof(host_y))
for row in range(ROWS):
    residual = [from_bf16(bf16(from_bf16(host_x[row * COLS + col]) +
                               from_bf16(host_d[row * COLS + col])))
                for col in range(COLS)]
    var = sum(v * v for v in residual) / COLS
    for col, value in enumerate(residual):
        idx = row * COLS + col
        expected = from_bf16(bf16(from_bf16(bf16(value / math.sqrt(var + EPS))) *
                                  from_bf16(host_w[col])))
        assert host_h[idx] == bf16(value), (idx, from_bf16(host_h[idx]), value)
        actual = from_bf16(host_y[idx])
        assert abs(actual - expected) < 0.02, (idx, actual, expected)
print(f"Add plus RMSNorm fused kernel: GPU correctness OK on SM{sm}")
for ptr in buffers:
    call("cuMemFree_v2", ptr)
call("cuModuleUnload", module)
call("cuCtxDestroy_v2", ctx)
