"""Run fused pointwise kernels on a Linux CUDA GPU without PyTorch.

Requires Triton 3.1 and the NVIDIA driver. This is a small hardware smoke test;
the full engine correctness suite still requires PyTorch and model weights.
"""
import ctypes as c
import importlib.machinery
import math
import os
import struct
import sys
import types

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

torch_stub = types.ModuleType("torch")
os.environ["ENGINE_PDL"] = "0"
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", None)
sys.modules["torch"] = torch_stub
sys.path.insert(0, "engine")
from fused import (  # noqa: E402
    _add_rms_kernel, _attn_combine_kernel, _attn_split_kernel,
    _gemv_kernel, _gemv_silu_kernel, _qkv_post_prefill_kernel, _silu_mul_kernel,
    _reduce_add_rms_kernel, _splitk_reduce_kernel,
)


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

# The final prefill layer computes only each sequence's last query, while all
# prompt keys and values must still reach the cache. Check that exact mapping.
B, S, CAP, NH, NKV, HD = 2, 3, 8, 4, 2, 8
TOTAL = (NH + 2 * NKV) * HD
source = ASTSource(
    _qkv_post_prefill_kernel,
    {**{i: "*bf16" for i in range(8)},
     8: "i32", 9: "i32", 10: "fp32"},
    {11: NH, 12: NKV, 13: HD, 14: True},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 1, "num_stages": 1})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
host_qkv = (c.c_uint16 * (B * S * TOTAL))(
    *(bf16(((i % 31) - 15) / 16) for i in range(B * S * TOTAL)))
host_qn = (c.c_uint16 * HD)(*(bf16(1.0) for _ in range(HD)))
host_kn = (c.c_uint16 * HD)(*(bf16(1.0) for _ in range(HD)))
# Identity, quarter turn, and a mixed rotation check position indexing,
# rotate-half signs, and the separate BF16 multiply/add rounding points.
rot_cos = [1.0, 0.0, 0.75]
rot_sin = [0.0, 1.0, 0.25]
host_cos = (c.c_uint16 * (S * HD))(
    *(bf16(rot_cos[s]) for s in range(S) for _ in range(HD)))
host_sin = (c.c_uint16 * (S * HD))(
    *(bf16(rot_sin[s]) for s in range(S) for _ in range(HD)))
host_q = (c.c_uint16 * (B * NH * HD))()
host_k = (c.c_uint16 * (B * NKV * CAP * HD))()
host_v = (c.c_uint16 * (B * NKV * CAP * HD))()
hosts = [host_qkv, host_qn, host_kn, host_cos, host_sin,
         host_q, host_k, host_v]
buffers = [c.c_uint64() for _ in hosts]
for ptr, buf in zip(buffers, hosts):
    call("cuMemAlloc_v2", c.byref(ptr), c.sizeof(buf))
    call("cuMemcpyHtoD_v2", ptr, buf, c.sizeof(buf))
arg_s, arg_cap, arg_eps = c.c_int(S), c.c_int(CAP), c.c_float(EPS)
args = (c.c_void_p * 11)(*(c.addressof(p) for p in buffers),
                         c.addressof(arg_s),
                         c.addressof(arg_cap), c.addressof(arg_eps))
call("cuLaunchKernel", function, B * S, NKV, 1, 32, 1, 1, 0, None, args, None)
call("cuCtxSynchronize")
for ptr, buf in zip(buffers[5:], hosts[5:]):
    call("cuMemcpyDtoH_v2", buf, ptr, c.sizeof(buf))


def expected_head(t, h):
    off = t * TOTAL + h * HD
    values = [from_bf16(host_qkv[off + d]) for d in range(HD)]
    var = sum(v * v for v in values) / HD
    normed = [from_bf16(bf16(v / math.sqrt(var + EPS))) for v in values]
    position = t % S
    if position == 1:
        return [-v for v in normed[HD // 2:]] + normed[:HD // 2]
    if position == 2:
        rotated = []
        for d in range(HD):
            partner = normed[d + HD // 2] if d < HD // 2 else normed[d - HD // 2]
            signed = -partner if d < HD // 2 else partner
            a = from_bf16(bf16(normed[d] * rot_cos[position]))
            b = from_bf16(bf16(signed * rot_sin[position]))
            rotated.append(from_bf16(bf16(a + b)))
        return rotated
    return normed


for b in range(B):
    for h in range(NH):
        expected = expected_head(b * S + S - 1, h)
        for d in range(HD):
            actual = from_bf16(host_q[(b * NH + h) * HD + d])
            assert abs(actual - expected[d]) < 0.03, ("Q", b, h, d, actual, expected[d])
    for h in range(NKV):
        for s in range(S):
            t = b * S + s
            expected = expected_head(t, NH + h)
            for d in range(HD):
                dst = ((b * NKV + h) * CAP + s) * HD + d
                actual = from_bf16(host_k[dst])
                assert abs(actual - expected[d]) < 0.03, ("K", b, h, s, d, actual, expected[d])
                src = t * TOTAL + (NH + NKV + h) * HD + d
                assert host_v[dst] == host_qkv[src], ("V", b, h, s, d)
        for s in range(S, CAP):
            for d in range(HD):
                dst = ((b * NKV + h) * CAP + s) * HD + d
                assert host_k[dst] == 0 and host_v[dst] == 0, ("cache tail", b, h, s, d)
print(f"Last-query QKV prefill kernel: GPU correctness OK on SM{sm}")
for ptr in buffers:
    call("cuMemFree_v2", ptr)
call("cuModuleUnload", module)

# At NSPLIT=1 the attention kernel writes directly into token-major output.
# Zero queries/keys give uniform weights; token 0 sees V[0], token 1 their mean.
B, NKV, G, HD, CAP, W = 2, 2, 4, 32, 32, 2
source = ASTSource(
    _attn_split_kernel,
    {0: "*bf16", 1: "*bf16", 2: "*bf16", 3: "*i64",
     4: "*bf16", 5: "*bf16", 6: "i32", 7: "fp32"},
    {8: 1, 9: G, 10: W, 11: 16, 12: HD, 13: 64,
     14: NKV, 15: 0, 16: False, 17: 0},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 4, "num_stages": 2})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
host_aq = (c.c_uint16 * (B * NKV * W * G * HD))()
host_ak = (c.c_uint16 * (B * NKV * CAP * HD))()
host_av = (c.c_uint16 * (B * NKV * CAP * HD))()
for b in range(B):
    for h in range(NKV):
        base = (b + 1) * (h + 1) * 0.25
        for d in range(HD):
            host_av[((b * NKV + h) * CAP) * HD + d] = bf16(base)
            host_av[((b * NKV + h) * CAP + 1) * HD + d] = bf16(3 * base)
host_ap = (c.c_int64 * 1)(0)
host_ao = (c.c_uint16 * (B * W * NKV * G * HD))()
attn_hosts = [host_aq, host_ak, host_av, host_ap, host_ao]
attn_buffers = [c.c_uint64() for _ in attn_hosts]
for ptr, buf in zip(attn_buffers, attn_hosts):
    call("cuMemAlloc_v2", c.byref(ptr), c.sizeof(buf))
    call("cuMemcpyHtoD_v2", ptr, buf, c.sizeof(buf))
arg_cap, arg_scale = c.c_int(CAP), c.c_float(HD ** -0.5)
args = (c.c_void_p * 8)(
    c.addressof(attn_buffers[0]), c.addressof(attn_buffers[1]),
    c.addressof(attn_buffers[2]), c.addressof(attn_buffers[3]),
    c.addressof(attn_buffers[4]), c.addressof(attn_buffers[4]),
    c.addressof(arg_cap), c.addressof(arg_scale),
)
call("cuLaunchKernel", function, B * NKV, 1, 1, 128, 1, 1, kernel.metadata.shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_ao, attn_buffers[4], c.sizeof(host_ao))
for b in range(B):
    for h in range(NKV):
        base = (b + 1) * (h + 1) * 0.25
        for s in range(W):
            expected = bf16(base if s == 0 else 2 * base)
            for g in range(G):
                for d in range(HD):
                    idx = ((b * W + s) * NKV * G + h * G + g) * HD + d
                    assert host_ao[idx] == expected, (b, s, h, g, d)
print(f"Single-split direct attention: GPU correctness OK on SM{sm}")
call("cuModuleUnload", module)

# Two splits exercise the log-sum-exp combine. Token 0 masks every key in the
# second split; token 1 uses both splits and must average their different V's.
source = ASTSource(
    _attn_split_kernel,
    {0: "*bf16", 1: "*bf16", 2: "*bf16", 3: "*i64",
     4: "*fp32", 5: "*bf16", 6: "i32", 7: "fp32"},
    {8: 2, 9: G, 10: W, 11: 16, 12: HD, 13: 64,
     14: NKV, 15: 0, 16: False, 17: 0},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 4, "num_stages": 2})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
gpu_ws = c.c_uint64()
ws_bytes = B * NKV * W * G * 2 * (HD + 2) * c.sizeof(c.c_float)
call("cuMemAlloc_v2", c.byref(gpu_ws), ws_bytes)
args = (c.c_void_p * 8)(
    c.addressof(attn_buffers[0]), c.addressof(attn_buffers[1]),
    c.addressof(attn_buffers[2]), c.addressof(attn_buffers[3]),
    c.addressof(gpu_ws), c.addressof(attn_buffers[4]),
    c.addressof(arg_cap), c.addressof(arg_scale),
)
call("cuLaunchKernel", function, B * NKV, 2, 1, 128, 1, 1, kernel.metadata.shared, None, args, None)
call("cuCtxSynchronize")
call("cuModuleUnload", module)
empty_output = (c.c_uint16 * len(host_ao))()
call("cuMemcpyHtoD_v2", attn_buffers[4], empty_output, c.sizeof(empty_output))

source = ASTSource(
    _attn_combine_kernel, {0: "*fp32", 1: "*bf16"},
    {2: 2, 3: 2, 4: HD, 5: NKV, 6: G, 7: W, 8: False, 9: 0},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 1, "num_stages": 1})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
args = (c.c_void_p * 2)(c.addressof(gpu_ws), c.addressof(attn_buffers[4]))
call("cuLaunchKernel", function, B * NKV * W * G, 1, 1, 32, 1, 1,
     kernel.metadata.shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_ao, attn_buffers[4], c.sizeof(host_ao))
for b in range(B):
    for h in range(NKV):
        base = (b + 1) * (h + 1) * 0.25
        for s in range(W):
            expected = bf16(base if s == 0 else 2 * base)
            for g in range(G):
                for d in range(HD):
                    idx = ((b * W + s) * NKV * G + h * G + g) * HD + d
                    assert host_ao[idx] == expected, ("combine", b, s, h, g, d)
print(f"Split-and-combine attention: GPU correctness OK on SM{sm}")
call("cuMemFree_v2", gpu_ws)
for ptr in attn_buffers:
    call("cuMemFree_v2", ptr)
call("cuModuleUnload", module)

# A two-row skinny GEMM checks tensor-core row/column mapping and even-K
# loads. The split-K variant must reproduce the direct accumulator result.
M, N, K, BM, BN, BK = 2, 32, 64, 16, 32, 32
host_gx = (c.c_uint16 * (M * K))()
for row in range(M):
    host_gx[row * K + row] = bf16(1.0)
    host_gx[row * K + 32 + row] = bf16(1.0)
host_gw = (c.c_uint16 * (N * K))()
for n in range(N):
    host_gw[n * K] = bf16((n % 4 + 1) * 0.125)
    host_gw[n * K + 32] = bf16((n % 3 + 1) * 0.0625)
    host_gw[n * K + 1] = bf16(-(n % 5 + 1) * 0.125)
    host_gw[n * K + 33] = bf16((n % 2 + 1) * 0.25)
host_go = (c.c_uint16 * (M * N))()
gemv_hosts = [host_gx, host_gw, host_go]
gemv_buffers = [c.c_uint64() for _ in gemv_hosts]
for ptr, buf in zip(gemv_buffers, gemv_hosts):
    call("cuMemAlloc_v2", c.byref(ptr), c.sizeof(buf))
    call("cuMemcpyHtoD_v2", ptr, buf, c.sizeof(buf))
arg_m, arg_n, arg_k = c.c_int(M), c.c_int(N), c.c_int(K)
arg_kps, arg_stride = c.c_int(K), c.c_int(N)


def gemv_function(final, evenk, bn=BN, bk=BK):
    source = ASTSource(
        _gemv_kernel,
        {0: "*bf16", 1: "*bf16", 2: "*bf16" if final else "*fp32",
         3: "i32", 4: "i32", 5: "i32", 6: "i32", 7: "i32"},
        {8: BM, 9: bn, 10: bk, 11: final, 12: "", 13: "",
         14: False, 15: 4, 16: 0, 17: evenk},
    )
    compiled = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                              options={"num_warps": 4, "num_stages": 2})
    binary = c.create_string_buffer(compiled.asm["cubin"])
    loaded = c.c_void_p()
    call("cuModuleLoadData", c.byref(loaded), binary)
    func = c.c_void_p()
    call("cuModuleGetFunction", c.byref(func), loaded, compiled.metadata.name.encode())
    return loaded, func, compiled.metadata.shared


module, function, shared = gemv_function(True, True)
args = (c.c_void_p * 8)(
    c.addressof(gemv_buffers[0]), c.addressof(gemv_buffers[1]),
    c.addressof(gemv_buffers[2]), c.addressof(arg_m),
    c.addressof(arg_n), c.addressof(arg_k),
    c.addressof(arg_kps), c.addressof(arg_stride),
)
call("cuLaunchKernel", function, 1, 1, 1, 128, 1, 1, shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_go, gemv_buffers[2], c.sizeof(host_go))


def check_gemv(label):
    for row in range(M):
        for n in range(N):
            expected = bf16(from_bf16(host_gw[n * K + row]) +
                            from_bf16(host_gw[n * K + 32 + row]))
            assert host_go[row * N + n] == expected, (label, row, n)


check_gemv("direct")
print(f"Direct even-K GEMV: GPU correctness OK on SM{sm}")
call("cuModuleUnload", module)

module, function, shared = gemv_function(True, False, bn=64, bk=128)
call("cuLaunchKernel", function, 1, 1, 1, 128, 1, 1, shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_go, gemv_buffers[2], c.sizeof(host_go))
check_gemv("masked")
print(f"Direct masked GEMV: GPU correctness OK on SM{sm}")
call("cuModuleUnload", module)

gpu_gws = c.c_uint64()
call("cuMemAlloc_v2", c.byref(gpu_gws), 2 * M * N * c.sizeof(c.c_float))
arg_kps = c.c_int(32)
module, function, shared = gemv_function(False, True)
args = (c.c_void_p * 8)(
    c.addressof(gemv_buffers[0]), c.addressof(gemv_buffers[1]),
    c.addressof(gpu_gws), c.addressof(arg_m),
    c.addressof(arg_n), c.addressof(arg_k),
    c.addressof(arg_kps), c.addressof(arg_stride),
)
call("cuLaunchKernel", function, 1, 2, 1, 128, 1, 1, shared, None, args, None)
call("cuCtxSynchronize")
call("cuModuleUnload", module)
empty_gemv_output = (c.c_uint16 * (M * N))()
call("cuMemcpyHtoD_v2", gemv_buffers[2], empty_gemv_output, c.sizeof(empty_gemv_output))
source = ASTSource(_splitk_reduce_kernel,
                   {0: "*fp32", 1: "*bf16", 2: "i32"},
                   {3: 2, 4: 128})
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 4, "num_stages": 1})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
arg_count = c.c_int(M * N)
args = (c.c_void_p * 3)(c.addressof(gpu_gws), c.addressof(gemv_buffers[2]),
                       c.addressof(arg_count))
call("cuLaunchKernel", function, 1, 1, 1, 128, 1, 1, kernel.metadata.shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_go, gemv_buffers[2], c.sizeof(host_go))
check_gemv("split-K")
print(f"Split-K GEMV and reduction: GPU correctness OK on SM{sm}")
call("cuModuleUnload", module)
call("cuMemFree_v2", gpu_gws)

# Gate/up GEMV fuses two matrix products with a BF16 SiLU epilogue. The gate
# weights above exercise positive and negative inputs; the up half uses a
# different scale per input row so both halves affect the output.
host_sw = (c.c_uint16 * (2 * N * K))()
for i in range(N * K):
    host_sw[i] = host_gw[i]
for n in range(N):
    for row in range(M):
        up_part = 1.0 if row == 0 else -0.5
        host_sw[(N + n) * K + row] = bf16(up_part)
        host_sw[(N + n) * K + 32 + row] = bf16(up_part)
gpu_sw = c.c_uint64()
call("cuMemAlloc_v2", c.byref(gpu_sw), c.sizeof(host_sw))
call("cuMemcpyHtoD_v2", gpu_sw, host_sw, c.sizeof(host_sw))
source = ASTSource(
    _gemv_silu_kernel,
    {0: "*bf16", 1: "*bf16", 2: "*bf16", 3: "i32", 4: "i32", 5: "i32"},
    {6: BM, 7: BN, 8: BK, 9: "", 10: "", 11: False,
     12: 4, 13: 0, 14: True},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": 4, "num_stages": 2})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
args = (c.c_void_p * 6)(
    c.addressof(gemv_buffers[0]), c.addressof(gpu_sw),
    c.addressof(gemv_buffers[2]), c.addressof(arg_m),
    c.addressof(arg_n), c.addressof(arg_k),
)
call("cuLaunchKernel", function, 1, 1, 1, 128, 1, 1, kernel.metadata.shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_go, gemv_buffers[2], c.sizeof(host_go))
for row in range(M):
    for n in range(N):
        gate = from_bf16(bf16(from_bf16(host_sw[n * K + row]) +
                              from_bf16(host_sw[n * K + 32 + row])))
        up = from_bf16(bf16(from_bf16(host_sw[(N + n) * K + row]) +
                            from_bf16(host_sw[(N + n) * K + 32 + row])))
        silu = from_bf16(bf16(gate / (1.0 + math.exp(-gate))))
        expected = from_bf16(bf16(silu * up))
        actual = from_bf16(host_go[row * N + n])
        assert abs(actual - expected) < 0.02, ("GEMV SiLU", row, n, actual, expected)
print(f"Fused gate/up GEMV SiLU: GPU correctness OK on SM{sm}")
call("cuModuleUnload", module)
call("cuMemFree_v2", gpu_sw)
for ptr in gemv_buffers:
    call("cuMemFree_v2", ptr)

# The fused split-K epilogue rounds the split sum to BF16 before residual
# addition, then rounds the normalized value before applying the RMS weight.
R_ROWS = 2
R_COLS = int(os.environ.get("PROBE_REDUCE_COLS", "7"))
R_SK = int(os.environ.get("PROBE_REDUCE_SPLITS", "2"))
R_SCALE = float(os.environ.get("PROBE_REDUCE_RESIDUAL_SCALE", "1"))
R_BLOCK = triton.next_power_of_2(R_COLS)
R_WARPS = 8 if R_SK >= 4 else 4
host_rws = (c.c_float * (R_SK * R_ROWS * R_COLS))(
    *((col + 1) * 0.0625 if split == 0 else
      (1 if col % 2 == 0 else -1) * (row + 1) * 0.03125
      for split in range(R_SK) for row in range(R_ROWS) for col in range(R_COLS)))
host_rh = (c.c_uint16 * (R_ROWS * R_COLS))(
    *(bf16(((col - 3) * 0.25 + row * 0.125) * R_SCALE)
      for row in range(R_ROWS) for col in range(R_COLS)))
host_rw = (c.c_uint16 * R_COLS)(*(bf16(0.5 + col * 0.125) for col in range(R_COLS)))
host_rhn = (c.c_uint16 * (R_ROWS * R_COLS))()
host_ry = (c.c_uint16 * (R_ROWS * R_COLS))()
reduce_hosts = [host_rws, host_rh, host_rw, host_rhn, host_ry]
reduce_buffers = [c.c_uint64() for _ in reduce_hosts]
for ptr, buf in zip(reduce_buffers, reduce_hosts):
    call("cuMemAlloc_v2", c.byref(ptr), c.sizeof(buf))
    call("cuMemcpyHtoD_v2", ptr, buf, c.sizeof(buf))
source = ASTSource(
    _reduce_add_rms_kernel,
    {0: "*fp32", 1: "*bf16", 2: "*bf16", 3: "*bf16", 4: "*bf16",
     5: "i32", 6: "i32", 7: "fp32"},
    {8: R_SK, 9: R_BLOCK, 10: False, 11: 0},
)
kernel = triton.compile(source, target=GPUTarget("cuda", sm, 32),
                        options={"num_warps": R_WARPS, "num_stages": 1})
binary = c.create_string_buffer(kernel.asm["cubin"])
module = c.c_void_p()
call("cuModuleLoadData", c.byref(module), binary)
function = c.c_void_p()
call("cuModuleGetFunction", c.byref(function), module, kernel.metadata.name.encode())
arg_rows, arg_cols, arg_eps = c.c_int(R_ROWS), c.c_int(R_COLS), c.c_float(EPS)
args = (c.c_void_p * 8)(*(c.addressof(ptr) for ptr in reduce_buffers),
                        c.addressof(arg_rows), c.addressof(arg_cols), c.addressof(arg_eps))
call("cuLaunchKernel", function, R_ROWS, 1, 1, R_WARPS * 32, 1, 1,
     kernel.metadata.shared, None, args, None)
call("cuCtxSynchronize")
call("cuMemcpyDtoH_v2", host_rhn, reduce_buffers[3], c.sizeof(host_rhn))
call("cuMemcpyDtoH_v2", host_ry, reduce_buffers[4], c.sizeof(host_ry))
for row in range(R_ROWS):
    residual = []
    for col in range(R_COLS):
        idx = row * R_COLS + col
        split_sum = sum(host_rws[(split * R_ROWS + row) * R_COLS + col]
                        for split in range(R_SK))
        value = from_bf16(bf16(from_bf16(host_rh[idx]) +
                               from_bf16(bf16(split_sum))))
        assert host_rhn[idx] == bf16(value), ("reduce residual", row, col)
        residual.append(value)
    var = sum(v * v for v in residual) / R_COLS
    for col, value in enumerate(residual):
        idx = row * R_COLS + col
        expected = from_bf16(bf16(from_bf16(bf16(value / math.sqrt(var + EPS))) *
                                  from_bf16(host_rw[col])))
        actual = from_bf16(host_ry[idx])
        assert abs(actual - expected) < 0.02, ("reduce RMS", row, col, actual, expected)
print(f"Fused split-K reduce/add/RMSNorm (SK={R_SK}, N={R_COLS}): GPU correctness OK on SM{sm}")
call("cuModuleUnload", module)
for ptr in reduce_buffers:
    call("cuMemFree_v2", ptr)
call("cuCtxDestroy_v2", ctx)
