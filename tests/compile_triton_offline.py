"""Compile fused QKV and attention variants for SM90 without launching kernels.

Run on Linux with Triton 3.1.0. Torch is stubbed only to import the JIT definitions.
This verifies code generation, not numerical correctness or performance.
"""
import importlib.machinery
import sys
import types

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

torch_stub = types.ModuleType("torch")
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", None)
sys.modules["torch"] = torch_stub
sys.path.insert(0, "engine")
from fused import _attn_combine_kernel, _attn_split_kernel, _qkv_post_kernel  # noqa: E402

pointer_types = {name: "*bf16" for name in
                 ("qkv_ptr", "qn_ptr", "kn_ptr", "cos_ptr", "sin_ptr",
                  "q_ptr", "kc_ptr", "vc_ptr")}
pointer_types["pos_ptr"] = "*i64"
scalar_types = {"B": "i32", "S": "i32", "cap": "i32", "eps": "fp32"}

for name, decode, last_query_only in (
    ("prefill", False, False),
    ("last-query prefill", False, True),
    ("decode", True, False),
):
    constants = {"NH": 32, "NKV": 8, "HD": 128,
                 "DECODE": decode, "LAST_QUERY_ONLY": last_query_only}
    signature = {i: (pointer_types | scalar_types)[arg]
                 for i, arg in enumerate(_qkv_post_kernel.arg_names)
                 if arg not in constants}
    constants = {i: constants[arg]
                 for i, arg in enumerate(_qkv_post_kernel.arg_names)
                 if arg in constants}
    source = ASTSource(_qkv_post_kernel, signature, constants)
    kernel = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                            options={"num_warps": 1, "num_stages": 1})
    assert kernel.asm["ptx"]
    print(f"{name}: SM90 PTX OK")

for nsplit in (1, 2):
    pointer_types = {
        "q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16",
        "pos_ptr": "*i64", "ws_ptr": "*bf16" if nsplit == 1 else "*fp32",
        "out_ptr": "*bf16",
    }
    scalar_types = {"cap": "i32", "sm_scale": "fp32"}
    constants = {"NSPLIT": nsplit, "G": 4, "GP": 16, "HD": 128, "BLOCK_N": 64}
    signature = {i: (pointer_types | scalar_types)[arg]
                 for i, arg in enumerate(_attn_split_kernel.arg_names)
                 if arg not in constants}
    constants = {i: constants[arg]
                 for i, arg in enumerate(_attn_split_kernel.arg_names)
                 if arg in constants}
    source = ASTSource(_attn_split_kernel, signature, constants)
    kernel = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                            options={"num_warps": 4, "num_stages": 2})
    assert kernel.asm["ptx"]
    print(f"attention split {nsplit}: SM90 PTX OK")

signature = {0: "*fp32", 1: "*bf16"}
constants = {2: 2, 3: 2, 4: 128}
source = ASTSource(_attn_combine_kernel, signature, constants)
kernel = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                        options={"num_warps": 1, "num_stages": 1})
assert kernel.asm["ptx"]
print("attention combine: SM90 PTX OK")
