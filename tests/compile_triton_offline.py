"""Compile fused kernel variants for SM90 without launching kernels.

Run on Linux with Triton 3.1.0. This verifies code generation, not numerical
correctness or H100 performance. Torch is stubbed only for module import.
"""
import importlib.machinery
import os
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
    _gemv_kernel, _gemv_silu_kernel, _qkv_post_kernel,
    _qkv_post_prefill_kernel, _reduce_add_rms_kernel, _silu_mul_kernel,
    _splitk_reduce_kernel,
)


def compile_variant(name, fn, types_by_name, constants_by_name, warps, stages=1):
    signature = {i: types_by_name[arg]
                 for i, arg in enumerate(fn.arg_names)
                 if arg not in constants_by_name}
    constants = {i: constants_by_name[arg]
                 for i, arg in enumerate(fn.arg_names)
                 if arg in constants_by_name}
    kernel = triton.compile(ASTSource(fn, signature, constants),
                            target=GPUTarget("cuda", 90, 32),
                            options={"num_warps": warps, "num_stages": stages})
    assert kernel.asm["ptx"]
    print(f"{name}: SM90 PTX OK")


bf = {name: "*bf16" for name in
      ("x_ptr", "d_ptr", "w_ptr", "h_ptr", "hn_ptr", "y_ptr",
       "gu_ptr", "out_ptr", "qkv_ptr", "qn_ptr", "kn_ptr",
       "cos_ptr", "sin_ptr", "q_ptr", "kc_ptr", "vc_ptr",
       "k_ptr", "v_ptr")}
ints = {name: "i32" for name in
        ("n", "inter", "n_cols", "M", "N", "K", "kps", "stride_om",
         "S", "cap")}
floats = {"eps": "fp32", "sm_scale": "fp32"}
common = bf | ints | floats | {"pos_ptr": "*i64", "ws_ptr": "*fp32"}

for has_add in (False, True):
    compile_variant(f"RMS add={has_add}", _add_rms_kernel, common,
                    {"HAS_ADD": has_add, "BLOCK": 4096}, 8)
compile_variant("SiLU multiply", _silu_mul_kernel, common, {"BLOCK": 1024}, 4)

for last in (False, True):
    compile_variant(f"prefill last={last}", _qkv_post_prefill_kernel, common,
                    {"NH": 32, "NKV": 8, "HD": 128,
                     "LAST_QUERY_ONLY": last}, 1)
compile_variant("decode QKV", _qkv_post_kernel, common,
                {"NH": 32, "NKV": 8, "HD": 128, "DECODE": True,
                 "POS_STRIDE": 0, "PDL": False, "TRIG": 0}, 1)

for w in (1, 7):
    compile_variant(f"attention direct W={w}", _attn_split_kernel, common,
                    {"NSPLIT": 1, "G": 4, "W": w,
                     "GP": max(16, triton.next_power_of_2(w * 4)),
                     "HD": 128, "BLOCK_N": 64, "NKV": 8,
                     "POS_STRIDE": 0 if w == 1 else 1,
                     "PDL": False, "TRIG": 0}, 4, 2)
    compile_variant(f"attention split W={w}", _attn_split_kernel, common,
                    {"NSPLIT": 2, "G": 4, "W": w,
                     "GP": max(16, triton.next_power_of_2(w * 4)),
                     "HD": 128, "BLOCK_N": 64, "NKV": 8,
                     "POS_STRIDE": 0 if w == 1 else 1,
                     "PDL": False, "TRIG": 0}, 4, 2)
    compile_variant(f"attention combine W={w}", _attn_combine_kernel, common,
                    {"NSPLIT": 2, "SP": 2, "HD": 128, "NKV": 8,
                     "G": 4, "W": w, "PDL": False, "TRIG": 0}, 1)

for final, bn, bk, warps, stages in (
    (True, 64, 128, 4, 5), (False, 64, 256, 4, 4),
    (False, 32, 128, 4, 4),
):
    compile_variant(f"GEMV final={final} BN={bn} BK={bk}", _gemv_kernel,
                    common | {"out_ptr": "*bf16" if final else "*fp32"},
                    {"BM": 16, "BN": bn, "BK": bk, "FINAL": final,
                     "CM": "", "EV": "", "PDL": False, "PF": 4,
                     "TRIG": 0, "EVENK": False}, warps, stages)

compile_variant("GEMV SiLU", _gemv_silu_kernel, common,
                {"BM": 16, "BN": 64, "BK": 128, "CM": "", "EV": "",
                 "PDL": False, "PF": 4, "TRIG": 0, "EVENK": False}, 4, 4)
for sk in (2, 4):
    compile_variant(f"split-K reduction {sk}", _splitk_reduce_kernel, common,
                    {"SK": sk, "BLOCK": 1024}, 4)
    compile_variant(f"reduce/add/RMS {sk}", _reduce_add_rms_kernel, common,
                    {"SK": sk, "BLOCK": 4096, "PDL": False, "TRIG": 0},
                    8 if sk == 4 else 4)
