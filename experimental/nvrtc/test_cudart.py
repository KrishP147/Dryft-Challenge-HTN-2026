"""NVRTC + driver API smoke test: compile, load, launch (also inside a CUDA graph), check results."""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
import cudart
print("canary delay (s):", cudart.canary())
src = r'''
extern "C" __global__ void axpy(const float* x, float* y, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}'''
t0 = time.time(); cubin = cudart.compile_cubin(src, "sm_90"); print(f"nvrtc compile {time.time()-t0:.2f}s, cubin {len(cubin)} B")
k = cudart.Module(cubin).function("axpy")
x = torch.randn(1 << 20, device="cuda"); y = torch.randn(1 << 20, device="cuda"); ref = 2.0 * x + y
k(((1 << 20) // 256,), (256,), [x, y, 1 << 20, 2.0]); torch.cuda.synchronize()
print("eager launch ok:", torch.allclose(y, ref))
y2 = torch.randn(1 << 20, device="cuda"); ref2 = 3 * (2.0 * x + y2) + 0  # placeholder shape check below
y3 = y2.clone(); g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): k(((1 << 20) // 256,), (256,), [x, y3, 1 << 20, 0.0])
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
with torch.cuda.graph(g): k(((1 << 20) // 256,), (256,), [x, y3, 1 << 20, 1.0])
base = y3.clone(); g.replay(); g.replay(); torch.cuda.synchronize()
print("graph replay ok:", torch.allclose(y3, base + 2 * x, atol=1e-4))
