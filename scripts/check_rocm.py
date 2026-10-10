"""Confirm PyTorch sees the GPU (ROCm on AMD, CUDA on NVIDIA) and measure sim throughput.

    python scripts/check_rocm.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print(f"torch {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    print(f"HIP (ROCm): {hip or '-'}   CUDA: {torch.version.cuda or '-'}")
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch. On AMD, install the ROCm build of PyTorch "
                 "(https://pytorch.org/get-started/locally/, choose ROCm) and check `rocm-smi`.")
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    print(f"device: {p.name}  memory: {p.total_memory / 2**30:.0f} GB  backend: {'ROCm' if hip else 'CUDA'}")

    a = torch.randn(8192, 8192, device=dev)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()
    print(f"fp32 matmul: {10 * 2 * 8192**3 / (time.time() - t) / 1e12:.1f} TFLOPS")

    from redcell.env import DefenseEnv, MapTensors
    from redcell.terrain import load_maps
    maps = MapTensors(load_maps(), dev)
    for B in (1024, 4096, 16384):
        env = DefenseEnv(maps, B, dev)
        env.reset()
        acts = torch.randint(9, (B, 6), device=dev)
        for _ in range(3):
            env.step(acts); env.observe()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(30):
            env.step(acts); env.observe()
        torch.cuda.synchronize()
        print(f"sim batch {B:6d}: {B * 30 / (time.time() - t):,.0f} battle-steps/s")
    print("OK")


if __name__ == "__main__":
    main()
