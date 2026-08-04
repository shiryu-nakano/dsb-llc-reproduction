"""
bench_slice_size.py

近似なし(coskew_batch=N=5000)のまま、slice_sizeを変えて1ステップの
実測時間とピークメモリを計測する。目的は「近似を入れずに高速化できる
余地があるか」を確認すること。

現在の実装(_coskewness_slice_cross)は192回(slice_size=16のとき)の
backward呼び出しを1ステップ内で行っており、これがボトルネックと
推測される。slice_sizeを上げるとループ回数が減る一方、中間テンソル
(N, slice_size, D)が線形に大きくなるので、GPUメモリとの兼ね合いを見る。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from deletion_third_order_minibatch import _coskewness_slice_cross


def bench(device, N=5000, D=3072, n_bench_steps=5, slice_sizes=(16, 32, 64, 128, 256, 512, 1024)):
    torch.manual_seed(0)
    real = torch.rand(N, D, device=device)
    fake = torch.rand(N, D, device=device, requires_grad=True)

    print(f"device={device}  N={N} D={D}  (coskew_batch=N, no approximation)")
    print(f"{'slice_size':>10} {'n_slices':>9} {'time/step(s)':>13} {'peak_mem(GB)':>13} {'status':>8}")

    results = []
    for ss in slice_sizes:
        num_slices = (D + ss - 1) // ss
        try:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(n_bench_steps):
                if fake.grad is not None:
                    fake.grad = None
                for j in range(0, D, ss):
                    s, e = j, min(j + ss, D)
                    fc = _coskewness_slice_cross(fake, fake, s, e)
                    with torch.no_grad():
                        rc = _coskewness_slice_cross(real, real, s, e)
                    loss = (fc - rc).norm() / rc.numel() / num_slices
                    loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / n_bench_steps
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            print(f"{ss:>10} {num_slices:>9} {dt:>13.3f} {peak:>13.2f} {'OK':>8}")
            results.append((ss, num_slices, dt, peak, "OK"))
        except torch.cuda.OutOfMemoryError:
            print(f"{ss:>10} {num_slices:>9} {'--':>13} {'--':>13} {'OOM':>8}")
            results.append((ss, num_slices, None, None, "OOM"))
            if device == "cuda":
                torch.cuda.empty_cache()
            break  # これ以上大きいslice_sizeは確実にOOMなので打ち切り

    if results:
        ok = [r for r in results if r[4] == "OK"]
        if ok:
            best = min(ok, key=lambda r: r[2])
            print(f"\nbest: slice_size={best[0]}  time/step={best[2]:.3f}s  "
                  f"(x1500 steps -> {best[2]*1500/60:.1f} min/class)")
            baseline = ok[0]
            speedup = baseline[2] / best[2]
            print(f"speedup vs slice_size={baseline[0]}: {speedup:.2f}x")
    return results


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bench(dev)