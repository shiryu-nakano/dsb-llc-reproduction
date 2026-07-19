"""
test_bounded_shift.py

bounded_shift の最小動作確認。
クラス0 -> クラス1 のみ、少数サンプルで試す。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # cuda_test/ をパスに追加

import numpy as np
from train_and_save import load_cifar10


def bounded_shift(x, target, bounds=(0.0, 1.0), max_iter=10_000, tol=1e-4, verbose=True):
    mu = x.mean(axis=0)
    for i in range(max_iter):
        x = x - (mu - target)
        x = np.clip(x, *bounds)
        mu = x.mean(axis=0)
        if np.max(np.abs(mu - target)) < tol:
            if verbose:
                print(f"    Converged in {i} iterations, max diff = {np.max(np.abs(mu - target)):.6f}")
            break
    else:
        if verbose:
            print(f"    Did NOT converge in {max_iter} iterations, max diff = {np.max(np.abs(mu - target)):.6f}")
    return x


def main():
    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    H, W, C = x_train.shape[1:]
    D = H * W * C

    source_class = 0
    target_class = 1
    n_samples = 20

    x_source_all = x_train[y_train == source_class].reshape(-1, D)
    x_target_all = x_train[y_train == target_class].reshape(-1, D)

    x_source = x_source_all[:n_samples].copy()
    target_mu = x_target_all.mean(axis=0)

    print(f"\nSource class {source_class}, n={n_samples}")
    print(f"Source mean (before): {x_source.mean(axis=0)[:5]} ...")
    print(f"Target mean:          {target_mu[:5]} ...")

    shifted = bounded_shift(x_source, target_mu, max_iter=10_000, tol=1e-4, verbose=True)

    print(f"\nSource mean (after):  {shifted.mean(axis=0)[:5]} ...")
    print(f"Diff from target:     {np.max(np.abs(shifted.mean(axis=0) - target_mu)):.6f}")
    print(f"Value range: [{shifted.min():.3f}, {shifted.max():.3f}]")

    out_path = Path(__file__).resolve().parent.parent / "test_shifted_sample.npz"
    np.savez(
        out_path,
        original=x_source_all[:n_samples].reshape(-1, H, W, C),
        shifted=shifted.reshape(-1, H, W, C),
        source_class=source_class,
        target_class=target_class,
    )
    print(f"\nSaved sample to {out_path}")


if __name__ == "__main__":
    main()