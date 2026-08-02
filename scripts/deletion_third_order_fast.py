"""
deletion_third_order_fast.py  (効率版・バッチ化 / メモリ制御)

3rd order hypercube-constrained max-entropy sampling (Deletion, 3次)

coskewness スライスを Python の for ループ無しで一括計算する。
素朴 einsum 'ij,ik,il->jkl' は N×slice×D×D の巨大中間で OOM するため使わない。
ここでは a(スライス幅)をバッチ次元にした batched matmul で書き、中間を
(N, slice_size, D) に抑える。slice_size でメモリを制御できる。
数値は素朴 einsum 版と一致 (python deletion_third_order_fast.py で検証)。
"""
import numpy as np
import torch


def compute_covariance(X):
    Xc = X - X.mean(dim=0)
    return Xc.t() @ Xc / X.shape[0]


def compute_coskewness_slice(X, s, e, slice_dim=0):
    """
    coskewness の slice_dim 方向 [s:e] 部分を計算 (Python ループ無し)。

        base[a,k,l] = (1/N) Σ_n Xc[n,s+a] Xc[n,k] Xc[n,l]

    実装: W[n,a,k] = Xs[n,a] * Xc[n,k]  (中間 N×slice×D) を作り、
          base[a,k,l] = (1/N) Σ_n W[n,a,k] Xc[n,l]
                      = (1/N) bmm(W.permute(a,k,n), Xc broadcast)  相当を
          einsum 'nak,nl->akl' で N を縮約して計算。
    中間 W は N×slice×D なので slice_size でメモリ制御可能。
    戻り値は slice_dim に応じて軸を並べ替えたビュー。
    """
    Xc = X - X.mean(dim=0)
    N = X.shape[0]
    Xs = Xc[:, s:e]                                  # (N, k)
    W = Xs.unsqueeze(2) * Xc.unsqueeze(1)            # (N, k, D)  中間
    base = torch.einsum('nak,nl->akl', W, Xc) / N    # (k, D, D)  N を縮約
    if slice_dim == 0:
        return base
    elif slice_dim == 1:
        return base.permute(1, 0, 2)
    elif slice_dim == 2:
        return base.permute(1, 2, 0)
    else:
        raise ValueError("slice_dim must be 0, 1, or 2")


def third_order_sample(
    x_class,
    n_fake_samples,
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    slice_dim=0,
    slice_size=16,
    device="cpu",
    verbose=False,
    log_every=100,
):
    torch.manual_seed(rng_seed)

    real = torch.as_tensor(x_class, dtype=torch.float32, device=device)
    N_real, D = real.shape

    real_mean = real.mean(dim=0)
    real_cov = compute_covariance(real)

    init_cov = real_cov + 2e-6 * torch.eye(D, device=device)
    from torch.distributions.multivariate_normal import MultivariateNormal
    while True:
        try:
            mvn = MultivariateNormal(real_mean, init_cov)
            break
        except Exception:
            init_cov = init_cov + 2e-6 * torch.eye(D, device=device)
    fake = mvn.sample((n_fake_samples,)).clamp(0, 1)
    del init_cov

    fake.requires_grad_(True)
    optimizer = torch.optim.Adam([fake], lr=lr)

    h = D
    num_slices = (h + slice_size - 1) // slice_size

    for i in range(n_steps):
        optimizer.zero_grad()

        coskew_val = 0.0
        for j in range(0, h, slice_size):
            s, e = j, min(j + slice_size, h)
            fake_coskew = compute_coskewness_slice(fake, s, e, slice_dim)
            with torch.no_grad():
                real_coskew = compute_coskewness_slice(real, s, e, slice_dim)
            slice_loss = (fake_coskew - real_coskew).norm() / real_coskew.numel()
            slice_loss = slice_loss / num_slices
            slice_loss.backward()
            coskew_val += float(slice_loss.detach())

        fake_mean = fake.mean(dim=0)
        fake_cov = compute_covariance(fake)
        means_loss = (fake_mean - real_mean).norm() / real_mean.numel()
        cov_loss = (fake_cov - real_cov).norm() / (D * D)
        bounds_loss = (torch.relu(fake - 1) + torch.relu(-fake)).mean()
        (means_loss + cov_loss + bounds_loss).backward()

        optimizer.step()

        if verbose and (i % log_every == 0 or i == n_steps - 1):
            total = means_loss.item() + cov_loss.item() + coskew_val + bounds_loss.item()
            print(f"    step {i}: total={total:.6f} "
                  f"means={means_loss.item():.6f} cov={cov_loss.item():.6f} "
                  f"coskew={coskew_val:.6f} bounds={bounds_loss.item():.6f}",
                  flush=True)

    with torch.no_grad():
        return fake.detach().cpu().numpy().astype(np.float32)


# ---------------- 数値一致検証 & ベンチ ----------------
def _coskewness_slice_naive(X, s, e, slice_dim=0):
    Xc = X - X.mean(dim=0)
    Xs = Xc[:, s:e]
    if slice_dim == 0:
        return torch.einsum('ij,ik,il->jkl', Xs, Xc, Xc) / X.shape[0]
    elif slice_dim == 1:
        return torch.einsum('ij,ik,il->jkl', Xc, Xs, Xc) / X.shape[0]
    elif slice_dim == 2:
        return torch.einsum('ij,ik,il->jkl', Xc, Xc, Xs) / X.shape[0]
    raise ValueError


def _bench_step(device):
    import time
    torch.manual_seed(0)
    N, D = 5000, 3072
    real = torch.rand(N, D, device=device)
    fake = torch.rand(20, D, device=device, requires_grad=True)
    for ss in (8, 16, 32, 64):
        num_slices = (D + ss - 1) // ss
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for j in range(0, D, ss):
            s, e = j, min(j + ss, D)
            fc = compute_coskewness_slice(fake, s, e, 0)
            with torch.no_grad():
                rc = compute_coskewness_slice(real, s, e, 0)
            loss = (fc - rc).norm() / rc.numel() / num_slices
            loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"  slice_size={ss:3d}: 1-step {dt:.3f}s  peak_mem {peak:.2f}GB "
              f"(x1500 -> {dt*1500/60:.1f} min/class, x10 classes -> {dt*1500*10/60:.1f} min)")


def _verify_against_naive(device="cpu"):
    torch.manual_seed(0)
    N, D = 2000, 120
    X = torch.rand(N, D, device=device, dtype=torch.float64)
    max_err = 0.0
    for slice_dim in (0, 1, 2):
        eff = compute_coskewness_slice(X, 0, D, slice_dim).double()
        naive = _coskewness_slice_naive(X, 0, D, slice_dim).double()
        err = (eff - naive).abs().max().item()
        max_err = max(max_err, err)
        print(f"  slice_dim={slice_dim}: max|eff-naive| = {err:.3e}")
        if slice_dim == 0:
            acc = torch.zeros(D, D, D, device=device, dtype=torch.float64)
            for j in range(0, D, 16):
                e = min(j + 16, D)
                acc[j:e] = compute_coskewness_slice(X, j, e, 0).double()
            err2 = (acc - naive).abs().max().item()
            print(f"           sliced-accumulate vs full: max err = {err2:.3e}")
    print(f"\n  overall max error = {max_err:.3e}")
    assert max_err < 1e-8, "効率版が旧版と一致しません"
    print("  OK: 効率版は旧版(einsum)と数値一致しました。")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={dev}")
    _verify_against_naive(dev)
    if dev == "cuda":
        print("\n[bench] 1-step timing at real scale (N=5000, D=3072):")
        _bench_step(dev)