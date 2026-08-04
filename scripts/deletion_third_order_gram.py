"""
deletion_third_order_gram.py

coskewness の Frobenius ノルム差 ||T^A - T^B||_F^2 を、d^3 のテンソルを
一切構成せず、n x n の Gram 行列だけで計算する。

恒等式(rank-1外積の内積 = ベクトル内積の冪):
  <T^A, T^B> = (1/(n_f*n_r)) * sum_{i,m} (A_i . B_m)^3
             = (1/(n_f*n_r)) * sum( (A @ B.T)**3 )

これにより計算量は d^3 から O(n_f * n_r * d)(Gram行列を作る行列積)に
落ちる。近似ではなく厳密に同じ量(スライス版のフルテンソルでの
Frobeniusノルム差二乗)。

数値検証: python deletion_third_order_gram.py で、小規模データにおいて
素朴なeinsumテンソル構成との一致を確認する。
"""
import time

import numpy as np
import torch


def compute_covariance(X):
    Xc = X - X.mean(dim=0)
    return Xc.t() @ Xc / X.shape[0]


def coskewness_sqfrob_gram(A, B, B_gram_cubed_sum=None):
    """
    ||T^A - T^B||_F^2 を Gram trick で計算(テンソル構成なし)。

    A: (n_f, d) fake data (中心化はここで行う)
    B: (n_r, d) real data (中心化はここで行う)
    B_gram_cubed_sum: sum((B_c @ B_c.T)**3) / n_r**2 を事前計算して渡すと
                       real側(定数)の再計算を省略できる。

    戻り値: スカラー ||T^A - T^B||_F^2 (二乗Frobeniusノルム)
    """
    Ac = A - A.mean(dim=0)
    n_f = Ac.shape[0]

    G_AA = Ac @ Ac.t()                          # (n_f, n_f)
    term1 = (G_AA ** 3).sum() / (n_f ** 2)

    Bc = B - B.mean(dim=0)
    n_r = Bc.shape[0]
    G_AB = Ac @ Bc.t()                          # (n_f, n_r)
    term2 = (G_AB ** 3).sum() / (n_f * n_r)

    if B_gram_cubed_sum is None:
        G_BB = Bc @ Bc.t()
        B_gram_cubed_sum = (G_BB ** 3).sum() / (n_r ** 2)

    return term1 - 2.0 * term2 + B_gram_cubed_sum


# ----------------------------------------------------------------------
# 検証: 素朴なeinsumテンソル構成との数値一致
# ----------------------------------------------------------------------
def _naive_sqfrob(A, B):
    """素朴にテンソルを構成してFrobeniusノルム差の二乗を計算(検証専用、小規模のみ)"""
    Ac = A - A.mean(dim=0)
    Bc = B - B.mean(dim=0)
    n_f, n_r = Ac.shape[0], Bc.shape[0]
    TA = torch.einsum('ij,ik,il->jkl', Ac, Ac, Ac) / n_f
    TB = torch.einsum('mj,mk,ml->jkl', Bc, Bc, Bc) / n_r
    return ((TA - TB) ** 2).sum()


def _verify(device="cpu"):
    torch.manual_seed(0)
    for d, n_f, n_r in [(30, 40, 50), (60, 80, 100), (100, 60, 120)]:
        A = torch.rand(n_f, d, device=device, dtype=torch.float64)
        B = torch.rand(n_r, d, device=device, dtype=torch.float64)

        naive = _naive_sqfrob(A, B).item()
        gram = coskewness_sqfrob_gram(A, B).item()
        rel_err = abs(naive - gram) / max(abs(naive), 1e-12)
        print(f"d={d:4d} n_f={n_f:4d} n_r={n_r:4d}  naive={naive:.6f}  "
              f"gram={gram:.6f}  rel_err={rel_err:.3e}")
        assert rel_err < 1e-8, "Gram trick が素朴計算と一致しません"
    print("\nOK: Gram trick は素朴なテンソル構成と数値一致しました。")


def _bench(device):
    """CIFAR-10スケール(N=5000, D=3072)での1ステップ実測(forward+backward)"""
    torch.manual_seed(0)
    N, D = 5000, 3072
    real = torch.rand(N, D, device=device)
    fake = torch.rand(N, D, device=device, requires_grad=True)

    # real側(定数)の事前計算
    with torch.no_grad():
        Bc = real - real.mean(dim=0)
        G_BB = Bc @ Bc.t()
        B_gram_cubed_sum = (G_BB ** 3).sum() / (N ** 2)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_bench = 10
    for _ in range(n_bench):
        if fake.grad is not None:
            fake.grad = None
        loss = coskewness_sqfrob_gram(fake, real, B_gram_cubed_sum=B_gram_cubed_sum)
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_bench
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"\n[bench] N={N} D={D} (coskew_batch=N, no approximation, Gram trick)")
    print(f"  time/step = {dt:.4f}s   peak_mem = {peak:.2f}GB")
    print(f"  x1500 steps -> {dt*1500/60:.2f} min/class")
    print(f"  vs current slice-loop implementation (~66.7s/step): "
          f"{66.7/dt:.0f}x speedup" if dt > 0 else "")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={dev}")
    _verify(dev)
    if dev == "cuda":
        _bench(dev)


# ----------------------------------------------------------------------
# third_order_sample (Gram trick版, 近似なし・高速)
# ----------------------------------------------------------------------
def third_order_sample_gram(
    x_class,
    n_fake_samples,
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    coskew_weight=1.0,      # coskew_loss の相対重み(比較実験用に明示化)
    device="cpu",
    verbose=False,
    log_every=100,
    final_eval=True,
    desc=None,
):
    """
    Gram trick でcoskewnessを厳密計算する版。近似(ミニバッチ)は不要。
    means_loss / cov_loss / bounds_loss は既存実装と同じ定義。
    coskew_loss は sqrt(||T^A-T^B||_F^2) を D^1.5 で正規化(means/covの
    正規化流儀に倣う: cov_lossがD^2で割るのと対称に、coskewはD^3個の
    要素を持つのでD^1.5で割ってスケールを揃える)。

    coskew_weight で全体に対するcoskew項の強さを明示的に調整できる。
    """
    torch.manual_seed(rng_seed)
    label = f"[{desc}] " if desc else ""

    real = torch.as_tensor(x_class, dtype=torch.float32, device=device)
    N_real, D = real.shape

    real_mean = real.mean(dim=0)
    real_cov = compute_covariance(real)

    # real側coskewness(定数)は一度だけ計算
    with torch.no_grad():
        Bc = real - real_mean
        G_BB = Bc @ Bc.t()
        B_gram_cubed_sum = (G_BB ** 3).sum() / (N_real ** 2)

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

    t0 = time.time()
    for i in range(n_steps):
        optimizer.zero_grad()

        fake_mean = fake.mean(dim=0)
        fake_cov = compute_covariance(fake)
        means_loss = (fake_mean - real_mean).norm() / real_mean.numel()
        cov_loss = (fake_cov - real_cov).norm() / (D * D)
        bounds_loss = (torch.relu(fake - 1) + torch.relu(-fake)).mean()

        coskew_sq = coskewness_sqfrob_gram(fake, real, B_gram_cubed_sum=B_gram_cubed_sum)
        coskew_loss = torch.sqrt(coskew_sq + 1e-24) / (D ** 1.5)

        loss = means_loss + cov_loss + coskew_weight * coskew_loss + bounds_loss
        loss.backward()
        optimizer.step()

        if verbose and (i % log_every == 0 or i == n_steps - 1):
            elapsed = time.time() - t0
            print(f"{label}step {i:5d}/{n_steps} means={means_loss.item():.8f} "
                  f"cov={cov_loss.item():.8f} coskew={coskew_loss.item():.8f} "
                  f"(weighted={coskew_weight*coskew_loss.item():.8f}) "
                  f"bounds={bounds_loss.item():.8f} elapsed={elapsed:.1f}s", flush=True)

    result = fake.detach()

    if final_eval:
        with torch.no_grad():
            fm = result.mean(dim=0)
            fcov = compute_covariance(result)
            ml = float((fm - real_mean).norm() / real_mean.numel())
            cl = float((fcov - real_cov).norm() / (D * D))
            cs = coskewness_sqfrob_gram(result, real, B_gram_cubed_sum=B_gram_cubed_sum)
            cskl_raw = float(torch.sqrt(cs + 1e-24))          # 正規化前(絶対量)
            cskl_norm = cskl_raw / (D ** 1.5)                  # means/covと同スケール
            bl = float((torch.relu(result - 1) + torch.relu(-result)).mean())
            print(f"{label}[final] coskew_weight={coskew_weight}  "
                  f"means={ml:.8f} cov={cl:.8f} "
                  f"coskew_norm={cskl_norm:.8f} coskew_raw={cskl_raw:.4f} "
                  f"bounds={bl:.8f} range=[{result.min():.4f},{result.max():.4f}]",
                  flush=True)

    return result.cpu().numpy().astype(np.float32)