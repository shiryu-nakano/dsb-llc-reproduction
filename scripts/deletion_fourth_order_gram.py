"""
deletion_fourth_order_gram.py

4次(cokurtosis)制約 deletion データ生成、Gram trick版。

3次(coskewness)の deletion_third_order_gram.py と全く同じ構造で、
恒等式の指数を3から4に変えるだけで拡張できる:

  3次: <T3^A, T3^B> = (1/(n_f*n_r)) * sum( (A @ B.T)**3 )
  4次: <T4^A, T4^B> = (1/(n_f*n_r)) * sum( (A @ B.T)**4 )

これにより cokurtosis テンソル(d^4 要素、CIFAR-10 d=3072 なら
3072^4 ≈ 8.9*10^13 要素 = 素朴計算では絶対に構成不可能)を一切作らず、
n×n の Gram 行列だけで正確に計算できる。3次と同じく近似は一切ない。

数値検証: python deletion_fourth_order_gram.py で、小規模データにおいて
素朴なeinsumテンソル構成との一致を確認する。
"""
import time

import numpy as np
import torch


def compute_covariance(X):
    Xc = X - X.mean(dim=0)
    return Xc.t() @ Xc / X.shape[0]


def cokurtosis_sqfrob_gram(A, B, B_gram_pow_sum=None):
    """
    ||T4^A - T4^B||_F^2 を Gram trick で計算(d^4 テンソル構成なし)。

    A: (n_f, d) fake data (中心化はここで行う)
    B: (n_r, d) real data (中心化はここで行う)
    B_gram_pow_sum: sum((B_c @ B_c.T)**4) / n_r**2 を事前計算して渡すと
                    real側(定数)の再計算を省略できる。
    """
    Ac = A - A.mean(dim=0)
    n_f = Ac.shape[0]

    G_AA = Ac @ Ac.t()
    term1 = (G_AA ** 4).sum() / (n_f ** 2)

    Bc = B - B.mean(dim=0)
    n_r = Bc.shape[0]
    G_AB = Ac @ Bc.t()
    term2 = (G_AB ** 4).sum() / (n_f * n_r)

    if B_gram_pow_sum is None:
        G_BB = Bc @ Bc.t()
        B_gram_pow_sum = (G_BB ** 4).sum() / (n_r ** 2)

    return term1 - 2.0 * term2 + B_gram_pow_sum


def fourth_order_sample_gram(
    x_class,
    n_fake_samples,
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    coskew_weight=100.0,     # 3次(coskewness)項の重み。3次sweepの結論(weight=100)を踏襲
    cokurt_weight=100.0,     # 4次(cokurtosis)項の重み。要sweep検証(実務上の注意: 4次はスケールがよりシビア)
    device="cpu",
    verbose=False,
    log_every=100,
    final_eval=True,
    desc=None,
):
    """
    means/cov/coskewness/cokurtosis/bounds を同時に最適化する。
    3次(third_order_gram)の拡張版: 4次(cokurtosis)項を追加。

    数値スケールの注意(4次は3次よりシビア):
    (A・A')^4 は内積の4乗なので値のダイナミックレンジが3次よりさらに広い。
    cokurt_weight は3次のweight=100をそのまま流用しているが、これは未検証の
    初期値であり、3次と同様に sweep して確認する必要がある。
    """
    torch.manual_seed(rng_seed)
    label = f"[{desc}] " if desc else ""

    real = torch.as_tensor(x_class, dtype=torch.float32, device=device)
    N_real, D = real.shape

    real_mean = real.mean(dim=0)
    real_cov = compute_covariance(real)

    # real側(定数)の3次・4次Gram量は一度だけ計算
    with torch.no_grad():
        Bc = real - real_mean
        G_BB = Bc @ Bc.t()
        B_gram_cubed_sum = (G_BB ** 3).sum() / (N_real ** 2)
        B_gram_pow4_sum = (G_BB ** 4).sum() / (N_real ** 2)

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

        # 3次
        Ac = fake - fake_mean
        G_AA3 = Ac @ Ac.t()
        term1_3 = (G_AA3 ** 3).sum() / (n_fake_samples ** 2)
        G_AB3 = Ac @ Bc.t()
        term2_3 = (G_AB3 ** 3).sum() / (n_fake_samples * N_real)
        coskew_sq = term1_3 - 2.0 * term2_3 + B_gram_cubed_sum
        coskew_loss = torch.sqrt(coskew_sq + 1e-24) / (D ** 1.5)

        # 4次 (同じGram行列 G_AA3, G_AB3 を再利用、累乗だけ変える)
        term1_4 = (G_AA3 ** 4).sum() / (n_fake_samples ** 2)
        term2_4 = (G_AB3 ** 4).sum() / (n_fake_samples * N_real)
        cokurt_sq = term1_4 - 2.0 * term2_4 + B_gram_pow4_sum
        cokurt_loss = torch.sqrt(cokurt_sq + 1e-24) / (D ** 2.0)

        loss = (means_loss + cov_loss
                + coskew_weight * coskew_loss
                + cokurt_weight * cokurt_loss
                + bounds_loss)
        loss.backward()
        optimizer.step()

        if verbose and (i % log_every == 0 or i == n_steps - 1):
            elapsed = time.time() - t0
            print(f"{label}step {i:5d}/{n_steps} means={means_loss.item():.8f} "
                  f"cov={cov_loss.item():.8f} coskew={coskew_loss.item():.8f} "
                  f"cokurt={cokurt_loss.item():.8f} bounds={bounds_loss.item():.8f} "
                  f"elapsed={elapsed:.1f}s", flush=True)

    result = fake.detach()

    if final_eval:
        with torch.no_grad():
            fm = result.mean(dim=0)
            fcov = compute_covariance(result)
            ml = float((fm - real_mean).norm() / real_mean.numel())
            cl = float((fcov - real_cov).norm() / (D * D))

            Ac = result - fm
            G_AA3 = Ac @ Ac.t()
            G_AB3 = Ac @ Bc.t()
            cs = ((G_AA3 ** 3).sum() / (n_fake_samples ** 2)
                  - 2.0 * (G_AB3 ** 3).sum() / (n_fake_samples * N_real)
                  + B_gram_cubed_sum)
            coskew_norm = float(torch.sqrt(cs + 1e-24)) / (D ** 1.5)

            ck = ((G_AA3 ** 4).sum() / (n_fake_samples ** 2)
                  - 2.0 * (G_AB3 ** 4).sum() / (n_fake_samples * N_real)
                  + B_gram_pow4_sum)
            cokurt_norm = float(torch.sqrt(ck + 1e-24)) / (D ** 2.0)

            bl = float((torch.relu(result - 1) + torch.relu(-result)).mean())
            print(f"{label}[final] coskew_weight={coskew_weight} cokurt_weight={cokurt_weight}  "
                  f"means={ml:.8f} cov={cl:.8f} coskew_norm={coskew_norm:.8f} "
                  f"cokurt_norm={cokurt_norm:.8f} bounds={bl:.8f} "
                  f"range=[{result.min():.4f},{result.max():.4f}]", flush=True)

    return result.cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------
# 検証: 素朴なeinsumテンソル構成との数値一致(4次)
# ----------------------------------------------------------------------
def _naive_sqfrob_4th(A, B):
    Ac = A - A.mean(dim=0)
    Bc = B - B.mean(dim=0)
    n_f, n_r = Ac.shape[0], Bc.shape[0]
    TA = torch.einsum('ij,ik,il,im->jklm', Ac, Ac, Ac, Ac) / n_f
    TB = torch.einsum('nj,nk,nl,nm->jklm', Bc, Bc, Bc, Bc) / n_r
    return ((TA - TB) ** 2).sum()


def _verify(device="cpu"):
    torch.manual_seed(0)
    for d, n_f, n_r in [(20, 30, 40), (40, 60, 70)]:
        A = torch.rand(n_f, d, device=device, dtype=torch.float64)
        B = torch.rand(n_r, d, device=device, dtype=torch.float64)

        naive = _naive_sqfrob_4th(A, B).item()
        gram = cokurtosis_sqfrob_gram(A, B).item()
        rel_err = abs(naive - gram) / max(abs(naive), 1e-12)
        print(f"d={d:4d} n_f={n_f:4d} n_r={n_r:4d}  naive={naive:.6f}  "
              f"gram={gram:.6f}  rel_err={rel_err:.3e}")
        assert rel_err < 1e-8, "4次Gram trickが素朴計算と一致しません"
    print("\nOK: 4次(cokurtosis) Gram trick は素朴なテンソル構成と数値一致しました。")


def _bench(device):
    """CIFAR-10スケール(N=5000, D=3072)での1ステップ実測(3次+4次同時)"""
    torch.manual_seed(0)
    N, D = 5000, 3072
    real = torch.rand(N, D, device=device)
    fake = torch.rand(N, D, device=device, requires_grad=True)

    with torch.no_grad():
        Bc = real - real.mean(dim=0)
        G_BB = Bc @ Bc.t()
        B3 = (G_BB ** 3).sum() / (N ** 2)
        B4 = (G_BB ** 4).sum() / (N ** 2)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_bench = 10
    for _ in range(n_bench):
        if fake.grad is not None:
            fake.grad = None
        Ac = fake - fake.mean(dim=0)
        G_AA = Ac @ Ac.t()
        G_AB = Ac @ Bc.t()
        cs = (G_AA ** 3).sum() / (N ** 2) - 2 * (G_AB ** 3).sum() / (N * N) + B3
        ck = (G_AA ** 4).sum() / (N ** 2) - 2 * (G_AB ** 4).sum() / (N * N) + B4
        loss = torch.sqrt(cs + 1e-24) + torch.sqrt(ck + 1e-24)
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_bench
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"\n[bench] N={N} D={D} (3次+4次同時, no approximation, Gram trick)")
    print(f"  time/step = {dt:.4f}s   peak_mem = {peak:.2f}GB")
    print(f"  x1500 steps -> {dt*1500/60:.2f} min/class")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={dev}")
    _verify(dev)
    if dev == "cuda":
        _bench(dev)