"""
deletion_order_gram.py

任意次数(1〜6次)のmoment-edited deletionデータを、Gram trickで生成する
汎用モジュール。エントロピー項(KoLeo)あり/なしの両方に対応。

- 次数kの中心モーメントテンソルのFrobeniusノルム差二乗は、d^kテンソルを
  一切構成せず n×n の Gram行列のk乗の和だけで厳密計算できる
  (rank-1外積の内積=ベクトル内積の冪、という恒等式の一般化):
    <T^A,T^B> = (1/(n_f n_r)) * sum( (A @ B.T)**k )
  この計算量はkに依存しない(Gram行列自体の計算コストが支配的)ため、
  5次・6次に拡張しても速度はほぼ変わらない。

- means(1次)・cov(2次)は直接計算(このGram式を使う必要がない、軽いので)。
  3次以上はGram trickで計算し、重み(moment_weights)を掛けて合算する。

- use_entropy=False: 従来通りReLU bounds_lossで[0,1]制約(third/fourth_order_gram方式)。
- use_entropy=True:  sigmoid変換によるbox制約 + KoLeoエントロピー項を追加
  (本家 truncated_normal.py の設計を踏襲、second_order_koleoの一般化)。
"""
import time

import numpy as np
import torch
from torch import nn


def compute_covariance(X):
    Xc = X - X.mean(dim=0)
    return Xc.t() @ Xc / X.shape[0]


def koleo(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x).kthvalue(2).values.log().mean()


def moment_k_sqfrob_gram(A, B, k, B_gram_pow_sum=None):
    """
    k次(k>=3)中心モーメントテンソルの ||T^A - T^B||_F^2 を Gram trick で計算。
    A: (n_f,d) fake, B: (n_r,d) real。中心化はここで行う。
    """
    Ac = A - A.mean(dim=0)
    n_f = Ac.shape[0]
    G_AA = Ac @ Ac.t()
    term1 = (G_AA ** k).sum() / (n_f ** 2)

    Bc = B - B.mean(dim=0)
    n_r = Bc.shape[0]
    G_AB = Ac @ Bc.t()
    term2 = (G_AB ** k).sum() / (n_f * n_r)

    if B_gram_pow_sum is None:
        G_BB = Bc @ Bc.t()
        B_gram_pow_sum = (G_BB ** k).sum() / (n_r ** 2)

    return term1 - 2.0 * term2 + B_gram_pow_sum


def moment_order_sample_gram(
    x_class,
    n_fake_samples,
    order,                    # 1〜6
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    moment_weights=None,      # {3: w3, 4: w4, ...} 省略時は3次以上すべて100.0
    use_entropy=False,
    koleo_weight=1e-3,
    device="cpu",
    verbose=False,
    log_every=300,
    final_eval=True,
    desc=None,
):
    assert 1 <= order <= 6
    if moment_weights is None:
        moment_weights = {k: 100.0 for k in range(3, order + 1)}

    torch.manual_seed(rng_seed)
    label = f"[{desc}] " if desc else ""

    real = torch.as_tensor(x_class, dtype=torch.float32, device=device)
    N_real, D = real.shape
    real_mean = real.mean(dim=0)
    real_cov = compute_covariance(real) if order >= 2 else None

    # real側(定数)の高次Gram量を事前計算
    real_gram_pow = {}
    if order >= 3:
        with torch.no_grad():
            Bc = real - real_mean
            G_BB = Bc @ Bc.t()
            for k in range(3, order + 1):
                real_gram_pow[k] = (G_BB ** k).sum() / (N_real ** 2)

    eps = torch.finfo(real.dtype).eps

    if use_entropy:
        # sigmoid変換によるbox制約(本家 truncated_normal.py 方式)
        if order >= 2:
            init_cov = real_cov + 2e-6 * torch.eye(D, device=device)
            from torch.distributions.multivariate_normal import MultivariateNormal
            while True:
                try:
                    mvn = MultivariateNormal(real_mean, init_cov)
                    break
                except Exception:
                    init_cov = init_cov + 2e-6 * torch.eye(D, device=device)
            x0 = mvn.sample((n_fake_samples,)).clamp(eps, 1 - eps)
        else:
            std = real.std(dim=0, keepdim=True)
            x0 = (real_mean.unsqueeze(0) + std * torch.randn(n_fake_samples, D, device=device)).clamp(eps, 1 - eps)
        z = nn.Parameter(x0.logit())
        optimizer = torch.optim.Adam([z], lr=lr)
    else:
        # ReLU bounds_lossで[0,1]制約(third/fourth_order_gram方式)
        if order >= 2:
            init_cov = real_cov + 2e-6 * torch.eye(D, device=device)
            from torch.distributions.multivariate_normal import MultivariateNormal
            while True:
                try:
                    mvn = MultivariateNormal(real_mean, init_cov)
                    break
                except Exception:
                    init_cov = init_cov + 2e-6 * torch.eye(D, device=device)
            fake = mvn.sample((n_fake_samples,)).clamp(0, 1)
        else:
            std = real.std(dim=0, keepdim=True)
            fake = (real_mean.unsqueeze(0) + std * torch.randn(n_fake_samples, D, device=device)).clamp(0, 1)
        fake.requires_grad_(True)
        optimizer = torch.optim.Adam([fake], lr=lr)

    t0 = time.time()
    for i in range(n_steps):
        optimizer.zero_grad()
        x = z.sigmoid() if use_entropy else fake

        means_loss = (x.mean(dim=0) - real_mean).norm() / real_mean.numel()
        loss = means_loss

        cov_loss = torch.tensor(0.0, device=device)
        if order >= 2:
            cov_loss = (compute_covariance(x) - real_cov).norm() / (D * D)
            loss = loss + cov_loss

        moment_losses = {}
        for k in range(3, order + 1):
            sq = moment_k_sqfrob_gram(x, real, k, B_gram_pow_sum=real_gram_pow[k])
            mloss = torch.sqrt(sq + 1e-24) / (D ** (k / 2))
            moment_losses[k] = mloss
            loss = loss + moment_weights.get(k, 100.0) * mloss

        if use_entropy:
            koleo_val = koleo(x)
            loss = loss - koleo_weight * koleo_val
        else:
            bounds_loss = (torch.relu(x - 1) + torch.relu(-x)).mean()
            loss = loss + bounds_loss

        loss.backward()
        optimizer.step()

        if verbose and (i % log_every == 0 or i == n_steps - 1):
            elapsed = time.time() - t0
            mstr = " ".join(f"m{k}={v.item():.6f}" for k, v in moment_losses.items())
            extra = f"koleo={koleo_val.item():.4f}" if use_entropy else f"bounds={bounds_loss.item():.8f}"
            print(f"{label}step {i:5d}/{n_steps} order={order} loss={loss.item():.6f} "
                  f"means={means_loss.item():.8f} cov={cov_loss.item():.8f} {mstr} {extra} "
                  f"elapsed={elapsed:.1f}s", flush=True)

    with torch.no_grad():
        result = z.sigmoid().detach() if use_entropy else fake.detach()

    if final_eval:
        with torch.no_grad():
            fm = result.mean(dim=0)
            ml = float((fm - real_mean).norm() / real_mean.numel())
            cl = float((compute_covariance(result) - real_cov).norm() / (D * D)) if order >= 2 else None
            mstrs = {}
            for k in range(3, order + 1):
                sq = moment_k_sqfrob_gram(result, real, k, B_gram_pow_sum=real_gram_pow[k])
                mstrs[k] = float(torch.sqrt(sq + 1e-24)) / (D ** (k / 2))
            extra = f"koleo={koleo(result).item():.4f}" if use_entropy else \
                    f"bounds={(torch.relu(result-1)+torch.relu(-result)).mean().item():.8f}"
            print(f"{label}[final] order={order} use_entropy={use_entropy} means={ml:.8f} "
                  f"cov={cl} moments={mstrs} {extra} "
                  f"range=[{result.min():.4f},{result.max():.4f}]", flush=True)

    return result.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------
# 数値検証: Gram trick(任意次数k)が素朴なテンソル構成と一致するか
# ---------------------------------------------------------------
def _naive_moment_sqfrob(A, B, k):
    import string
    Ac = A - A.mean(dim=0)
    Bc = B - B.mean(dim=0)
    n_f, n_r = Ac.shape[0], Bc.shape[0]
    # サンプル軸は 'z' に固定し、出力軸ラベル(a,b,c,...)と衝突しないようにする
    letters = string.ascii_lowercase[0:k]  # k<=6 なら 'abcdef' まで、'z'とは衝突しない
    idx = "z" + ",z".join(letters)
    out = "".join(letters)
    args_A = [Ac] * k
    args_B = [Bc] * k
    TA = torch.einsum(f"{idx}->{out}", *args_A) / n_f
    TB = torch.einsum(f"{idx}->{out}", *args_B) / n_r
    return ((TA - TB) ** 2).sum()


def _verify(device="cpu"):
    torch.manual_seed(0)
    d, n_f, n_r = 8, 15, 18
    A = torch.rand(n_f, d, device=device, dtype=torch.float64)
    B = torch.rand(n_r, d, device=device, dtype=torch.float64)
    for k in range(1, 7):
        naive = _naive_moment_sqfrob(A, B, k).item()
        gram = moment_k_sqfrob_gram(A, B, k).item()
        abs_err = abs(naive - gram)
        if k == 1:
            # k=1は中心化後の平均が恒等的にゼロになる退化ケース(naive≈0)。
            # 相対誤差は分母がほぼ0で発散するため、絶対誤差で判定する。
            print(f"k={k} (degenerate, naive≈0): naive={naive:.3e} gram={gram:.3e} abs_err={abs_err:.3e}")
            assert abs_err < 1e-6, f"k={k} で不一致(絶対誤差)"
        else:
            rel_err = abs_err / max(abs(naive), 1e-12)
            print(f"k={k}: naive={naive:.6f} gram={gram:.6f} rel_err={rel_err:.3e}")
            assert rel_err < 1e-8, f"k={k} で不一致"
    print("OK: k=1..6 すべてGram trickは素朴計算と数値一致しました。")


def _bench(device):
    torch.manual_seed(0)
    N, D = 5000, 3072
    real = torch.rand(N, D, device=device)
    for order in (1, 2, 3, 4, 5, 6):
        fake = torch.rand(N, D, device=device, requires_grad=True)
        real_gram_pow = {}
        if order >= 3:
            with torch.no_grad():
                Bc = real - real.mean(dim=0)
                G_BB = Bc @ Bc.t()
                for k in range(3, order + 1):
                    real_gram_pow[k] = (G_BB ** k).sum() / (N ** 2)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        n_bench = 5
        for _ in range(n_bench):
            if fake.grad is not None:
                fake.grad = None
            loss = (fake.mean(dim=0) - real.mean(dim=0)).norm()
            if order >= 2:
                loss = loss + (compute_covariance(fake) - compute_covariance(real)).norm()
            for k in range(3, order + 1):
                sq = moment_k_sqfrob_gram(fake, real, k, B_gram_pow_sum=real_gram_pow[k])
                loss = loss + torch.sqrt(sq + 1e-24)
            loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_bench
        print(f"order={order}: time/step={dt:.4f}s -> x1500 = {dt*1500/60:.2f} min/class")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verify] device={dev}")
    _verify(dev)
    if dev == "cuda":
        print("\n[bench] N=5000 D=3072, order=1..6:")
        _bench(dev)