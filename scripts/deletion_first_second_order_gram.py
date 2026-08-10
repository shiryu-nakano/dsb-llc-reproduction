"""
deletion_first_second_order_gram.py

1次(means only)・2次(means+cov)制約のdeletionデータを、third_order_gramと
同じ勾配降下・Gram trickの枠組みで生成する。

目的: 既存4手法(conrad/ics=1次, gaussian/truncated_normal=2次)は解析的
サンプリングで生成されているのに対し、third_order_gram(3次)は勾配降下で
生成されている。この「生成方法の違い」が、次数比較(特にLLC)に交絡して
いないかを確認するため、1次・2次も同じ勾配降下方式で作り、
「生成方法を揃えた上での次数依存性」を見られるようにする。

- first_order_gram:  loss = means_loss + bounds_loss
- second_order_gram: loss = means_loss + cov_loss + bounds_loss

third_order_gram.py の third_order_sample_gram から該当項を削っただけの
実装。weightは付けず(means/covのみなら素直に収束するはず)、必要なら
後で導入する。
"""
import time

import numpy as np
import torch


def compute_covariance(X):
    Xc = X - X.mean(dim=0)
    return Xc.t() @ Xc / X.shape[0]


def order_sample_gram(
    x_class,
    n_fake_samples,
    order,                  # 1 または 2
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    device="cpu",
    verbose=False,
    log_every=300,
    final_eval=True,
    desc=None,
):
    """
    order=1: means のみ制約
    order=2: means + cov を制約
    """
    assert order in (1, 2), "order は 1 か 2 のみ対応"

    torch.manual_seed(rng_seed)
    label = f"[{desc}] " if desc else ""

    real = torch.as_tensor(x_class, dtype=torch.float32, device=device)
    N_real, D = real.shape

    real_mean = real.mean(dim=0)
    real_cov = compute_covariance(real) if order >= 2 else None

    # 初期化: order>=2ならreal_covを使ったガウス、order=1ならreal_meanのみ
    # (isotropicな小さい分散から開始し、means項だけで押していく)
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
        del init_cov
    else:
        # order=1: 共分散は制約しないので、実データ全体のピクセル分散
        # 程度のノイズを与えた初期化にする(全くのランダムだと means_loss
        # だけで大きく動かす必要がありすぎるため)
        std = real.std(dim=0, keepdim=True)
        fake = (real_mean.unsqueeze(0) + std * torch.randn(n_fake_samples, D, device=device)).clamp(0, 1)

    fake.requires_grad_(True)
    optimizer = torch.optim.Adam([fake], lr=lr)

    t0 = time.time()
    for i in range(n_steps):
        optimizer.zero_grad()

        fake_mean = fake.mean(dim=0)
        means_loss = (fake_mean - real_mean).norm() / real_mean.numel()
        bounds_loss = (torch.relu(fake - 1) + torch.relu(-fake)).mean()

        if order >= 2:
            fake_cov = compute_covariance(fake)
            cov_loss = (fake_cov - real_cov).norm() / (D * D)
            loss = means_loss + cov_loss + bounds_loss
        else:
            cov_loss = torch.tensor(0.0, device=fake.device)
            loss = means_loss + bounds_loss

        loss.backward()
        optimizer.step()

        if verbose and (i % log_every == 0 or i == n_steps - 1):
            elapsed = time.time() - t0
            print(f"{label}step {i:5d}/{n_steps} order={order} "
                  f"means={means_loss.item():.8f} cov={float(cov_loss):.8f} "
                  f"bounds={bounds_loss.item():.8f} elapsed={elapsed:.1f}s", flush=True)

    result = fake.detach()

    if final_eval:
        with torch.no_grad():
            fm = result.mean(dim=0)
            ml = float((fm - real_mean).norm() / real_mean.numel())
            if order >= 2:
                fcov = compute_covariance(result)
                cl = float((fcov - real_cov).norm() / (D * D))
            else:
                cl = None
            bl = float((torch.relu(result - 1) + torch.relu(-result)).mean())
            print(f"{label}[final] order={order} means={ml:.8f} "
                  f"cov={cl if cl is not None else 'N/A'} bounds={bl:.8f} "
                  f"range=[{result.min():.4f},{result.max():.4f}]", flush=True)

    return result.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------
# 動作確認用(小規模)
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 200, 60
    x = torch.rand(N, D).numpy()
    for order in (1, 2):
        print(f"\n=== order={order} smoke test ===")
        s = order_sample_gram(x, N, order=order, n_steps=200, device="cpu",
                               verbose=True, log_every=50, desc=f"smoke_o{order}")
        print("shape:", s.shape, "range:", s.min(), s.max())