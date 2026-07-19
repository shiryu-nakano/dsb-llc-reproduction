"""
deletion_third_order.py

3rd order hypercube-constrained max-entropy sampling (Deletion, 3次)

features-across-time/scripts/preprocess/third_order_approx.py の
アルゴリズムに忠実に移植したもの。

フルの (D,D,D) coskewnessテンソルを計算せず、次元をスライスに分割して
逐次計算・逐次バックプロパゲーションすることでメモリを節約する。
"""
import numpy as np
import torch


def compute_covariance(X):
    """X: (N, D) -> (D, D) 共分散行列 (Nで割る、不偏推定ではない点に注意、本家仕様)"""
    X_centered = X - X.mean(dim=0)
    covariance = X_centered.t() @ X_centered / X.shape[0]
    return covariance


def compute_coskewness_slice(X, slice_start_index, slice_end_index, slice_dim=0):
    """
    フルの (D,D,D) coskewnessテンソルの代わりに、
    slice_dim方向の [slice_start_index:slice_end_index] だけの
    部分テンソル (k, D, D) を計算する。

    X: (N, D)
    """
    X_centered = X - X.mean(dim=0)
    if slice_dim == 0:
        X_sliced = X_centered[:, slice_start_index:slice_end_index]
        coskewness_slice = torch.einsum('ij,ik,il->jkl', X_sliced, X_centered, X_centered) / X.shape[0]
    elif slice_dim == 1:
        X_sliced = X_centered[:, slice_start_index:slice_end_index]
        coskewness_slice = torch.einsum('ij,ik,il->jkl', X_centered, X_sliced, X_centered) / X.shape[0]
    elif slice_dim == 2:
        X_sliced = X_centered[:, slice_start_index:slice_end_index]
        coskewness_slice = torch.einsum('ij,ik,il->jkl', X_centered, X_centered, X_sliced) / X.shape[0]
    else:
        raise ValueError("slice_dim must be 0, 1, or 2")
    return coskewness_slice


def third_order_sample(
    x_class,           # (N_c, D) 対象クラスの実データ(flatten済み)
    n_fake_samples,
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    slice_dim=0,
    slice_size=1,
    device="cpu",
    verbose=False,
):
    """
    本家 third_order_approx.py のロジックをそのまま移植。

    x_class: (N_c, D) 実データ(このクラスの平均・共分散・coskewnessの目標元)
    D: 画像の次元数 (本家の h に相当。ここでは flatten済みの全次元数)
    """
    torch.manual_seed(rng_seed)

    real = torch.tensor(x_class, dtype=torch.float32, device=device)
    D = real.shape[1]

    real_mean = real.mean(dim=0)
    real_cov = compute_covariance(real)

    # 正定値性の確保 (本家と同じepsilon)
    real_cov = real_cov + 2e-6 * torch.eye(D, device=device)

    from torch.distributions.multivariate_normal import MultivariateNormal
    while True:
        try:
            mvn = MultivariateNormal(real_mean, real_cov)
            break
        except Exception:
            real_cov = real_cov + 2e-6 * torch.eye(D, device=device)

    fake = mvn.sample((n_fake_samples,))
    fake = torch.clamp(fake, 0, 1)
    del real_cov  # 本家同様、以降は都度 compute_covariance で再計算する

    fake.requires_grad = True
    optimizer = torch.optim.Adam([fake], lr=lr)

    # h は本家では画像の一辺のサイズ(スライス対象の次元数)。
    # ここでは flatten済みの全次元数 D をそのままスライス対象とする。
    h = D
    num_slices = (h + slice_size - 1) // slice_size

    for i in range(n_steps):
        optimizer.zero_grad()

        fake_mean = fake.mean(dim=0)
        fake_cov = compute_covariance(fake)

        coskew_loss = torch.tensor(0.0, device=device)
        for j in range(0, h, slice_size):
            start_index = j
            end_index = min(start_index + slice_size, h)

            fake_coskew = compute_coskewness_slice(fake, start_index, end_index, slice_dim)
            real_coskew = compute_coskewness_slice(real, start_index, end_index, slice_dim)

            slice_loss = (fake_coskew - real_coskew).norm() / real_coskew.numel()
            slice_loss = slice_loss / num_slices

            slice_loss.backward()
            coskew_loss = coskew_loss + slice_loss.detach()

        means_loss = (fake_mean - real_mean).norm() / real_mean.numel()
        cov_loss_recomputed = compute_covariance(fake)  # 再計算(本家通り、backward後に評価)
        cov_loss = (fake_cov - (compute_covariance(real))).norm() / (D * D)

        bounds_loss = (torch.relu(fake - 1) + torch.relu(-fake)).mean()

        loss = means_loss + cov_loss + bounds_loss
        loss.backward()
        optimizer.step()

        if verbose and i % 100 == 0:
            print(f"    step {i}: total={loss.item()+coskew_loss.item():.6f} "
                  f"means={means_loss.item():.6f} cov={cov_loss.item():.6f} "
                  f"coskew={coskew_loss.item():.6f} bounds={bounds_loss.item():.6f}")

    with torch.no_grad():
        final_samples = fake.cpu().numpy()

    return final_samples.astype(np.float32)
