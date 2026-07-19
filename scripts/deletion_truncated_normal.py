"""
deletion_truncated_normal.py

Truncated normal (Deletion, 2次、hypercube制約付き)
論文2.4節「Approximate Sampling」。
勾配ベース最適化(Adam)で平均・共分散を合わせつつ
KoLeoエントロピー推定量を最大化する。
"""
import numpy as np
import torch


def truncated_normal_sample(
    n_samples, mean, cov, rng,
    n_iter=500, lr=0.1, koleo_weight=1.0, device="cpu",
):
    D = len(mean)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    cov_t = torch.tensor(cov, dtype=torch.float32, device=device)

    try:
        L = torch.linalg.cholesky(cov_t)
    except Exception:
        eigvals, eigvecs = torch.linalg.eigh(cov_t)
        eigvals = eigvals.clamp_min(1e-6)
        cov_fixed = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        L = torch.linalg.cholesky(cov_fixed)

    z0 = torch.randn(n_samples, D, device=device) @ L.T + mean_t
    z0 = z0.clamp(1e-4, 1 - 1e-4)
    logits = torch.logit(z0).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([logits], lr=lr)


    def koleo_entropy_estimate(x, eps=1e-8):
        """Kozachenko-Leonenko最近傍距離ベースのエントロピー推定量"""
        dists = torch.cdist(x, x)
        # インプレースの fill_diagonal_ を避け、マスクで対角成分を除外する
        n = dists.shape[0]
        mask = torch.eye(n, dtype=torch.bool, device=dists.device)
        dists_masked = dists.masked_fill(mask, float("inf"))  # 非インプレース版
        nn_dist = dists_masked.min(dim=1).values
        return torch.log(nn_dist + eps).mean()

    for i in range(n_iter):
        optimizer.zero_grad()
        x = torch.sigmoid(logits)

        cur_mean = x.mean(dim=0)
        centered = x - cur_mean
        cur_cov = (centered.T @ centered) / (n_samples - 1)

        mean_loss = ((cur_mean - mean_t) ** 2).sum()
        cov_loss = ((cur_cov - cov_t) ** 2).sum()
        entropy = koleo_entropy_estimate(x)

        loss = mean_loss + cov_loss - koleo_weight * entropy
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_samples = torch.sigmoid(logits).cpu().numpy()

    return final_samples.astype(np.float32)
