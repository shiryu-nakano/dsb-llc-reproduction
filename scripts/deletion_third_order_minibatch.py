"""
deletion_third_order_minibatch.py  (ミニバッチ近似版)

3rd order hypercube-constrained max-entropy sampling (Deletion, 3次)

coskewness の backward 演算量は fake サンプル数 N に比例するため、full(N=5000)
だと 1 step ~15s と重い。ここでは各ステップで fake から coskew_batch 枚を
ランダム抽出して coskewness を計算・backward する確率的近似を導入し、演算量を
N/coskew_batch 倍だけ削減する。SGD 同様、多数ステップの平均として正しい方向へ
収束する。生成物は fake 全 n_fake_samples 枚。

- means/cov/bounds は全 fake で計算 (軽いので近似しない)
- coskewness は fake のミニバッチ vs 実データ全体(no_grad, 固定)で計算
- 数値検証: deletion_third_order_fast.py の compute_coskewness_slice と同一実装

注意: これは本家アルゴリズムの確率的近似。最終品質は全 fake の subloss で確認する。
"""
import sys
import time

import numpy as np
import torch


def compute_covariance(X):
    Xc = X - X.mean(dim=0)
    return Xc.t() @ Xc / X.shape[0]


def _coskewness_slice_cross(A, B, s, e):
    """
    A の [s:e] 列を slice 軸に、A,B の中心化データから coskewness スライスを計算。
    out[a,k,l] = (1/N) Σ_n Ac[n,s+a] Ac[n,k] Ac[n,l]  (A==B のとき通常の coskewness)

    ここでは fake の coskewness を測るのに fake 自身の中心で計算する
    (A=B=fake_batch)。実データ側も同様に real 全体で計算する。
    Python ループ無し、中間 (N, k, D)。
    """
    N = A.shape[0]
    Ac = A - A.mean(dim=0)
    Bc = B - B.mean(dim=0)
    As = Ac[:, s:e]                               # (N, k)
    W = As.unsqueeze(2) * Bc.unsqueeze(1)         # (N, k, D)
    out = torch.einsum('nak,nl->akl', W, Bc) / N  # (k, D, D)
    return out


def third_order_sample(
    x_class,
    n_fake_samples,
    rng_seed=42,
    n_steps=1500,
    lr=0.01,
    slice_size=16,
    coskew_batch=1000,     # 毎ステップ coskewness に使う fake サンプル数
    device="cpu",
    verbose=False,
    log_every=100,
    final_eval=True,       # 終了時に全 fake で subloss を測る
    desc=None,              # tqdm バーに出すラベル(例: "class1")
):
    torch.manual_seed(rng_seed)
    g = torch.Generator(device=device).manual_seed(rng_seed)

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
    cb = min(coskew_batch, n_fake_samples)

    if verbose:
        print(f"  [setup] N_real={N_real} D={D} n_fake={n_fake_samples} "
              f"coskew_batch={cb} slice_size={slice_size} num_slices={num_slices} "
              f"n_steps={n_steps} device={device}", flush=True)

    t_loop_start = time.time()
    t_last_log = t_loop_start

    label = f"[{desc}] " if desc else ""

    for i in range(n_steps):
        optimizer.zero_grad()

        # ---- 3次: fake・real 両方をミニバッチ化 ----
        # real 側を全 N_real(=5000) で毎ステップ計算すると、fake 側を
        # 減らしても real 側の演算量がボトルネックのまま残ってしまう
        # (実測: fake だけ 5000->500 にしても 1step ~20s のまま変化なし)。
        # real は学習中不変な母集団なので、その3次統計量の推定に全サンプルは
        # 不要 -> fake と同様に毎ステップ異なる部分集合でサブサンプルする
        # (真の統計量に対する不偏推定量なので、十分なバッチサイズがあれば
        #  収束先の期待値は変わらない)。
        # coskew_batch >= N の場合はサンプリングせず全データを確定的に使う。
        # (torch.randint は重複ありサンプリングなので、cb==N でも「毎回違う
        #  重複ありの5000個」になってしまい、真の「ノイズなし」にならない。
        #  ここで明示的に分岐し、cb>=N なら fast.py の正確版と完全一致させる。)
        use_full_fake = cb >= n_fake_samples
        use_full_real = min(cb, N_real) >= N_real
        if not use_full_fake:
            fake_idx = torch.randint(0, n_fake_samples, (cb,), generator=g, device=device)
        if not use_full_real:
            real_idx = torch.randint(0, N_real, (min(cb, N_real),), generator=g, device=device)

        coskew_val = 0.0
        for j in range(0, h, slice_size):
            s, e = j, min(j + slice_size, h)
            fake_batch = fake if use_full_fake else fake[fake_idx]   # スライスごとに独立グラフ
            fc = _coskewness_slice_cross(fake_batch, fake_batch, s, e)
            with torch.no_grad():
                real_batch = real if use_full_real else real[real_idx]
                rc = _coskewness_slice_cross(real_batch, real_batch, s, e)
            slice_loss = (fc - rc).norm() / rc.numel() / num_slices
            slice_loss.backward()
            coskew_val += float(slice_loss.detach())

        # ---- 1次・2次・境界: 全 fake ----
        fake_mean = fake.mean(dim=0)
        fake_cov = compute_covariance(fake)
        means_loss = (fake_mean - real_mean).norm() / real_mean.numel()
        cov_loss = (fake_cov - real_cov).norm() / (D * D)
        bounds_loss = (torch.relu(fake - 1) + torch.relu(-fake)).mean()
        (means_loss + cov_loss + bounds_loss).backward()

        optimizer.step()

        if verbose:
            now = time.time()
            elapsed_total = now - t_loop_start
            total_now = means_loss.item() + cov_loss.item() + coskew_val + bounds_loss.item()

            # 毎ステップ: 軽量な1行を \r で上書き表示(常に「今どこか」が見える)。
            # ただし標準出力がファイルにリダイレクトされている場合(tty でない)は
            # \r がそのまま文字として残りログが読みにくくなるため、その時は省略し、
            # log_every おきの詳細ブロックだけを残す。
            if sys.stdout.isatty():
                sec_per_step_running = elapsed_total / (i + 1)
                eta_running = sec_per_step_running * (n_steps - 1 - i)
                print(f"\r{label}step {i+1:5d}/{n_steps} ({100*(i+1)/n_steps:5.1f}%) "
                      f"total={total_now:.6f} coskew(mb)={coskew_val:.6f} "
                      f"| {sec_per_step_running:.3f}s/step ETA={eta_running/60:.1f}min   ",
                      end="", flush=True)

            # log_every おき: 内訳・実測速度・メモリを改行付きで詳細表示
            if i % log_every == 0 or i == n_steps - 1:
                interval = now - t_last_log
                steps_done_in_interval = log_every if i > 0 else 1
                sec_per_step = interval / max(steps_done_in_interval, 1)
                t_last_log = now

                mem_str = ""
                if device == "cuda" and torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    mem_str = f" | mem alloc={alloc:.2f}GB reserved={reserved:.2f}GB"

                print(f"\n{label}  [detail] step {i:5d}/{n_steps} "
                      f"means={means_loss.item():.6f} cov={cov_loss.item():.6f} "
                      f"coskew(mb)={coskew_val:.6f} bounds={bounds_loss.item():.6f} "
                      f"| interval {sec_per_step:.3f}s/step elapsed={elapsed_total/60:.1f}min"
                      f"{mem_str}", flush=True)

    result = fake.detach()

    # ---- 最終品質: 全 fake での subloss (近似なしの真の一致度) ----
    if final_eval:
        with torch.no_grad():
            fm = result.mean(dim=0)
            fcov = compute_covariance(result)
            means_l = (fm - real_mean).norm() / real_mean.numel()
            cov_l = (fcov - real_cov).norm() / (D * D)
            coskew_l = 0.0
            for j in range(0, h, slice_size):
                s, e = j, min(j + slice_size, h)
                fc = _coskewness_slice_cross(result, result, s, e)
                rc = _coskewness_slice_cross(real, real, s, e)
                coskew_l += float((fc - rc).norm() / rc.numel() / num_slices)
            bounds_l = (torch.relu(result - 1) + torch.relu(-result)).mean().item()
            print(f"  [final full-fake subloss] means={means_l:.6f} cov={cov_l:.6f} "
                  f"coskew={coskew_l:.6f} bounds={bounds_l:.6f} "
                  f"range=[{result.min():.4f},{result.max():.4f}]", flush=True)

    return result.cpu().numpy().astype(np.float32)