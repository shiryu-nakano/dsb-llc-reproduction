"""
calibrate_epsilon_binary.py

train_and_save_binary.py で保存した2値分類(First5 vs Last5)の
チェックポイントに対して、SGLDのステップサイズ epsilon を
MALA acceptance probability を診断指標にしてキャリブレーションする。

Lau et al. (2025) Appendix H の推奨:
  - 平均 MALA acceptance probability が 0.9〜0.95 の範囲になるよう
    epsilon を調整する。
  - それとは別に、loss trace が burn-in 後に平坦になっているかも
    目視で確認する。
  - gamma・batch_size・num_steps は固定し、epsilon のみを振る
    (10クラス版で確認済みの値を出発点にする: gamma=1.0,
     batch_size=2048, num_steps=3000)。

使い方の想定:
  学習中盤〜終盤の1点 (例: t=18000) と、学習初期の1点 (例: t=1500) の
  少なくとも2チェックポイントで実行し、両者で妥当な epsilon が
  一致するか確認する。一致しなければ、単一の epsilon では
  全チェックポイントをカバーできないことになるので、その旨を
  報告する。

使用例:
  cd ~/cuda_test/scripts
  XLA_FLAGS="--xla_gpu_autotune_level=0" CUDA_VISIBLE_DEVICES=1 python calibrate_epsilon_binary.py \
      --ckpt_dir ~/cuda_test/checkpoints/nakkiran_binary_resnet18/seed42 \
      --steps 1500 18000 \
      --epsilons 2e-8 2e-7 2e-6 2e-5 \
      --out_dir ~/cuda_test/calibration_results \
      --plot
"""

import sys
from pathlib import Path

# sgld_utils / SGLDConfig は既存の LLC 再現実験リポジトリからそのまま import する
# (コピーはしない。eval_llc_cifar10.py と同じパターン)
LLC_SRC = Path.home() / "src" / "github.com" / "shiryu-nakano" / "llc_reproduction" / "src"
sys.path.insert(0, str(LLC_SRC))

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

# 2値分類固有のもの (モデル・データ・標準化) は同じ scripts/ ディレクトリの
# train_and_save_binary.py からそのまま import する
from train_and_save_binary import (
    make_resnet18,
    load_cifar10_binary,
    per_image_standardize,
)
from sgld_utils import SGLDConfig, run_sgld


def compute_loss(params, state, model, rng, x, y, num_classes, is_training=False):
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    logits, new_state = model.apply(params, state, rng, x, is_training)
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
    return loss_val, new_state


def stratified_sample_by_class(x, y, n_total, seed=42):
    """クラスごとに均等抽出してn_totalに揃える (eval_llc_cifar10.py と同じロジック)"""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_per_class = max(1, n_total // len(classes))

    selected_indices = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_take = min(n_per_class, len(idx_c))
        chosen = rng.choice(idx_c, n_take, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)
    return x[selected_indices], y[selected_indices]


def plot_diagnostic(loss_trace, acceptance_probs, step, epsilon, out_path):
    """Lau et al. Figure H.1 と同じ形式: loss trace (左軸) + MALA acceptance probability (右軸)"""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(loss_trace, color="tab:blue", alpha=0.8, label="loss trace")
    ax1.set_xlabel("Num SGLD Steps")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if len(acceptance_probs) > 0:
        probs = np.array(acceptance_probs)
        ax2 = ax1.twinx()
        ax2.plot(probs[:, 0], probs[:, 1], "rx", alpha=0.6, label="MALA acceptance prob")
        ax2.set_ylabel("Acceptance Probability", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(-0.05, 1.05)
        mean_prob = probs[:, 1].mean()
    else:
        mean_prob = float("nan")

    fig.suptitle(f"step={step}, epsilon={epsilon:.2e}, mean_accept_prob={mean_prob:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True,
                         help="train_and_save_binary.py が保存したチェックポイントのディレクトリ "
                              "(例: ~/cuda_test/checkpoints/nakkiran_binary_resnet18/seed42)")
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                         help="キャリブレーションに使うチェックポイントのstep番号 "
                              "(学習初期の1点+中盤〜終盤の1点を推奨、例: 1500 18000)")
    parser.add_argument("--epsilons", type=float, nargs="+",
                         default=[2e-8, 2e-7, 2e-6, 2e-5],
                         help="振る epsilon の候補 (1桁刻み推奨)")
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/calibration_results")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # SGLD ハイパラ: epsilon 以外は固定 (10クラス版で確認済みの値を出発点にする)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)
    parser.add_argument("--sgld_logging_period", type=int, default=200)

    # 評価データ (キャリブレーション用に全訓練データを使わず層化抽出でサブサンプルする)
    parser.add_argument("--stratified", action="store_true", default=True)
    parser.add_argument("--n_total", type=int, default=50000,
                         help="キャリブレーションに使うサンプル数 (層化抽出、両クラス均等)")

    parser.add_argument("--plot", action="store_true",
                         help="loss trace + acceptance probability の診断プロットをPDFで保存する")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10 (binary: First5 vs Last5)...")
    x_train, y_train, x_test, y_test = load_cifar10_binary()

    if args.stratified:
        x_eval, y_eval = stratified_sample_by_class(
            x_train, y_train, args.n_total, seed=args.seed)
    else:
        x_eval, y_eval = x_train, y_train

    # 学習時と同じ前処理 (per-image standardization) を適用する。
    # SGLD 自体は拡張 (crop/flip) を行わないので、ここでは標準化のみでよい。
    x_eval = per_image_standardize(np.array(x_eval))
    print(f"Calibration data shape: {x_eval.shape} "
          f"(label distribution: {np.bincount(y_eval)})")

    n_data = len(x_eval)
    itemp = 1 / np.log(n_data)
    print(f"n = {n_data}, itemp (1/log(n)) = {itemp:.6f}")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(args.seed)

    summary = []  # (step, epsilon, mean_accept_prob) の一覧。最後にまとめて表示する。

    for step in args.steps:
        ckpt_path = Path(args.ckpt_dir).expanduser() / f"step_{step:07d}"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for step={step} not found ({ckpt_path}), skipping.")
            continue

        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        results_this_step = []

        for epsilon in args.epsilons:
            sgld_config = SGLDConfig(
                epsilon=epsilon,
                gamma=args.gamma,
                num_steps=args.sgld_num_steps,
                batch_size=args.sgld_batch_size,
            )

            rng, subkey = jax.random.split(rng)
            loss_fn = lambda p, x, y: compute_loss(
                p, state, model, rng, x, y, args.num_classes, is_training=False)[0]

            loss_trace, distances, acceptance_probs = run_sgld(
                subkey, loss_fn, sgld_config, params,
                jnp.array(x_eval), jnp.array(y_eval),
                itemp=itemp, trace_batch_loss=True,
                compute_distance=False,
                compute_mala_acceptance=True,   # ★ ここが診断の要 (デフォルトFalseなので明示的にTrue)
                verbose=False,
                logging_period=args.sgld_logging_period,
            )

            loss_trace_list = [float(l) for l in loss_trace]
            probs = np.array(acceptance_probs) if len(acceptance_probs) > 0 else np.zeros((0, 2))
            mean_accept_prob = float(probs[:, 1].mean()) if len(probs) > 0 else float("nan")

            # burn-in (90%) 前後で loss の平均を比較し、平坦化しているかの簡易診断値を出す
            burnin = int(len(loss_trace_list) * 0.9)
            mean_pre_burnin = float(np.mean(loss_trace_list[:burnin])) if burnin > 0 else float("nan")
            mean_post_burnin = float(np.mean(loss_trace_list[burnin:]))

            in_target_range = 0.9 <= mean_accept_prob <= 0.95

            print(
                f"step={step:6d} | epsilon={epsilon:.2e} "
                f"| mean_accept_prob={mean_accept_prob:.4f} "
                f"{'<- 0.9-0.95 OK' if in_target_range else ''} "
                f"| loss(pre-burnin)={mean_pre_burnin:.4f} "
                f"| loss(post-burnin)={mean_post_burnin:.4f}"
            )

            rec = {
                "step": step,
                "epsilon": epsilon,
                "mean_accept_prob": mean_accept_prob,
                "in_target_range_0.9_0.95": in_target_range,
                "loss_mean_pre_burnin": mean_pre_burnin,
                "loss_mean_post_burnin": mean_post_burnin,
                "loss_trace": loss_trace_list,
                "acceptance_probs": probs.tolist(),
            }
            results_this_step.append(rec)
            summary.append((step, epsilon, mean_accept_prob, in_target_range))

            if args.plot:
                plot_path = out_dir / f"calib_step{step}_eps{epsilon:.0e}.pdf"
                plot_diagnostic(loss_trace_list, acceptance_probs, step, epsilon, plot_path)

        out_path = out_dir / f"calib_step{step}.json"
        with open(out_path, "w") as f:
            json.dump({
                "step": step,
                "gamma": args.gamma,
                "sgld_num_steps": args.sgld_num_steps,
                "sgld_batch_size": args.sgld_batch_size,
                "n_calibration_samples": n_data,
                "results": results_this_step,
            }, f, indent=2)
        print(f"Saved: {out_path}")

    # ── サマリー表示: どの epsilon が全チェックポイントで 0.9〜0.95 に収まっているか ──
    print("\n" + "=" * 70)
    print("SUMMARY: mean MALA acceptance probability")
    print("=" * 70)
    steps_seen = sorted(set(s for s, _, _, _ in summary))
    epsilons_seen = sorted(set(e for _, e, _, _ in summary))
    header = "epsilon".ljust(12) + "".join(f"step={s}".rjust(16) for s in steps_seen)
    print(header)
    for eps in epsilons_seen:
        row = f"{eps:.2e}".ljust(12)
        all_ok = True
        for s in steps_seen:
            match = [(p, ok) for (ss, ee, p, ok) in summary if ss == s and ee == eps]
            if match:
                p, ok = match[0]
                row += f"{p:.4f}{'*' if ok else ' '}".rjust(16)
                all_ok = all_ok and ok
            else:
                row += "N/A".rjust(16)
                all_ok = False
        row += "  <- 全ステップで0.9-0.95" if all_ok else ""
        print(row)
    print("(* は 0.9-0.95 の目標範囲内)")


if __name__ == "__main__":
    main()
