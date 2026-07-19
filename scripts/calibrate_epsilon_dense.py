"""
calibrate_epsilon_dense.py

train_and_save.py (Refinetti学習レシピ、10クラスCIFAR10) で保存した
dsb_dense のチェックポイントに対して、SGLDのステップサイズ epsilon を
MALA acceptance probability を診断指標にしてキャリブレーションする。

Lau et al. (2025) Appendix H の推奨:
  - 平均 MALA acceptance probability が 0.9〜0.95 の範囲になるよう
    epsilon を調整する。
  - loss trace が burn-in 後に平坦になっているかも目視で確認する。
  - gamma・batch_size・num_steps は固定し、epsilon のみを振る。

注意: run_sgld が返す accept_probs は [t, prob] のペアのリストである
      (t: SGLDステップ番号, prob: そのステップでのMALA受理確率)。
      平均を取る際は prob 部分だけを抽出すること。

使用例:
  cd ~/cuda_test/scripts
  XLA_FLAGS="--xla_gpu_autotune_level=0" CUDA_VISIBLE_DEVICES=0 python calibrate_epsilon_dense.py \
      --ckpt_dir /home/nakano/server/checkpoints_dense/dsb_dense/seed42 \
      --steps 1 100 1000 18587 78000 \
      --epsilons 2e-9 1e-8 5e-8 1e-7 2e-7 5e-7 1e-6 \
      --out_dir /home/nakano/server/calibration_results_dense \
      --plot
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LLC_SRC = Path.home() / "src" / "github.com" / "shiryu-nakano" / "llc_reproduction" / "src"
sys.path.insert(0, str(LLC_SRC))

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from train_and_save import make_resnet18
from cifar10_loader import load_cifar10
from sgld_utils import SGLDConfig, run_sgld


def compute_loss(params, state, model, rng, x, y, num_classes, is_training=False):
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    logits, new_state = model.apply(params, state, rng, x, is_training)
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
    return loss_val, new_state


def stratified_sample_by_class(x, y, n_total, seed=42):
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


def extract_accept_probs(accept_probs_raw):
    """
    run_sgld が返す accept_probs は [t, prob] のペアのリスト。
    prob 部分だけを抽出して 1次元配列として返す。
    """
    if len(accept_probs_raw) == 0:
        return np.array([])
    arr = np.array(accept_probs_raw)  # shape: (num_records, 2) = [[t0, p0], [t1, p1], ...]
    return arr[:, 1]  # prob列だけ取り出す


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seed_label", type=str, default=None,
                         help="どのtrain_seedのckpt_dirか記録用のラベル(例: seed42)")
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                         help="キャリブレーションに使うチェックポイントのstep番号")
    parser.add_argument("--epsilons", type=float, nargs="+", required=True,
                         help="試すepsilonの候補リスト")
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/calibration_results_dense")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                         help="SGLDチェーン自体の乱数seed(train_seedとは別物)")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)
    parser.add_argument("--n_calibration_samples", type=int, default=50000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_label = args.seed_label or Path(args.ckpt_dir).name

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    if args.n_calibration_samples < len(x_train):
        x_calib, y_calib = stratified_sample_by_class(
            x_train, y_train, args.n_calibration_samples, seed=args.seed)
    else:
        x_calib, y_calib = x_train, y_train

    n_calib = len(x_calib)
    itemp = 1 / np.log(n_calib)
    print(f"Calibration data: n={n_calib}, itemp={itemp:.6f}")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()

    for step in args.steps:
        print(f"\n{'='*60}")
        print(f"Calibrating {seed_label} at step={step}")
        print(f"{'='*60}")

        ckpt_path = Path(args.ckpt_dir).expanduser() / f"step_{step:07d}"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for step={step} not found, skipping.")
            continue

        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        rng = jax.random.PRNGKey(args.seed)

        results = []
        for epsilon in args.epsilons:
            print(f"\n--- epsilon={epsilon:.2e} ---")
            rng, subkey = jax.random.split(rng)

            sgld_config = SGLDConfig(
                epsilon=epsilon, gamma=args.gamma,
                num_steps=args.sgld_num_steps, batch_size=args.sgld_batch_size,
            )

            loss_fn = lambda p, x, y: compute_loss(
                p, state, model, rng, x, y, args.num_classes, is_training=False)[0]

            loss_trace, distances, acceptance_probs_raw = run_sgld(
                subkey, loss_fn, sgld_config, params,
                jnp.array(x_calib), jnp.array(y_calib),
                itemp=itemp, trace_batch_loss=True,
                compute_distance=False,
                compute_mala_acceptance=True,
                verbose=False,
            )

            loss_trace_arr = np.array(loss_trace)
            valid_loss = loss_trace_arr[~np.isnan(loss_trace_arr) & ~np.isinf(loss_trace_arr)]

            # [t, prob] のペアから prob だけを取り出す
            probs_only = extract_accept_probs(acceptance_probs_raw)

            if len(valid_loss) == 0:
                print("  DIVERGED (all NaN/Inf)")
                mean_accept = float("nan")
                loss_pre = float("nan")
                loss_post = float("nan")
            else:
                burn_in = int(len(loss_trace_arr) * 0.9)
                mean_accept = float(np.mean(probs_only)) if len(probs_only) > 0 else float("nan")
                loss_pre  = float(np.mean(valid_loss[:burn_in])) if burn_in > 0 else float("nan")
                loss_post = float(np.mean(valid_loss[burn_in:])) if len(valid_loss) > burn_in else float("nan")
                print(f"  mean_accept_prob = {mean_accept:.4f}")
                print(f"  loss_mean_pre_burnin  = {loss_pre:.4f}")
                print(f"  loss_mean_post_burnin = {loss_post:.4f}")
                if not np.isnan(mean_accept):
                    print(f"  in_target_range_0.9_0.95 = {0.9 <= mean_accept <= 0.95}")

            results.append({
                "epsilon": epsilon,
                "mean_accept_prob": mean_accept,
                "in_target_range_0.9_0.95": bool(0.9 <= mean_accept <= 0.95) if not np.isnan(mean_accept) else False,
                "loss_mean_pre_burnin": loss_pre,
                "loss_mean_post_burnin": loss_post,
                "n_accept_prob_records": len(probs_only),
                "loss_trace": [float(l) for l in loss_trace],
            })

        out_path = out_dir / f"calib_dense_{seed_label}_step{step}.json"
        with open(out_path, "w") as f:
            json.dump({
                "step": step,
                "seed_label": seed_label,
                "ckpt_dir": str(args.ckpt_dir),
                "gamma": args.gamma,
                "sgld_num_steps": args.sgld_num_steps,
                "sgld_batch_size": args.sgld_batch_size,
                "n_calibration_samples": n_calib,
                "results": results,
            }, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
