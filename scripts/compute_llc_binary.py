"""
compute_llc_binary.py

train_and_save_binary.py で保存した2値分類(First5 vs Last5)の
全チェックポイントに対して、SGLD-based LLC estimator でλを計算する。

epsilon 等の SGLD ハイパラは calibrate_epsilon_binary.py で決定した
単一の値を --epsilon で明示的に渡すことを想定 (デフォルト値は
10クラス版で確認済みの ε=2e-7 だが、キャリブレーション結果で
上書きすること)。

使用例:
  cd ~/cuda_test/scripts
  XLA_FLAGS="--xla_gpu_autotune_level=0" CUDA_VISIBLE_DEVICES=1 python compute_llc_binary.py \
      --ckpt_dir ~/cuda_test/checkpoints/nakkiran_binary_resnet18/seed42 \
      --steps $(seq 0 100 20000) \
      --epsilon 2e-7 \
      --out_dir ~/cuda_test/llc_results \
      --out_name nakkiran_binary_llc_full
"""

import sys
from pathlib import Path

# sgld_utils / SGLDConfig は既存の LLC 再現実験リポジトリからそのまま import する
LLC_SRC = Path.home() / "src" / "github.com" / "shiryu-nakano" / "llc_reproduction" / "src"
sys.path.insert(0, str(LLC_SRC))

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                         help="LLCを計算するstep番号のリスト "
                              "(例: $(seq 0 100 20000) で全201点)")
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/llc_results")
    parser.add_argument("--out_name", type=str, default="nakkiran_binary_llc_full")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # SGLD ハイパラ: epsilon は calibrate_epsilon_binary.py の結果で
    # 明示的に上書きすること。他は calibrate 時と同じ値で固定する。
    parser.add_argument("--epsilon", type=float, default=2e-7,
                         help="calibrate_epsilon_binary.py で決定した値を渡すこと")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)

    parser.add_argument("--stratified", action="store_true", default=True)
    parser.add_argument("--n_total", type=int, default=50000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.json"

    print("Loading CIFAR-10 (binary: First5 vs Last5)...")
    x_train, y_train, x_test, y_test = load_cifar10_binary()

    if args.stratified:
        x_eval, y_eval = stratified_sample_by_class(
            x_train, y_train, args.n_total, seed=args.seed)
    else:
        x_eval, y_eval = x_train, y_train

    # 学習時と同じ前処理 (per-image standardization)
    x_eval = per_image_standardize(np.array(x_eval))
    print(f"LLC eval data shape: {x_eval.shape} "
          f"(label distribution: {np.bincount(y_eval)})")

    n_data = len(x_eval)
    itemp = 1 / np.log(n_data)
    print(f"n = {n_data}, itemp (1/log(n)) = {itemp:.6f}")
    print(f"Using epsilon = {args.epsilon:.2e} "
          f"(calibrate_epsilon_binary.py の結果を反映していることを確認すること)")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(args.seed)

    sgld_config = SGLDConfig(
        epsilon=args.epsilon, gamma=args.gamma,
        num_steps=args.sgld_num_steps, batch_size=args.sgld_batch_size,
    )

    results = []
    t_start_all = time.time()

    for i, step in enumerate(args.steps):
        ckpt_path = Path(args.ckpt_dir).expanduser() / f"step_{step:07d}"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for step={step} not found, skipping.")
            continue

        t0 = time.time()
        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        rng, subkey = jax.random.split(rng)
        loss_fn = lambda p, x, y: compute_loss(
            p, state, model, rng, x, y, args.num_classes, is_training=False)[0]

        # 本番計算では MALA 診断は不要 (compute_mala_acceptance=False)
        loss_trace, distances, acceptance_probs = run_sgld(
            subkey, loss_fn, sgld_config, params,
            jnp.array(x_eval), jnp.array(y_eval),
            itemp=itemp, trace_batch_loss=True,
            compute_distance=False, compute_mala_acceptance=False,
            verbose=False,
        )

        init_loss = float(loss_fn(params, jnp.array(x_eval), jnp.array(y_eval)))
        loss_trace_list = [float(l) for l in loss_trace]

        # burn-in 90% を切ってからの平均で drLLC/LLC 相当のλを推定
        burnin = int(len(loss_trace_list) * 0.9)
        mean_post_burnin = float(np.mean(loss_trace_list[burnin:]))
        lambdahat = (mean_post_burnin - init_loss) * n_data * itemp

        elapsed = time.time() - t0
        results.append({
            "step": step,
            "init_loss": init_loss,
            "mean_loss_post_burnin": mean_post_burnin,
            "lambdahat": lambdahat,
        })
        eta_sec = elapsed * (len(args.steps) - i - 1)
        print(
            f"[{i+1}/{len(args.steps)}] step={step:6d} "
            f"| init_loss={init_loss:.4f} | lambdahat={lambdahat:.2f} "
            f"| elapsed={elapsed:.1f}s | ETA={eta_sec/60:.1f}min"
        )

        # 途中経過をこまめに保存 (長時間実行中の万一のクラッシュに備える)
        with open(out_path, "w") as f:
            json.dump({
                "data_path": "cifar10_binary_first5_last5",
                "stratified": args.stratified,
                "n_llc_samples_actual": n_data,
                "sgld_config": {
                    "epsilon": args.epsilon, "gamma": args.gamma,
                    "num_steps": args.sgld_num_steps,
                    "batch_size": args.sgld_batch_size,
                },
                "results": results,
            }, f, indent=2)

    total_elapsed = time.time() - t_start_all
    print(f"\nDone. Total elapsed: {total_elapsed/60:.1f} min")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
