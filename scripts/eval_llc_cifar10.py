"""
eval_llc_cifar10.py

保存済みチェックポイントに対して、通常のCIFAR10訓練データで
LLC(SGLD)を計算する。モーメント編集データとの比較基準として使う。

eval_llc_moment_data.py と同じ設計・同じSGLDハイパラを使用する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LLC_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "github.com" / "shiryu-nakano" / "llc_reproduction" / "src"
sys.path.insert(0, str(LLC_SRC))

import argparse
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from train_and_save import make_resnet18
from cifar10_loader import load_cifar10  # jax非依存版
from sgld_utils import SGLDConfig, run_sgld


def compute_loss(params, state, model, rng, x, y, num_classes, is_training=False):
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    logits, new_state = model.apply(params, state, rng, x, is_training)
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
    return loss_val, new_state


def stratified_sample_by_class(x, y, n_total, seed=42):
    """クラスごとに均等抽出してn_totalに揃える"""
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
    parser.add_argument("--ckpt_dir", type=str,
                         default="~/checkpoints/dsb_cifar10/seed42")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/llc_results")
    parser.add_argument("--out_name", type=str, default="cifar10_train_llc_full")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--n_total", type=int, default=50000)
    parser.add_argument("--epsilon", type=float, default=2e-7)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.json"

    print("Loading CIFAR-10 (train split)...")
    x_train, y_train, x_test, y_test = load_cifar10()

    if args.stratified:
        x_eval, y_eval = stratified_sample_by_class(
            x_train, y_train, args.n_total, seed=args.seed)
    else:
        x_eval, y_eval = x_train, y_train

    print(f"LLC eval data shape: {x_eval.shape}")

    n_data = len(x_eval)
    itemp = 1 / np.log(n_data)
    print(f"n = {n_data}, itemp (1/log(n)) = {itemp:.6f}")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(args.seed)

    sgld_config = SGLDConfig(
        epsilon=args.epsilon, gamma=args.gamma,
        num_steps=args.sgld_num_steps, batch_size=args.sgld_batch_size,
    )

    results = []
    for step in args.steps:
        ckpt_path = Path(args.ckpt_dir).expanduser() / f"step_{step:07d}"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for step={step} not found, skipping.")
            continue

        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        rng, subkey = jax.random.split(rng)
        loss_fn = lambda p, x, y: compute_loss(
            p, state, model, rng, x, y, args.num_classes, is_training=False)[0]

        loss_trace, distances, acceptance_probs = run_sgld(
            subkey, loss_fn, sgld_config, params,
            jnp.array(x_eval), jnp.array(y_eval),
            itemp=itemp, trace_batch_loss=True,
            compute_distance=False, compute_mala_acceptance=False,
            verbose=False,
        )

        init_loss = float(loss_fn(params, jnp.array(x_eval), jnp.array(y_eval)))
        loss_trace_list = [float(l) for l in loss_trace]
        mean_full = float(np.mean(loss_trace_list))
        mean_last10pct = float(np.mean(loss_trace_list[int(len(loss_trace_list) * 0.9):]))

        results.append({
            "step": step,
            "init_loss": init_loss,
            "loss_trace": loss_trace_list,
        })
        print(
            f"step={step:6d} | init_loss={init_loss:.4f} "
            f"| mean(full)={mean_full:.4f} | mean(last10%)={mean_last10pct:.4f}"
        )

    with open(out_path, "w") as f:
        json.dump({
            "data_path": "cifar10_train",
            "stratified": args.stratified,
            "n_llc_samples_actual": n_data,
            "sgld_config": {
                "epsilon": args.epsilon, "gamma": args.gamma,
                "num_steps": args.sgld_num_steps, "batch_size": args.sgld_batch_size,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
