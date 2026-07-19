"""
eval_llc_moment_data.py

任意のモーメント編集データ(.npz)に対して、
保存済みチェックポイントを起点にSGLDを実行しLLCを推定する汎用スクリプト。

drLLC (data-refined LLC) の考え方に基づき、
評価対象データ q' (ここではモーメント編集データ全体)に対して
LLC推定を行う。n は q' のサンプル数を使用する。

計算コストの観点から、n を論文 (Muhamed & Smith, 2025 "The Geometry
of Forgetting") の運用 (N_docs <= 50,000) に揃えるため、
(original_label, target_label) の組み合わせごとに均等な
層別サンプリングを行い、n ≈ 50,000 に揃えるオプションを用意する。

SGLDハイパラは Lau et al. の論文値を使用 (expt_llc_curve.py と揃える):
  epsilon=2e-7, gamma=1.0, num_steps=3000, batch_size=2048

burn-inについて:
  このスクリプトでは burn-in 処理を行わず、loss_trace を全ステップ分
  そのまま保存する。LLC(lambdahat)の計算は後処理(ノートブック等)で
  任意の burn_in_ratio を指定して行うこと。これにより burn-inあり/なし
  両方を後から自由に比較できる。

使用例:
  python scripts/eval_llc_moment_data.py \
      --data moment_data/grafting/bounded_shift/shifted_cifar10.npz \
      --label_key target_label \
      --ckpt_dir ~/checkpoints/dsb_cifar10/seed42 \
      --steps 1000 10000 63096 \
      --stratified --n_total 50000 \
      --out_name bounded_shift_llc
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# src/ (Lau et al. LLC実装) へのパスも通す
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
from sgld_utils import SGLDConfig, run_sgld


def compute_loss(params, state, model, rng, x, y, num_classes, is_training=False):
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    logits, new_state = model.apply(params, state, rng, x, is_training)
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
    return loss_val, new_state


def stratified_sample(x, original_label, target_label, n_total, seed=42):
    """
    (original_label, target_label) の組み合わせ(ペア)ごとに均等にサンプリングし、
    合計 n_total 件に近いデータを抽出する。

    x:              (N, H, W, C)
    original_label: (N,)
    target_label:   (N,)
    n_total:        抽出後の目標総サンプル数
    """
    rng = np.random.RandomState(seed)

    pairs = np.stack([original_label, target_label], axis=1)
    unique_pairs, pair_inverse = np.unique(pairs, axis=0, return_inverse=True)
    n_pairs = len(unique_pairs)

    n_per_pair = max(1, n_total // n_pairs)

    selected_indices = []
    for pair_idx in range(n_pairs):
        idx_in_pair = np.where(pair_inverse == pair_idx)[0]
        n_take = min(n_per_pair, len(idx_in_pair))
        chosen = rng.choice(idx_in_pair, n_take, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    print(f"Stratified sampling: {n_pairs} pairs, "
          f"{n_per_pair} samples/pair (target), "
          f"{len(selected_indices)} total samples")

    return (
        x[selected_indices],
        original_label[selected_indices],
        target_label[selected_indices],
    )


def load_moment_data(
    data_path,
    label_key="target_label",
    stratified=False,
    n_total=50000,
    seed=42,
):
    """
    モーメント編集データを読み込む。

    label_key: LLC計算(=損失関数)に使うラベルのキー
      - grafting系: "target_label" (移植先クラスとして分類できるか)
      - deletion系: "original_label" (元クラスとして分類できるか)

    stratified: True の場合、(original_label, target_label) の
                組み合わせごとに均等抽出して n_total 件に揃える。
    """
    data = np.load(data_path)
    x = data["pixel_values"].astype(np.float32)

    if stratified:
        original_label = data["original_label"].astype(np.int32)
        target_label = data["target_label"].astype(np.int32)
        x, orig_sampled, targ_sampled = stratified_sample(
            x, original_label, target_label, n_total, seed=seed
        )
        y = targ_sampled if label_key == "target_label" else orig_sampled
    else:
        y = data[label_key].astype(np.int32)

    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--label_key", type=str, default="target_label",
                         choices=["target_label", "original_label"])
    parser.add_argument("--ckpt_dir", type=str,
                         default="~/checkpoints/dsb_cifar10/seed42")
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                         help="LLCを計算したいstep番号のリスト")
    parser.add_argument("--out_dir", type=str,
                         default="~/cuda_test/llc_results")
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # 層別サンプリング設定 (n を論文の運用規模に揃える)
    parser.add_argument("--stratified", action="store_true",
                         help="(original_label, target_label)の組み合わせごとに均等抽出する")
    parser.add_argument("--n_total", type=int, default=50000,
                         help="層別サンプリング後の総サンプル数 (Muhamed & Smith 2025 に合わせデフォルト50000)")

    # SGLDハイパラ (Lau et al. 論文値がデフォルト、expt_llc_curve.py と同一)
    parser.add_argument("--epsilon", type=float, default=2e-7)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.json"

    print(f"Loading moment-edited data from {args.data} (label_key={args.label_key})...")
    x_eval, y_eval = load_moment_data(
        args.data,
        label_key=args.label_key,
        stratified=args.stratified,
        n_total=args.n_total,
        seed=args.seed,
    )
    print(f"LLC eval data shape: {x_eval.shape}")

    n_data = len(x_eval)  # drLLC式のnはこのデータ(q')のサンプル数
    itemp = 1 / np.log(n_data)
    print(f"n = {n_data}, itemp (1/log(n)) = {itemp:.6f}")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(args.seed)

    sgld_config = SGLDConfig(
        epsilon=args.epsilon,
        gamma=args.gamma,
        num_steps=args.sgld_num_steps,
        batch_size=args.sgld_batch_size,
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

        # burn-inは後処理で行う。ここではloss_traceをそのまま保存する。
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
            "data_path": args.data,
            "label_key": args.label_key,
            "stratified": args.stratified,
            "n_total_requested": args.n_total,
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
