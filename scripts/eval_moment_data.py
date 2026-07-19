"""
eval_moment_data.py

任意のモーメント編集データ(.npz)に対して、
保存済みチェックポイント群でのloss/accuracyを計算する汎用スクリプト。

データフォーマット規約:
  .npz は以下のキーを持つこと
    - pixel_values:    (N, H, W, C) float32, [0,1]
    - original_label:  (N,) int (元のCIFAR10クラス)
    - target_label:    (N,) int (評価に使うラベル。grafting系はtarget、
                                   deletion系はoriginalと同じ値を入れる)

使用例:
  python scripts/eval_moment_data.py \
      --data moment_data/grafting/bounded_shift/shifted_cifar10.npz \
      --ckpt_dir ~/checkpoints/dsb_cifar10/seed42 \
      --out_name bounded_shift
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from train_and_save import make_resnet18


def compute_loss_and_acc(model, params, state, x, y, rng, num_classes=10, batch_size=1000):
    """全データをバッチに分けて評価 (メモリ節約)"""
    n = len(x)
    total_loss = 0.0
    total_correct = 0
    for i in range(0, n, batch_size):
        x_batch = jnp.array(x[i:i+batch_size])
        y_batch = y[i:i+batch_size]
        logits, _ = model.apply(params, state, rng, x_batch, False)
        labels_one_hot = jax.nn.one_hot(y_batch, num_classes)
        loss = jnp.sum(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
        preds = jnp.argmax(logits, axis=-1)
        correct = jnp.sum(preds == y_batch)
        total_loss += float(loss)
        total_correct += int(correct)
    return total_loss / n, total_correct / n


def load_moment_data(data_path, label_key="target_label"):
    """
    モーメント編集データを読み込む。
    label_key: 評価に使うラベルのキー
      - grafting系: "target_label" (移植先クラスとして分類できるか)
      - deletion系: "original_label" (元クラスとして分類できるか)
    """
    data = np.load(data_path)
    x = data["pixel_values"].astype(np.float32)
    y = data[label_key].astype(np.int32)
    return x, y


def find_checkpoints(ckpt_dir):
    """チェックポイントディレクトリをstep順にソートして返す"""
    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dirs = sorted(
        ckpt_dir.glob("step_*"),
        key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1))
    )
    return ckpt_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                         help="モーメント編集データの.npzパス")
    parser.add_argument("--label_key", type=str, default="target_label",
                         choices=["target_label", "original_label"],
                         help="評価に使うラベルのキー")
    parser.add_argument("--ckpt_dir", type=str,
                         default="~/checkpoints/dsb_cifar10/seed42")
    parser.add_argument("--out_dir", type=str,
                         default="~/cuda_test/eval_results")
    parser.add_argument("--out_name", type=str, required=True,
                         help="出力ファイル名(拡張子なし)")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="評価に使う最大サンプル数(指定なしなら全部)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.json"

    print(f"Loading moment-edited data from {args.data} (label_key={args.label_key})...")
    x_eval, y_eval = load_moment_data(args.data, label_key=args.label_key)
    if args.max_samples is not None and len(x_eval) > args.max_samples:
        # 均等にサブサンプリング(先頭からmax_samples件)
        x_eval = x_eval[:args.max_samples]
        y_eval = y_eval[:args.max_samples]
    print(f"Eval data shape: {x_eval.shape}, labels shape: {y_eval.shape}")

    ckpt_dirs = find_checkpoints(args.ckpt_dir)
    print(f"Found {len(ckpt_dirs)} checkpoints")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(0)

    results = []
    for ckpt_path in ckpt_dirs:
        step = int(re.search(r"step_(\d+)", ckpt_path.name).group(1))
        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        loss, acc = compute_loss_and_acc(
            model, params, state, x_eval, y_eval, rng,
            num_classes=args.num_classes, batch_size=args.batch_size)

        results.append({"step": step, "loss": loss, "accuracy": acc})
        print(f"step={step:6d} | loss={loss:.4f} | accuracy={acc:.4f}")

    with open(out_path, "w") as f:
        json.dump({
            "data_path": args.data,
            "label_key": args.label_key,
            "n_eval_samples": int(len(x_eval)),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()