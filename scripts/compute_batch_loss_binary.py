"""
compute_batch_loss_binary.py

各チェックポイントについて、Nakkiranの訓練時と同じ1ミニバッチ(128枚)
だけを使って損失を計算する、軽量な診断スクリプト。

train_and_save_binary.py が記録している train_loss は「500ステップ
区間平均」であり、ステップごとの生の値ではない。また
compute_llc_binary.py の init_loss は「50000枚全データでの評価」で
あり、これもNakkiranが訓練中に実際に見ていた値とは異なる。

本スクリプトは、各チェックポイントの重みを読み込み、固定の1バッチ
(訓練データの先頭128枚、または --batch_size で指定した枚数)に
対する損失だけを計算する。SGLDのような反復計算は行わないため、
チェックポイント読み込み+1回の順伝播のみで、全チェックポイントを
通しても数分で完了する。

使用例:
  cd ~/cuda_test/scripts
  XLA_FLAGS="--xla_gpu_autotune_level=0" CUDA_VISIBLE_DEVICES=0 python compute_batch_loss_binary.py \
      --ckpt_dir ~/cuda_test/checkpoints/nakkiran_binary_resnet18/seed42 \
      --steps $(seq 0 200 20000) \
      --out_dir ~/cuda_test/llc_results \
      --out_name nakkiran_binary_batch_loss
"""

import sys
from pathlib import Path

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


def compute_loss(params, state, model, rng, x, y, num_classes, is_training=False):
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    logits, new_state = model.apply(params, state, rng, x, is_training)
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
    return loss_val, new_state, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/llc_results")
    parser.add_argument("--out_name", type=str, default="nakkiran_binary_batch_loss")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128,
                         help="Nakkiranの訓練時のbatch_sizeと同じ128がデフォルト")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.json"

    print("Loading CIFAR-10 (binary: First5 vs Last5)...")
    x_train, y_train, x_test, y_test = load_cifar10_binary()

    # 固定の1バッチ (先頭 batch_size 枚) を全チェックポイント共通で使う。
    # 訓練時のシャッフル順序は再現しないが、
    # 「同じ入力に対して、パラメータがどれだけ損失を変動させるか」を
    # 見たいだけなので、固定バッチで問題ない。
    rng0 = np.random.default_rng(args.seed)
    perm = rng0.permutation(len(x_train))[:args.batch_size]
    x_batch_raw = x_train[perm]
    y_batch = y_train[perm]
    x_batch = per_image_standardize(np.array(x_batch_raw))
    print(f"Fixed batch shape: {x_batch.shape} "
          f"(label distribution: {np.bincount(y_batch)})")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(args.seed)

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
        batch_loss, _, logits = compute_loss(
            params, state, model, subkey,
            jnp.array(x_batch), jnp.array(y_batch),
            args.num_classes, is_training=False,
        )
        batch_loss = float(batch_loss)
        predictions = jnp.argmax(logits, axis=-1)
        batch_acc = float(jnp.mean(predictions == y_batch))

        elapsed = time.time() - t0
        results.append({"step": step, "batch_loss": batch_loss, "batch_acc": batch_acc})
        print(f"[{i+1}/{len(args.steps)}] step={step:6d} "
              f"| batch_loss={batch_loss:.4f} | batch_acc={batch_acc:.4f} "
              f"| elapsed={elapsed:.2f}s")

        with open(out_path, "w") as f:
            json.dump({
                "data_path": "cifar10_binary_first5_last5",
                "fixed_batch_size": args.batch_size,
                "fixed_batch_seed": args.seed,
                "results": results,
            }, f, indent=2)

    total_elapsed = time.time() - t_start_all
    print(f"\nDone. Total elapsed: {total_elapsed/60:.1f} min")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
