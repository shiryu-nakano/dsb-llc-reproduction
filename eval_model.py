"""
eval_dsb.py

保存済みチェックポイントを全て読み込み，
train / test 両方のデータで loss / accuracy を計算してJSON保存する．
"""
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from pathlib import Path
import json
import re
import optax

from train_and_save import make_resnet18, load_cifar10


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


def main():
    CKPT_DIR = Path("~/checkpoints/dsb_cifar10/seed42").expanduser()
    OUT_PATH = Path("~/cuda_test/dsb_eval_results.json").expanduser()

    ckpt_dirs = sorted(
        CKPT_DIR.glob("step_*"),
        key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1))
    )
    print(f"Found {len(ckpt_dirs)} checkpoints")

    model = make_resnet18(num_classes=10, k=64)
    x_train, y_train, x_test, y_test = load_cifar10()

    checkpointer = ocp.PyTreeCheckpointer()
    rng = jax.random.PRNGKey(0)

    results = []
    for ckpt_path in ckpt_dirs:
        step = int(re.search(r"step_(\d+)", ckpt_path.name).group(1))
        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        test_loss, test_acc = compute_loss_and_acc(
            model, params, state, x_test, y_test, rng)
        train_loss, train_acc = compute_loss_and_acc(
            model, params, state, x_train, y_train, rng)

        results.append({
            "step": step,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })
        print(
            f"step={step:6d} "
            f"| train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
            f"| test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
        )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()