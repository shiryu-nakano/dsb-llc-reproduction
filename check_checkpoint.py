"""
チェックポイントの読み込み・推論の最小限確認スクリプト
"""
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from pathlib import Path

# train_and_save.py からモデル定義を再利用
from train_and_save import make_resnet18, load_cifar10

CKPT_DIR = Path("~/checkpoints/dsb_cifar10/seed42").expanduser()

# ── 1. 読み出し確認 ──
step = 10000  # 適当な中間チェックポイントで確認
ckpt_path = (CKPT_DIR / f"step_{step:07d}").resolve()
print(f"Loading checkpoint from: {ckpt_path}")

checkpointer = ocp.PyTreeCheckpointer()
restored = checkpointer.restore(str(ckpt_path))

print("Keys in restored:", restored.keys())
print("Restored step:", restored["step"])

params = restored["params"]
state = restored["state"]

# パラメータ数を確認 (11,181,642 と一致するはず)
import jax.tree_util as jtree
import numpy as np
param_count = sum(np.prod(p.shape) for p in jtree.tree_leaves(params))
print(f"Loaded param count: {param_count:,}")

# ── 2. 推論確認 ──
model, _, _ = None, None, None
model = make_resnet18(num_classes=10, k=64)

x_train, y_train, x_test, y_test = load_cifar10()

# 小さいバッチで推論テスト
x_sample = jnp.array(x_test[:16])
y_sample = y_test[:16]

rng = jax.random.PRNGKey(0)
logits, _ = model.apply(params, state, rng, x_sample, False) # ここでパラメータと状態を使って推論する
preds = jnp.argmax(logits, axis=-1)

acc = jnp.mean(preds == y_sample)
print(f"Sample inference (16 images): accuracy = {acc:.4f}")
print(f"Predictions: {preds}")
print(f"True labels: {y_sample}")
