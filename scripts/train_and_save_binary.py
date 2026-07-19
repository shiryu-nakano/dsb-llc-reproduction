"""
train_and_save_binary.py

Nakkiran et al. (2019) "SGD on Neural Networks Learns Functions of
Increasing Complexity" の CIFAR "First 5 vs Last 5" 設定で ResNet18 を
学習し、100 ステップおきにチェックポイントを保存するスクリプト。

train_and_save.py (Refinetti et al. 設定) からの変更点:
  A. 必須 (2値分類実験として成立させるための変更)
     - ラベルを 2 値化 (class 0-4 -> 0, class 5-9 -> 1)
     - num_classes のデフォルトを 2 に
     - 学習ハイパラを Nakkiran 設定に変更
       (lr=0.01 固定・schedule無し, momentum=0.9, weight decay 無し)
     - save_steps を 0..20000 の 100 刻み線形間隔に変更

  B. 再現度向上 (Nakkiran のデータパイプラインに合わせる)
     - padding を reflect -> zero (constant) に変更
       (tf.image.resize_image_with_crop_or_pad 相当)
     - per-image standardization を追加
       (tf.image.per_image_standardization 相当。train/test 両方に適用)

  C. アーキテクチャ (pre-activation 化, stem の変更) は見送り。
     モデル定義は Lau et al. (expt_llc_curve.py) と同一のまま
     (resnet_v2=False, 7x7 stride2 initial conv + max_pool)。
     これは LLC 計算パイプラインとの互換性を優先した意図的な判断。

使用例:
  CUDA_VISIBLE_DEVICES=1 python train_and_save_binary.py with \
      expt_name=nakkiran_binary_resnet18 seed=42 \
      -F ../results/nakkiran_binary_training
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
import haiku as hk
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np
import optax
import orbax.checkpoint as ocp

from typing import NamedTuple, Optional, Sequence
import gc
import os
from pathlib import Path
from sacred import Experiment

ex = Experiment('train_and_save_binary')


# ─────────────────────────────────────────────
# モデル定義
# Lau et al. (expt_llc_curve.py) と完全同一の定義 (7x7 stride2)
# ★ ここは変更しない (C は見送り。LLC 実験のパラメータ構造を維持するため) ★
# ─────────────────────────────────────────────
class CustomResNet18(hk.nets.ResNet):
    """ResNet18 (Lau et al. / expt_llc_curve.py と同一定義)."""

    def __init__(
        self,
        num_classes: int,
        k: int = 64,
        name: Optional[str] = None,
        strides: Sequence[int] = (1, 2, 2, 2),
    ):
        custom_configs = {
            "blocks_per_group": (2, 2, 2, 2),
            "bottleneck": False,
            "channels_per_group": (k, 2 * k, 4 * k, 8 * k),
            "use_projection": (False, True, True, True),
        }
        super().__init__(
            num_classes=num_classes,
            bn_config=None,
            initial_conv_config={
                "output_channels": k,
                "kernel_shape": 7,
                "stride": 2,
                "padding": "SAME",
            },
            resnet_v2=False,
            strides=strides,
            logits_config=None,
            name=name,
            **custom_configs,
        )


def make_resnet18(num_classes=2, k=64):
    def net_fn(x, is_training=True):
        model = CustomResNet18(num_classes=num_classes, k=k)
        return model(x, is_training)
    return hk.transform_with_state(net_fn)


# ─────────────────────────────────────────────
# データ読み込み + ラベル2値化 [A]
# ─────────────────────────────────────────────
def relabel_first5_last5(labels):
    """
    Nakkiran et al. "CIFAR-10 First 5 vs Last 5":
    class 0-4 (airplane, automobile, bird, cat, deer)   -> label 0
    class 5-9 (dog, frog, horse, ship, truck)            -> label 1
    """
    return (labels >= 5).astype(labels.dtype)


def load_cifar10_binary():
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(
        split='train', batch_size=-1, shuffle_files=False))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(
        split='test', batch_size=-1))

    train_images = train_ds['image'].astype(np.float32) / 255.0
    train_labels = relabel_first5_last5(train_ds['label'])
    test_images  = test_ds['image'].astype(np.float32) / 255.0
    test_labels  = relabel_first5_last5(test_ds['label'])

    return train_images, train_labels, test_images, test_labels


# ─────────────────────────────────────────────
# データ拡張 + 正規化 [B]
# Nakkiran (ml_theory_mi.ipynb) の transform() に合わせる:
#   train: zero-pad -> random crop -> random flip -> per-image standardize
#   eval:  per-image standardize のみ
# ─────────────────────────────────────────────
def per_image_standardize(images):
    """tf.image.per_image_standardization 相当。
    images: (N, H, W, C), 想定レンジは [0, 1]
    """
    axes = (1, 2, 3)
    mean = images.mean(axis=axes, keepdims=True)
    std = images.std(axis=axes, keepdims=True)
    num_pixels = images.shape[1] * images.shape[2] * images.shape[3]
    adjusted_std = np.maximum(std, 1.0 / np.sqrt(num_pixels))
    return (images - mean) / adjusted_std


def augment_batch(np_rng, images, training=True):
    """
    Nakkiran のデータ拡張:
    - zero padding (4px) -> random crop -> random horizontal flip
    - per-image standardization (train/eval 共通)

    NOTE: クロップ位置・フリップ判定は勾配計算に関与しない(微分不要)ため、
    JAX の RNG ではなく numpy の RNG (np.random.Generator) を使う。
    JAX RNG をバッチ内 for ループで呼ぶと、1サンプルごとに device<->host の
    同期が発生し、致命的に遅くなるため。
    """
    batch_size, H, W, C = images.shape

    if training:
        pad = 4
        padded = np.pad(
            images, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
            mode='constant', constant_values=0.0,
        )
        # クロップ位置・フリップ判定をバッチ分まとめて一括生成
        tops  = np_rng.integers(0, 2 * pad + 1, size=batch_size)
        lefts = np_rng.integers(0, 2 * pad + 1, size=batch_size)
        flips = np_rng.random(batch_size) > 0.5

        augmented = np.empty_like(images)
        for i in range(batch_size):
            crop = padded[i, tops[i]:tops[i] + H, lefts[i]:lefts[i] + W, :]
            augmented[i] = crop[:, ::-1, :] if flips[i] else crop
        images = augmented

    images = per_image_standardize(images)
    return images


def batch_generator(x, y, batch_size, seed, augment=True, drop_remainder=True):
    """
    numpy の RNG でシャッフル・拡張を行うジェネレータ。
    drop_remainder=True の場合、バッチサイズに満たない最後の半端バッチは
    捨てる (BatchNorm の統計を安定させ、jax.jit の再コンパイルを避けるため)。
    """
    np_rng = np.random.default_rng(seed)
    num_examples = len(x)
    while True:
        perm = np_rng.permutation(num_examples)
        num_batches = (
            num_examples // batch_size if drop_remainder
            else -(-num_examples // batch_size)  # ceil division
        )
        for b in range(num_batches):
            batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]
            x_batch = augment_batch(np_rng, x_batch, training=augment)
            yield x_batch, y_batch


def initialize_model(rng, num_classes=2, k=64):
    model = make_resnet18(num_classes=num_classes, k=k)
    dummy_input = jnp.ones([1, 32, 32, 3], jnp.float32)
    params, state = model.init(rng, dummy_input, True)
    return model, params, state


def evaluate_accuracy(model, params, state, x, y, rngkey, batch_size=1000):
    """全データを標準化した上でバッチ評価 (メモリ節約のためミニバッチで回す)。"""
    correct = 0
    n = len(x)
    for i in range(0, n, batch_size):
        x_batch = np.array(x[i:i + batch_size])
        y_batch = y[i:i + batch_size]
        x_batch = per_image_standardize(x_batch)
        logits, _ = model.apply(params, state, rngkey, x_batch, False)
        predictions = jnp.argmax(logits, axis=-1)
        correct += int(jnp.sum(predictions == y_batch))
    return correct / n


# ─────────────────────────────────────────────
# Sacred config
# ─────────────────────────────────────────────
@ex.config
def cfg():
    expt_name     = None
    seed          = 42
    num_classes   = 2            # [A] 10 -> 2
    k             = 64           # ResNet幅パラメータ (Lau et al. と同一)

    # --- 学習設定 (Nakkiran et al., Section 4 / CIFAR First5vsLast5) --- [A]
    learning_rate = 0.01         # 固定 (schedule 無し)
    momentum      = 0.9
    weight_decay  = None         # Nakkiran は weight decay を使わない
    batch_size    = 128
    total_steps   = 20000        # epoch ではなく total step 数で管理

    # --- チェックポイント設定 --- [A]
    # Nakkiran Figure 5 の評価間隔 (train_simple 実装内 save_frequency=100) に合わせる
    save_every    = 100
    checkpoint_dir = "../checkpoints"

    # --- ロギング ---
    verbose        = True
    log_every_steps = 500        # 何stepごとにloss/accを表示するか


@ex.automain
def run_experiment(
    _run,
    expt_name,
    seed,
    num_classes,
    k,
    learning_rate,
    momentum,
    weight_decay,
    batch_size,
    total_steps,
    save_every,
    checkpoint_dir,
    verbose,
    log_every_steps,
):
    # ── シード設定 ──
    np.random.seed(seed)
    rngkey = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)

    # ── データ読み込み ── [A] 2値ラベル化
    x_train, y_train, x_test, y_test = load_cifar10_binary()
    num_training_data = len(x_train)
    print(f"num_training_data: {num_training_data}, total_steps: {total_steps}")
    print(f"train label distribution: {np.bincount(y_train)}")
    print(f"test  label distribution: {np.bincount(y_test)}")

    # ── チェックポイント保存stepの確定 ── [A] 線形間隔
    save_steps_set = set(range(0, total_steps + 1, save_every))
    print(f"Checkpoint steps ({len(save_steps_set)} points): "
          f"0..{total_steps} step {save_every}")

    # ── モデル初期化 ──
    rngkey, init_key = jax.random.split(rngkey)
    model, params, state = initialize_model(init_key, num_classes=num_classes, k=k)
    param_count = sum(np.prod(p.shape) for p in jtree.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")

    # ── オプティマイザ設定 ── [A] 固定lr, weight decay無し
    if weight_decay is not None and weight_decay > 0.0:
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.sgd(learning_rate=learning_rate, momentum=momentum),
        )
    else:
        optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    opt_state = optimizer.init(params)

    # ── 損失関数 ──
    def compute_loss(params, state, rng, x, y, is_training):
        labels_one_hot = jax.nn.one_hot(y, num_classes)
        logits, new_state = model.apply(params, state, rng, x, is_training)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, labels=labels_one_hot))
        return loss, new_state

    @jax.jit
    def update_step(params, state, rng, x, y, opt_state):
        (loss, new_state), grad = jax.value_and_grad(
            compute_loss, has_aux=True)(params, state, rng, x, y, True)
        updates, new_opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state, new_opt_state

    # ── チェックポイント設定 ──
    ckpt_base = (Path(checkpoint_dir) / (expt_name or "unnamed") / f"seed{seed}").resolve()
    ckpt_base.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(params, state, step):
        ckpt_path = (ckpt_base / f"step_{step:07d}").resolve()
        ckpt_path.mkdir(parents=True, exist_ok=True)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            str(ckpt_path),
            {"params": params, "state": state, "step": step},
            force=True,
        )
        if verbose:
            print(f"  [Checkpoint saved] step={step}, path={ckpt_path}")

    # ── データイテレータ ── [B] augment=True で zero-pad crop/flip + standardize
    # NOTE: シャッフル・拡張は numpy RNG で行う (JAX RNG は使わない、上記参照)
    train_iter = batch_generator(
        x_train, y_train, batch_size, seed=seed, augment=True, drop_remainder=True)

    # ── 学習ループ ──
    _run.info = []
    global_step = 0

    if global_step in save_steps_set:
        save_checkpoint(params, state, step=global_step)

    # ── 学習開始前 (t=0) のベースライン精度も記録 ──
    rngkey, eval_key = jax.random.split(rngkey)
    init_test_acc  = evaluate_accuracy(model, params, state, x_test,  y_test,  eval_key)
    init_train_acc = evaluate_accuracy(model, params, state, x_train, y_train, eval_key)
    _run.info.append({
        "step": 0, "train_loss": None,
        "test_acc": init_test_acc, "train_acc": init_train_acc,
    })
    if verbose:
        print(f"step=     0/{total_steps} | (init) "
              f"| train_acc={init_train_acc:.4f} | test_acc={init_test_acc:.4f}")

    running_losses = []
    while global_step < total_steps:
        x_batch, y_batch = next(train_iter)
        rngkey, step_key = jax.random.split(rngkey)
        loss, params, state, opt_state = update_step(
            params, state, step_key,
            jnp.array(x_batch), jnp.array(y_batch),
            opt_state,
        )
        running_losses.append(float(loss))
        global_step += 1

        # ── 指定step でのチェックポイント保存 ── [A]
        if global_step in save_steps_set:
            save_checkpoint(params, state, step=global_step)

        # ── ロギング ──
        if global_step % log_every_steps == 0 or global_step == total_steps:
            rngkey, eval_key = jax.random.split(rngkey)
            test_acc  = evaluate_accuracy(model, params, state, x_test,  y_test,  eval_key)
            train_acc = evaluate_accuracy(model, params, state, x_train, y_train, eval_key)
            train_loss_avg = float(np.mean(running_losses))
            running_losses = []

            rec = {
                "step":       global_step,
                "train_loss": train_loss_avg,
                "test_acc":   test_acc,
                "train_acc":  train_acc,
            }
            _run.info.append(rec)

            if verbose:
                print(
                    f"step={global_step:6d}/{total_steps} "
                    f"| train_loss={train_loss_avg:.4f} "
                    f"| train_acc={train_acc:.4f} "
                    f"| test_acc={test_acc:.4f}"
                )

            gc.collect()

    print("Training complete.")
    print(f"Checkpoints saved to: {ckpt_base}")
    print(f"Total checkpoints saved: {len(save_steps_set)}")
    return
