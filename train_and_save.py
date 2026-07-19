"""
train_and_save.py

Refinetti et al. (2023) の学習設定でResNet18をCIFAR-10で学習し，
チェックポイントを指定したstepで保存するスクリプト．

モデル定義は Lau et al. (expt_llc_curve.py) と完全に同一 (7x7 stride2)．
Refinetti et al. のCIFAR用初層 (3x3 stride1) はコメントで併記しており，
切り替え可能．

学習設定 (Refinetti et al.):
  optimizer:    SGD
  momentum:     0.9
  weight_decay: 5e-4
  batch_size:   128
  epochs:       200
  lr:           0.005
  lr_schedule:  cosine annealing
  augmentation: random crop (padding=4) + horizontal flip

チェックポイント保存:
  save_steps に手動でstep番号のリストを渡す．
  (論文Fig 4のx軸が対数スケールのため，対数間隔のリストを渡すことを想定)

使用例:
  CUDA_VISIBLE_DEVICES=1 python train_and_save.py with \
      expt_name=dsb_cifar10 seed=42 \
      'save_steps=[0,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025]' \
      -F ../results/dsb_training
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

ex = Experiment('train_and_save')


# ─────────────────────────────────────────────
# モデル定義
# Lau et al. (expt_llc_curve.py) と完全同一の定義 (7x7 stride2)
# ★ ここは絶対に変更しない ★ (LLC実験のSGLDハイパラを流用するため)
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
            # ── Lau et al. (現行LLC実験と同一): 7x7 stride2 ──
            initial_conv_config={
                "output_channels": k,
                "kernel_shape": 7,
                "stride": 2,
                "padding": "SAME",
            },
            # ── Refinetti et al. のCIFAR用設定に切り替える場合は
            #    上をコメントアウトして以下を使用 (strides も変更すること):
            # initial_conv_config={
            #     "output_channels": k,
            #     "kernel_shape": 3,
            #     "stride": 1,
            #     "padding": "SAME",
            # },
            resnet_v2=False,
            # ── Lau et al.: (1, 2, 2, 2) ──
            strides=strides,
            # ── Refinetti et al. の場合: strides=(1, 1, 2, 2) ──
            logits_config=None,
            name=name,
            **custom_configs,
        )


def make_resnet18(num_classes=10, k=64):
    def net_fn(x, is_training=True):
        model = CustomResNet18(num_classes=num_classes, k=k)
        return model(x, is_training)
    return hk.transform_with_state(net_fn)


# ─────────────────────────────────────────────
# データ読み込み + Augmentation
# ─────────────────────────────────────────────
def load_cifar10():
    ds_builder = tfds.builder('cifar10')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(
        split='train', batch_size=-1, shuffle_files=False))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(
        split='test', batch_size=-1))

    train_images = train_ds['image'].astype(np.float32) / 255.0
    train_labels = train_ds['label']
    test_images  = test_ds['image'].astype(np.float32) / 255.0
    test_labels  = test_ds['label']

    return train_images, train_labels, test_images, test_labels


def augment_batch(rng, images):
    """
    Refinetti et al.のaugmentation:
    - random crop (padding=4)
    - random horizontal flip
    """
    batch_size, H, W, C = images.shape
    pad = 4

    padded = np.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')

    augmented = np.zeros_like(images)
    for i in range(batch_size):
        rng, key = jax.random.split(rng)
        top  = int(jax.random.randint(key, (), 0, 2 * pad))
        left = int(jax.random.randint(key, (), 0, 2 * pad))
        augmented[i] = padded[i, top:top+H, left:left+W, :]

        rng, key = jax.random.split(rng)
        if jax.random.uniform(key) > 0.5:
            augmented[i] = augmented[i, :, ::-1, :]

    return augmented, rng


def batch_generator(x, y, batch_size, rngkey, augment=True):
    num_examples = len(x)
    while True:
        rngkey, perm_key = jax.random.split(rngkey)
        perm = jax.random.permutation(perm_key, jnp.arange(num_examples))
        for i in range(0, num_examples, batch_size):
            batch_idx = perm[i:i + batch_size]
            x_batch = np.array(x[batch_idx])
            y_batch = np.array(y[batch_idx])
            if augment:
                x_batch, rngkey = augment_batch(rngkey, x_batch)
            yield x_batch, y_batch


def initialize_model(rng, num_classes=10, k=64):
    model = make_resnet18(num_classes=num_classes, k=k)
    dummy_input = jnp.ones([1, 32, 32, 3], jnp.float32)
    params, state = model.init(rng, dummy_input, True)
    return model, params, state


def evaluate_accuracy(model, params, state, x, y, rngkey):
    logits, _ = model.apply(params, state, rngkey, x, False)
    predictions = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(predictions == y))


# ─────────────────────────────────────────────
# Sacred config
# ─────────────────────────────────────────────
@ex.config
def cfg():
    expt_name     = None
    seed          = 42
    num_classes   = 10
    k             = 64          # ResNet幅パラメータ
    # --- 学習設定 (Refinetti et al.) ---
    learning_rate = 0.005
    momentum      = 0.9
    weight_decay  = 5e-4
    batch_size    = 128
    num_epochs    = 200
    # --- チェックポイント設定 ---
    # 保存したいstep番号を手動で指定する (例: 対数間隔のリストなど)
    save_steps = [
        0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
        610, 987, 1597, 2584, 4181, 6765, 10946, 17711,
        28657, 46368, 75025,
    ]
    checkpoint_dir = "../checkpoints"
    # --- ロギング ---
    verbose           = True
    log_every_epochs  = 1        # 何epochごとにloss/accを表示するか


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
    num_epochs,
    save_steps,
    checkpoint_dir,
    verbose,
    log_every_epochs,
):
    # ── シード設定 ──
    np.random.seed(seed)
    rngkey = jax.random.PRNGKey(seed)
    tf.random.set_seed(seed)

    # ── データ読み込み ──
    x_train, y_train, x_test, y_test = load_cifar10()
    num_training_data = len(x_train)
    steps_per_epoch   = num_training_data // batch_size
    total_steps       = num_epochs * steps_per_epoch
    print(f"steps_per_epoch: {steps_per_epoch}, total_steps: {total_steps}")

    # ── チェックポイント保存stepの確認 ──
    save_steps_set = set(save_steps)
    over_limit = sorted(s for s in save_steps_set if s > total_steps)
    if over_limit:
        print(f"WARNING: total_steps({total_steps})を超えるsave_stepsは無視されます: {over_limit}")
    save_steps_set = {s for s in save_steps_set if s <= total_steps}
    print(f"Checkpoint steps ({len(save_steps_set)} points): {sorted(save_steps_set)}")

    # ── モデル初期化 ──
    rngkey, init_key = jax.random.split(rngkey)
    model, params, state = initialize_model(init_key, num_classes=num_classes, k=k)
    param_count = sum(np.prod(p.shape) for p in jtree.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")

    # ── オプティマイザ設定 ──
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
    )
    optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.sgd(learning_rate=lr_schedule, momentum=momentum),
    )
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

    # ── データイテレータ ──
    rngkey, data_key = jax.random.split(rngkey)
    train_iter = batch_generator(
        x_train, y_train, batch_size, data_key, augment=True)

    # ── 学習ループ ──
    _run.info = []
    global_step = 0

    # step=0 がsave_stepsに含まれていれば初期状態を保存
    if global_step in save_steps_set:
        save_checkpoint(params, state, step=global_step)

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            x_batch, y_batch = next(train_iter)
            rngkey, step_key = jax.random.split(rngkey)
            loss, params, state, opt_state = update_step(
                params, state, step_key,
                jnp.array(x_batch), jnp.array(y_batch),
                opt_state,
            )
            epoch_losses.append(float(loss))
            global_step += 1

            # ── 指定step でのチェックポイント保存 ──
            if global_step in save_steps_set:
                save_checkpoint(params, state, step=global_step)

        # ── epochごとのロギング ──
        if epoch % log_every_epochs == 0 or epoch == num_epochs:
            rngkey, eval_key = jax.random.split(rngkey)
            test_acc  = evaluate_accuracy(model, params, state, x_test,  y_test,  eval_key)
            train_acc = evaluate_accuracy(model, params, state, x_train, y_train, eval_key)
            train_loss_avg = float(np.mean(epoch_losses))

            rec = {
                "epoch":      epoch,
                "step":       global_step,
                "train_loss": train_loss_avg,
                "test_acc":   test_acc,
                "train_acc":  train_acc,
            }
            _run.info.append(rec)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{num_epochs} "
                    f"| step={global_step:6d} "
                    f"| train_loss={train_loss_avg:.4f} "
                    f"| train_acc={train_acc:.4f} "
                    f"| test_acc={test_acc:.4f}"
                )

        gc.collect()

    print("Training complete.")
    print(f"Checkpoints saved to: {ckpt_base}")
    print(f"Total checkpoints saved: {len(save_steps_set)}")
    return
