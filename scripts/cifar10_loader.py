"""
cifar10_loader.py

CIFAR10データセットの読み込みだけを行う、JAXに依存しない独立モジュール。

train_and_save.py の load_cifar10 と同一のロジックだが、
tensorflow_datasetsのみに依存し、jax/haikuをimportしない。
これにより、PyTorch(concept_erasure等)と同じプロセス内で
CUDAコンテキストが競合する問題を回避できる。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # TensorFlowにはGPUを使わせない

import numpy as np
import tensorflow_datasets as tfds


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
