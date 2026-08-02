# scripts/eval_full_curve.py
"""
deletion_vision (ConvNeXt-atto) の全チェックポイントに対し、
  - train_acc, train_loss : 学習に使った編集データ (moment-edited) 上での指標
  - test_acc,  test_loss  : 実CIFAR-10テストセット上での指標
を1パスで計算し、JSONLに逐次保存する。

高速化・再開性:
  - データは手法ごとに1回だけロードしGPU上に保持
  - モデルは1回だけ構築し、state_dictだけ都度差し替え
  - JSONLに1件ずつ即書き込み + resume対応 (既存の(method,seed,step)はスキップ)

使用例:
  CUDA_VISIBLE_DEVICES=2 python3 eval_full_curve.py \
      --out /home/nakano/server/llc_results_dense/deletion_vision_full_curve.jsonl
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import torchvision.transforms as T
from datasets import load_dataset
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification


def build_convnext_atto(num_classes=10, image_size=32):
    cfg = ConvNextV2Config(
        image_size=image_size, num_labels=num_classes,
        hidden_sizes=[40, 80, 160, 320], depths=[2, 2, 6, 2],
        drop_path_rate=0.1, patch_size=1,
    )
    return ConvNextV2ForImageClassification(cfg)


def load_moment_train_data(data_root, method, device):
    dir_path = Path(data_root) / "deletion" / method
    npz_files = sorted(dir_path.glob("*.npz"))
    assert len(npz_files) == 1
    d = np.load(npz_files[0])

    images = d["pixel_values"]
    if images.dtype != np.uint8:
        if images.max() <= 1.5:
            images = (images * 255.0).clip(0, 255)
        images = images.astype(np.uint8)

    labels = d["original_label"]  # deletion系はoriginal_labelを使用

    imgs = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    imgs = imgs.to(device)
    labels = torch.from_numpy(labels).long().to(device)
    return imgs, labels


@torch.no_grad()
def evaluate_acc_and_loss(model, imgs, labels, bs=1000):
    correct, total, loss_sum = 0, 0, 0.0
    for i in range(0, imgs.shape[0], bs):
        out = model(pixel_values=imgs[i:i + bs])
        logits = out.logits
        batch_labels = labels[i:i + bs]
        preds = logits.argmax(dim=-1)
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.numel()
        loss_sum += F.cross_entropy(logits, batch_labels, reduction="sum").item()
    return correct / total, loss_sum / total


def main():
    p = ArgumentParser()
    p.add_argument("--base-dir", default="/home/nakano/server/checkpoints_dense/deletion_vision")
    p.add_argument("--data-root", default="/home/nakano/server/moment_data")
    p.add_argument("--methods", nargs="+",
                    default=["conrad", "gaussian", "ics", "truncated_normal"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--net", default="convnext-atto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["method"], r["seed"], r["step"]))
        print(f"Resuming: {len(done)} already computed, skipping them")

    # ── 実テストセット (1回だけロード, 全手法共通) ──
    real_ds = load_dataset("uoft-cs/cifar10")
    eval_ds = real_ds["test"]
    img_key = "img" if "img" in eval_ds.column_names else "image"
    label_key = "label" if "label" in eval_ds.column_names else "fine_label"
    tf = T.Compose([T.ToTensor()])
    print("Preloading real test set onto GPU...")
    test_imgs = torch.stack([tf(x.convert("RGB")) for x in eval_ds[img_key]]).to(device)
    test_labels = torch.tensor(eval_ds[label_key]).to(device)

    model = build_convnext_atto().to(device).eval()

    f_out = open(out_path, "a")
    for method in args.methods:
        print(f"Loading moment-edited train data: {method}")
        train_imgs, train_labels = load_moment_train_data(args.data_root, method, device)

        for seed in args.seeds:
            ckpt_root = Path(args.base_dir) / f"deletion_{method}" / args.net / f"seed{seed}"
            if not ckpt_root.exists():
                print(f"skip (not found): {ckpt_root}")
                continue
            steps = sorted(
                int(d.name.split("-")[1]) for d in ckpt_root.glob("checkpoint-*")
            )
            n_todo = sum(1 for s in steps if (method, seed, s) not in done)
            print(f"{method} seed{seed}: {len(steps)} checkpoints, {n_todo} to do")

            for step in steps:
                if (method, seed, step) in done:
                    continue
                ckpt_dir = ckpt_root / f"checkpoint-{step}"
                state_dict = load_file(str(ckpt_dir / "model.safetensors"), device=device)
                model.load_state_dict(state_dict)

                train_acc, train_loss = evaluate_acc_and_loss(model, train_imgs, train_labels)
                test_acc, test_loss = evaluate_acc_and_loss(model, test_imgs, test_labels)

                rec = {
                    "method": method, "seed": seed, "step": step,
                    "train_acc": train_acc, "train_loss": train_loss,
                    "test_acc": test_acc, "test_loss": test_loss,
                }
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()

            print(f"  done: {method} seed{seed}")

        del train_imgs, train_labels
        torch.cuda.empty_cache()

    f_out.close()
    print("All evaluations complete.")


if __name__ == "__main__":
    main()