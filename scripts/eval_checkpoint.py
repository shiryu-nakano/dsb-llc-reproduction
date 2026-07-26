# scripts/eval_checkpoint.py
"""
保存済みチェックポイントを読み込み、実CIFAR-10テストセットでの精度を評価する。
train_vision_moment.py のログが失われていても、ここから直接検証できる。

使用例:
  python3 eval_checkpoint.py \
      --ckpt-dir /home/nakano/server/checkpoints_dense/deletion_vision/deletion_conrad/convnext-atto/seed42/checkpoint-100000 \
      --net convnext-atto
"""
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms as T
from datasets import load_dataset
from transformers import ConvNextV2ForImageClassification

from train_vision_moment import build_model  # 同ディレクトリの定義を再利用


def main():
    p = ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"

    if args.net.startswith("convnext"):
        model = ConvNextV2ForImageClassification.from_pretrained(args.ckpt_dir)
    else:
        # RegNet/Swin (HfWrapper経由) はbuild_modelで骨格を作り、state_dictをロード
        model = build_model(args.net, args.num_classes, args.image_size)
        state_dict = torch.load(
            Path(args.ckpt_dir) / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device).eval()

    real_ds = load_dataset("uoft-cs/cifar10")
    eval_ds = real_ds["test"]
    img_key = "img" if "img" in eval_ds.column_names else "image"
    label_key = "label" if "label" in eval_ds.column_names else "fine_label"

    tf = T.Compose([T.ToTensor()])

    def collate(batch):
        return {
            "pixel_values": torch.stack(
                [tf(x[img_key].convert("RGB")) for x in batch]),
            "labels": torch.tensor([x[label_key] for x in batch]),
        }

    loader = torch.utils.data.DataLoader(eval_ds, batch_size=512, collate_fn=collate)

    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            out = model(pixel_values=pixel_values, labels=labels)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    print(f"{args.ckpt_dir}: test_acc = {correct/total:.4f}")


if __name__ == "__main__":
    main()