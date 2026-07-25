# scripts/eval_vision_plain_checkpoints.py
from argparse import ArgumentParser
from pathlib import Path
import csv
import re
import json

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from safetensors.torch import load_file

from scripts.train_vision_plain import build_model, _DATASET_ALIASES


def natural_ckpt_key(p: Path):
    m = re.search(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1


def load_weights(model, ckpt_dir: Path, device):
    st_path = ckpt_dir / "model.safetensors"
    if st_path.exists():
        state_dict = load_file(str(st_path), device=device)
    else:
        state_dict = torch.load(ckpt_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values=pixel_values, labels=None)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_n += labels.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    p = ArgumentParser()
    p.add_argument("--net", required=True,
                    help="例: convnext-atto / regnet-400mf / swin-atto")
    p.add_argument("--run-dir", type=Path, required=True,
                    help="例: /home/nakano/server/checkpoints_dense/vision_plain/cifar10/convnext-atto")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset(_DATASET_ALIASES.get(args.dataset, args.dataset))
    img_key = "img" if "img" in ds["test"].column_names else "image"
    label_key = "label" if "label" in ds["test"].column_names else "fine_label"
    num_classes = ds["test"].features[label_key].num_classes
    image_size = ds["test"][0][img_key].size[0]

    test_ds = ds["test"]
    tf = T.ToTensor()

    def collate(batch):
        return {
            "pixel_values": torch.stack([tf(x[img_key].convert("RGB")) for x in batch]),
            "labels": torch.tensor([x[label_key] for x in batch]),
        }

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=4, pin_memory=True,
    )

    ckpt_dirs = sorted(
        [d for d in args.run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=natural_ckpt_key,
    )

    done_steps = set()
    if args.out_csv.exists():
        with open(args.out_csv) as f:
            for row in csv.DictReader(f):
                done_steps.add(int(row["step"]))

    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "test_loss", "test_acc"])

        for ckpt_dir in ckpt_dirs:
            step = natural_ckpt_key(ckpt_dir)
            if step in done_steps:
                continue
            try:
                model = build_model(args.net, num_classes, image_size)
                model = load_weights(model, ckpt_dir, device).to(device)
            except Exception as e:
                print(f"[skip] {ckpt_dir}: load failed ({e})")
                continue

            loss, acc = evaluate(model, loader, device)
            writer.writerow([step, loss, acc])
            f.flush()
            print(f"step={step:>6}  test_loss={loss:.4f}  test_acc={acc:.4f}")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()