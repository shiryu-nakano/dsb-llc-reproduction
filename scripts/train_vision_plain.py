# scripts/inference/train_vision_plain.py
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
from datasets import load_dataset
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    RegNetConfig, RegNetForImageClassification,
    SwinConfig, SwinForImageClassification,
    Trainer, TrainingArguments, TrainerCallback,
)


@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    """Save at 1, base, base^2, ... steps."""
    base: float = 2.0
    next: int = 1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.next:
            self.next = max(self.next + 1, round(self.next * self.base))
            control.should_save = True
        return control


def build_model(net_str: str, num_classes: int, image_size: int):
    arch = net_str.partition("-")[2]

    if net_str.startswith("convnext"):
        cfg = {
            "atto":  ([40, 80, 160, 320],  [2, 2, 6, 2]),
            "femto": ([48, 96, 192, 384],  [2, 2, 6, 2]),
            "pico":  ([64, 128, 256, 512], [2, 2, 6, 2]),
            "nano":  ([80, 160, 320, 640], [2, 2, 8, 2]),
            "tiny":  ([96, 192, 384, 768], [3, 3, 9, 3]),
        }[arch or "atto"]
        return ConvNextV2ForImageClassification(
            ConvNextV2Config(
                image_size=image_size, num_labels=num_classes,
                hidden_sizes=cfg[0], depths=cfg[1],
                patch_size=1 if image_size <= 64 else 4,
            )
        )

    if net_str.startswith("regnet"):
        cfg = {
            "400mf": ([32, 64, 160, 384],  [1, 2, 7, 12], 16),
            "800mf": ([64, 128, 288, 672], [1, 3, 8, 2],  16),
            "1.6gf": ([72, 168, 408, 912], [2, 4, 10, 2], 24),
        }[arch or "400mf"]
        return RegNetForImageClassification(
            RegNetConfig(
                num_labels=num_classes, embedding_size=32,
                hidden_sizes=cfg[0], depths=cfg[1], groups_width=cfg[2],
                layer_type="y",
            )
        )

    if net_str.startswith("swin"):
        cfg = {
            "atto":  (40,  [2, 2, 6, 2],  [2, 4, 8, 16]),
            "femto": (48,  [2, 2, 6, 2],  [2, 4, 8, 16]),
            "pico":  (64,  [2, 2, 6, 2],  [2, 4, 8, 16]),
            "nano":  (80,  [2, 2, 8, 2],  [2, 4, 8, 16]),
            "tiny":  (96,  [2, 2, 6, 2],  [3, 6, 12, 24]),
        }[arch or "atto"]
        return SwinForImageClassification(
            SwinConfig(
                image_size=image_size, num_labels=num_classes,
                embed_dim=cfg[0], depths=cfg[1], num_heads=cfg[2],
                patch_size=1 if image_size <= 32 else 2,
                window_size=4 if image_size <= 32 else 7,
            )
        )

    raise ValueError(f"unknown net: {net_str}")


def main():
    p = ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--nets", nargs="+",
                   default=["convnext-atto", "regnet-400mf", "swin-atto"])
    p.add_argument("--max-steps", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ckpt-base", type=float, default=2.0,
                   help="1.19 で約80点, 1.09 で約160点")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs-plain"))
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    img_key = "img" if "img" in ds["train"].column_names else "image"
    label_key = "label" if "label" in ds["train"].column_names else "fine_label"
    num_classes = ds["train"].features[label_key].num_classes
    image_size = ds["train"][0][img_key].size[0]

    train_tf = T.Compose([
        T.RandAugment(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_tf = T.ToTensor()

    def collate(tf):
        def fn(batch):
            return {
                "pixel_values": torch.stack([tf(x[img_key].convert("RGB")) for x in batch]),
                "labels": torch.tensor([x[label_key] for x in batch]),
            }
        return fn

    for net_str in args.nets:
        model = build_model(net_str, num_classes, image_size)
        run_dir = args.out / args.dataset / net_str

        targs = TrainingArguments(
            output_dir=str(run_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=512,
            learning_rate=1e-3 if not net_str.startswith("regnet") else 5e-3,
            adam_beta2=0.95,
            weight_decay=0.05,
            lr_scheduler_type="linear",
            warmup_steps=0 if net_str.startswith("regnet") else 2000,
            save_strategy="no",          # コールバック駆動
            save_total_limit=None,
            save_safetensors=True,
            eval_strategy="no",
            logging_steps=100,
            seed=args.seed,
            data_seed=args.seed,
            bf16=torch.cuda.is_bf16_supported(),
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=8,        # 追加
            dataloader_pin_memory=True,      # 追加
            dataloader_persistent_workers=True,  # 追加: エポック跨ぎでworker再起動を避ける
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds["train"],
            data_collator=collate(train_tf),
            callbacks=[LogSpacedCheckpoint(base=args.ckpt_base)],
        )

        if net_str.startswith("regnet"):
            trainer.create_optimizer = lambda: None
            trainer.optimizer = torch.optim.SGD(
                model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-5
            )

        trainer.train()
        trainer.save_model(str(run_dir / "final"))


if __name__ == "__main__":
    main()