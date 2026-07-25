# scripts/train_vision_plain.py
"""
論文 "Neural Networks Learn Statistics of Increasing Complexity" (Belrose et al., ICML 2024)
の vision 実験(Sec. 3.2)を再現する学習スクリプト。

論文本文(Sec. 3.2)に明記されている仕様:
  - steps: 2^16 = 65536, batch size: 128
  - ConvNeXt V2 / Swin V2: AdamW(beta1=0.9, beta2=0.95), warmup=2000
  - RegNet-Y: SGD with momentum, no LR warmup
  - Data augmentation: RandAugment -> random horizontal flip -> random crop

著者公式リポジトリ(features-across-time, scripts/inference/train_vision.py)を
直接確認して判明した実装上の詳細(論文本文には無いか、本文と食い違うもの):
  - 全アーキ共通の lr_scheduler_type は "cosine"(本文は "linear" と記載、コードが正)
  - RegNet-Y は torchvision.models.regnet_y_* を使用(HFのRegNet実装ではない)
    SGD(lr=0.005, momentum=0.9, weight_decay=5e-5), cosine schedule, warmup=0
    さらに net.stem[0].stride を (2,2) -> (1,1) に変更(低解像度画像向け調整)
  - Swin V2 は torchvision.models.swin_transformer.SwinTransformer を使用
    (V2ブロック、patch_size=[2,2]、stochastic_depth_prob=0.2)
  - ConvNeXt V2 は transformers.ConvNextV2ForImageClassification(著者と同一実装)
    drop_path_rate=0.1、patch_size=1(低解像度画像向け調整)
  - data augmentation の RandomCrop は padding=h//8 (h=32 の場合 padding=4)
  - weight_decay=0.05 は AdamW(ConvNeXt/Swin)にのみ適用。RegNetはSGD側で
    weight_decay=5e-5 を別途指定
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torchvision.transforms as T
from datasets import load_dataset
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    Trainer, TrainingArguments, TrainerCallback,
)
from transformers.modeling_outputs import ModelOutput
from transformers.optimization import get_cosine_schedule_with_warmup


_DATASET_ALIASES = {"cifar10": "uoft-cs/cifar10"}


class HfWrapper(nn.Module):
    """torchvision モデルの出力を HF Trainer が期待する形式に変換する。
    著者コード train_vision.py L138-148 に準拠。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        loss = (
            torch.nn.functional.cross_entropy(logits, labels)
            if labels is not None else None
        )
        return ModelOutput(logits=logits, loss=loss)


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
                drop_path_rate=0.1,   # 著者コード L369
                patch_size=1,          # 著者コード L375: 低解像度画像向け
            )
        )

    if net_str.startswith("regnet"):
        from torchvision.models import (
            regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf,
        )
        net = {
            "400mf": regnet_y_400mf,
            "800mf": regnet_y_800mf,
            "1.6gf": regnet_y_1_6gf,
            "3.2gf": regnet_y_3_2gf,
        }[arch or "400mf"](num_classes=num_classes)
        net.stem[0].stride = (1, 1)   # 著者コード L399: 低解像度画像向け
        return HfWrapper(net)

    if net_str.startswith("swin"):
        from torchvision.models.swin_transformer import (
            SwinTransformer, SwinTransformerBlockV2, PatchMergingV2,
        )
        cfg = {
            "atto":  ([2, 4, 8, 16], 40),
            "femto": ([2, 4, 8, 16], 48),
            "pico":  ([2, 4, 8, 16], 64),
            "nano":  ([2, 4, 8, 16], 80),
            "tiny":  ([3, 6, 12, 24], 96),
        }[arch or "atto"]
        num_heads, embed_dim = cfg
        swin = SwinTransformer(
            patch_size=[2, 2],   # 著者コード L419: 低解像度画像向け
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=num_heads,
            window_size=[7, 7],
            num_classes=num_classes,
            stochastic_depth_prob=0.2,
            block=SwinTransformerBlockV2,
            downsample_layer=PatchMergingV2,
        )
        return HfWrapper(swin)

    raise ValueError(f"unknown net: {net_str}")


def main():
    p = ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--nets", nargs="+",
                   default=["convnext-atto", "regnet-400mf", "swin-atto"])
    p.add_argument("--max-steps", type=int, default=65536)   # 論文: 2^16
    p.add_argument("--batch-size", type=int, default=128)    # 論文: 128
    p.add_argument("--ckpt-base", type=float, default=2.0,
                   help="1.19 で約80点, 1.09 で約160点")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs-plain"))
    args = p.parse_args()

    ds = load_dataset(_DATASET_ALIASES.get(args.dataset, args.dataset))
    img_key = "img" if "img" in ds["train"].column_names else "image"
    label_key = "label" if "label" in ds["train"].column_names else "fine_label"
    num_classes = ds["train"].features[label_key].num_classes
    image_size = ds["train"][0][img_key].size[0]

    # 著者コード L228-234: RandAugment -> RandomHorizontalFlip -> RandomCrop
    # padding = h // 8 (h=32 の場合 padding=4)
    train_tf = T.Compose([
        T.RandAugment(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.ToTensor(),
    ])

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
        is_regnet = net_str.startswith("regnet")

        targs = TrainingArguments(
            output_dir=str(run_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=512,
            # ConvNeXt/Swin: 論文通り lr=1e-3, AdamW(beta1=0.9, beta2=0.95)
            # RegNet: 以下で optimizer/scheduler を明示的に上書きするため、
            # ここでの learning_rate/scheduler 指定は実質使われない
            learning_rate=1e-3,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=0.05,   # 著者コード L303: AdamW側のweight_decay
            lr_scheduler_type="cosine",   # 著者コード L295: 全アーキ共通でcosine
            warmup_steps=0 if is_regnet else 2000,
            save_strategy="no",          # コールバック駆動
            save_total_limit=None,
            eval_strategy="no",
            logging_steps=100,
            seed=args.seed,
            data_seed=args.seed,
            bf16=torch.cuda.is_bf16_supported(),
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds["train"],
            data_collator=collate(train_tf),
            callbacks=[LogSpacedCheckpoint(base=args.ckpt_base)],
        )

        if is_regnet:
            # 著者公式コード(train_vision.py L445-447)に厳密準拠:
            #   opt = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-5)
            #   schedule = get_cosine_schedule_with_warmup(opt, 0, args.max_steps)
            opt = torch.optim.SGD(
                model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-5
            )
            sched = get_cosine_schedule_with_warmup(
                opt, num_warmup_steps=0, num_training_steps=args.max_steps
            )
            trainer.optimizer = opt
            trainer.lr_scheduler = sched
            trainer.create_optimizer = lambda: opt
            trainer.create_scheduler = lambda num_training_steps, optimizer=None: sched

        trainer.train()
        trainer.save_model(str(run_dir / "final"))


if __name__ == "__main__":
    main()
