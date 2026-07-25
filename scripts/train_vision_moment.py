# scripts/train_vision_moment.py
"""
train_vision_plain.py をベースに、moment-edited data (grafting/deletion) での
学習・NFS保存・チェックポイント方式に対応させたスクリプト。

train_vision_plain.py からの変更点:
  1. データ読み込み: --data-family / --data-method / --data-root を指定すると
     実CIFAR-10の代わりに moment-edited train data (npz) を読み込む。
     test set は常に実CIFAR-10 (Belrose et al. の設計:
     編集データで学習し、実データでの汎化をprobeとして評価する)。
  2. チェックポイント保存: LogSpacedCheckpoint (幾何級数) をやめ、
     0〜max_steps を等間隔 num_points 点 (デフォルト505) で保存する
     ExplicitStepCheckpoint に変更。--save-steps-file で外部リストの指定も可能。
  3. save_only_model=True: optimizer/scheduler stateを保存せず軽量化。
  4. --out のデフォルトを NFS 上の絶対パス
     (/home/nakano/server/checkpoints_dense/deletion_vision) に固定。
     (~/cuda_test はローカルディスクで空き容量が少ないため、
      チェックポイント保存には使わないこと)
  5. --max-steps のデフォルトを 100000 に変更 (Belrose論文の65536ではなく、
     本実験用の値として明示的に指定)。

それ以外 (モデル定義, augmentation, optimizer 設定) は train_vision_plain.py と同一。
"""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset
from transformers import (
    ConvNextV2Config, ConvNextV2ForImageClassification,
    Trainer, TrainingArguments, TrainerCallback,
)
from transformers.modeling_outputs import ModelOutput
from transformers.optimization import get_cosine_schedule_with_warmup


_DATASET_ALIASES = {"cifar10": "uoft-cs/cifar10"}


class HfWrapper(nn.Module):
    """torchvision モデルの出力を HF Trainer が期待する形式に変換する。"""
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


# ─────────────────────────────────────────────
# moment-edited data 用 Dataset
# ─────────────────────────────────────────────
class MomentEditedDataset(Dataset):
    """
    deletion/grafting の npz を読み込み、
    __getitem__(idx) -> {"img": PIL.Image, "label": int} を返す。
    train_vision_plain.py の collate() がそのまま使える形式に合わせている。
    """
    def __init__(self, data_root, data_family, data_method):
        dir_path = Path(data_root) / data_family / data_method
        npz_files = sorted(dir_path.glob("*.npz"))
        assert len(npz_files) == 1, (
            f"Expected exactly one .npz in {dir_path}, found {npz_files}"
        )
        npz_path = npz_files[0]
        print(f"Loading moment-edited data from: {npz_path}")

        d = np.load(npz_path)
        for required_key in ("pixel_values", "original_label", "target_label"):
            assert required_key in d.files, f"{required_key} missing in {npz_path}"

        images = d["pixel_values"]
        # uint8 [0,255] を想定 (PIL.Image.fromarray用)。float [0,1] で来た場合は変換。
        if images.dtype != np.uint8:
            if images.max() <= 1.5:
                images = (images * 255.0).clip(0, 255)
            images = images.astype(np.uint8)
        self.images = images

        original_label = d["original_label"]
        target_label   = d["target_label"]

        if data_family == "deletion":
            n_mismatch = int(np.sum(original_label != target_label))
            if n_mismatch > 0:
                print(
                    f"  WARNING: deletion data だが original_label と target_label が "
                    f"{n_mismatch}/{len(original_label)} 件で不一致です。"
                    f"original_label を教師信号として使用します。"
                )
            self.labels = original_label
        elif data_family == "grafting":
            self.labels = target_label
        else:
            raise ValueError(f"Unknown data_family: {data_family}")

        self.num_classes = int(len(np.unique(self.labels)))
        print(f"  images: {self.images.shape} {self.images.dtype}, "
              f"labels: {self.labels.shape}, n_classes_used={self.num_classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return {"img": img, "label": int(self.labels[idx])}

    @property
    def column_names(self):
        return ["img", "label"]


# ─────────────────────────────────────────────
# 等間隔 save_steps 対応チェックポイントコールバック
# ─────────────────────────────────────────────
@dataclass
class ExplicitStepCheckpoint(TrainerCallback):
    """save_steps に含まれる global_step でのみ保存する。"""
    save_steps: List[int] = field(default_factory=list)

    def __post_init__(self):
        self._save_steps_set = set(self.save_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self._save_steps_set:
            control.should_save = True
        return control


def load_save_steps(path: Optional[str], max_steps: int, num_points: int = 505) -> List[int]:
    """
    save_steps リストをファイルから読み込む。指定がなければ
    0〜max_steps を等間隔 num_points 点で生成する。
    """
    if path is not None:
        p = Path(path)
        if p.suffix == ".json":
            import json
            steps = json.loads(p.read_text())
        else:
            steps = [int(x) for x in p.read_text().split() if x.strip()]
    else:
        steps = sorted(set(int(x) for x in np.linspace(0, max_steps, num_points)))
    steps = sorted(s for s in set(steps) if s <= max_steps)
    interval = steps[1] - steps[0] if len(steps) > 1 else None
    print(f"save_steps: {len(steps)} points, interval≈{interval}, "
          f"range=[{steps[0]}, {steps[-1]}]")
    return steps


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
                drop_path_rate=0.1,
                patch_size=1,
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
        net.stem[0].stride = (1, 1)
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
            patch_size=[2, 2],
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
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("/home/nakano/server/checkpoints_dense/deletion_vision"))

    # ── moment-edited data 設定 ──
    p.add_argument("--data-family", choices=["deletion", "grafting"], default=None,
                    help="指定しなければ実CIFAR-10で学習 (train_vision_plain.py と同じ挙動)")
    p.add_argument("--data-method", default=None,
                    help="例: gaussian, ics, conrad, truncated_normal (deletion), "
                         "bounded_shift, cqn, gaussian_ot (grafting)")
    p.add_argument("--data-root", default="/home/nakano/server/moment_data")

    # ── save_steps 設定 ──
    p.add_argument("--save-steps-file", default=None,
                    help="save_steps を記載したファイル (改行区切り整数 or JSON list)。"
                         "指定しなければ 0〜max_steps を等間隔505点で自動生成する。")
    p.add_argument("--num-save-points", type=int, default=505)

    args = p.parse_args()

    if args.data_family is not None:
        assert args.data_method is not None, \
            "--data-family を指定したら --data-method も指定してください"

    # ── train set: 実データ or moment-edited ──
    if args.data_family is not None:
        train_dataset = MomentEditedDataset(
            args.data_root, args.data_family, args.data_method)
        img_key, label_key = "img", "label"
        num_classes = train_dataset.num_classes
        image_size = train_dataset.images.shape[1]
        run_name = f"{args.data_family}_{args.data_method}"
    else:
        ds = load_dataset(_DATASET_ALIASES.get(args.dataset, args.dataset))
        train_dataset = ds["train"]
        img_key = "img" if "img" in ds["train"].column_names else "image"
        label_key = "label" if "label" in ds["train"].column_names else "fine_label"
        num_classes = ds["train"].features[label_key].num_classes
        image_size = ds["train"][0][img_key].size[0]
        run_name = "real"

    # ── test set: 常に実CIFAR-10 (汎化probe用) ──
    real_ds = load_dataset(_DATASET_ALIASES.get(args.dataset, args.dataset))
    eval_dataset = real_ds["test"]
    eval_img_key = "img" if "img" in real_ds["test"].column_names else "image"
    eval_label_key = "label" if "label" in real_ds["test"].column_names else "fine_label"

    train_tf = T.Compose([
        T.RandAugment(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(image_size, padding=image_size // 8),
        T.ToTensor(),
    ])
    eval_tf = T.Compose([T.ToTensor()])

    def collate(tf, i_key, l_key):
        def fn(batch):
            return {
                "pixel_values": torch.stack(
                    [tf(x[i_key].convert("RGB")) for x in batch]),
                "labels": torch.tensor([x[l_key] for x in batch]),
            }
        return fn

    save_steps = load_save_steps(args.save_steps_file, args.max_steps, args.num_save_points)

    for net_str in args.nets:
        model = build_model(net_str, num_classes, image_size)
        run_dir = args.out / run_name / net_str / f"seed{args.seed}"
        is_regnet = net_str.startswith("regnet")

        targs = TrainingArguments(
            output_dir=str(run_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=512,
            learning_rate=1e-3,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            warmup_steps=0 if is_regnet else 2000,
            save_strategy="no",       # コールバック駆動 (ExplicitStepCheckpoint)
            save_only_model=True,     # optimizer/scheduler stateを保存せず軽量化
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
            train_dataset=train_dataset,
            data_collator=collate(train_tf, img_key, label_key),
            callbacks=[ExplicitStepCheckpoint(save_steps=save_steps)],
        )

        if is_regnet:
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

        # ── 実データでの最終test accuracy (汎化probe) をログ ──
        with torch.no_grad():
            eval_collate = collate(eval_tf, eval_img_key, eval_label_key)
            correct, total = 0, 0
            model.eval()
            loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=512, collate_fn=eval_collate)
            device = next(model.parameters()).device
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                out = model(pixel_values=pixel_values, labels=labels)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
            print(f"[{run_name}/{net_str}/seed{args.seed}] "
                  f"final real test_acc = {correct/total:.4f}")


if __name__ == "__main__":
    main()