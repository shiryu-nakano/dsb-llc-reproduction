# scripts/eval_all_checkpoints.py
"""
deletion_vision 配下の全チェックポイントに対して、実CIFAR-10テストセットでの
accuracyを一括評価し、JSONLに逐次保存する。

高速化のポイント:
  - テストセットは1回だけロードしGPU上に保持 (毎回のPIL変換/転送を排除)
  - モデルは1回だけ構築し、state_dictだけ都度差し替え (from_pretrainedの
    再構築・HF Hub通信コストを排除)
  - JSONLに1件ずつ即書き込み + 既存分をスキップ (resume対応)

使用例:
  CUDA_VISIBLE_DEVICES=2 python3 eval_all_checkpoints.py \
      --methods conrad gaussian \
      --out /home/nakano/server/llc_results_dense/deletion_vision_acc_curve_gpu2.jsonl

  CUDA_VISIBLE_DEVICES=3 python3 eval_all_checkpoints.py \
      --methods ics truncated_normal \
      --out /home/nakano/server/llc_results_dense/deletion_vision_acc_curve_gpu3.jsonl
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import torch
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


def main():
    p = ArgumentParser()
    p.add_argument("--base-dir", default="/home/nakano/server/checkpoints_dense/deletion_vision")
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

    # ── resume: 既に計算済みの (method, seed, step) を読み込む ──
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["method"], r["seed"], r["step"]))
        print(f"Resuming: {len(done)} already computed, skipping them")

    # ── テストセットを1回だけロードしGPU上に保持 ──
    real_ds = load_dataset("uoft-cs/cifar10")
    eval_ds = real_ds["test"]
    img_key = "img" if "img" in eval_ds.column_names else "image"
    label_key = "label" if "label" in eval_ds.column_names else "fine_label"
    tf = T.Compose([T.ToTensor()])

    print("Preloading test set onto GPU...")
    imgs = torch.stack([tf(x.convert("RGB")) for x in eval_ds[img_key]]).to(device)
    labels = torch.tensor(eval_ds[label_key]).to(device)
    print(f"  imgs={tuple(imgs.shape)}, labels={tuple(labels.shape)}")

    # ── モデルは1回だけ構築 ──
    assert args.net == "convnext-atto", "現状 convnext-atto のみ対応"
    model = build_convnext_atto().to(device).eval()

    f_out = open(out_path, "a")
    for method in args.methods:
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

                correct, total = 0, 0
                bs = 1000
                with torch.no_grad():
                    for i in range(0, imgs.shape[0], bs):
                        out = model(pixel_values=imgs[i:i + bs])
                        preds = out.logits.argmax(dim=-1)
                        correct += (preds == labels[i:i + bs]).sum().item()
                        total += labels[i:i + bs].numel()
                acc = correct / total

                rec = {"method": method, "seed": seed, "step": step, "test_acc": acc}
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()  # 即座にディスクへ (中断への耐性)

            print(f"  done: {method} seed{seed}")

    f_out.close()
    print("All evaluations complete.")


if __name__ == "__main__":
    main()