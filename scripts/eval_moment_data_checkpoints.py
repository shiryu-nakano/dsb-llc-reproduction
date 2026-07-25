# scripts/eval_moment_data_checkpoints.py
from argparse import ArgumentParser
from pathlib import Path
import csv
import re

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from scripts.train_vision_plain import build_model


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
def evaluate(model, pixel_values, labels, device, batch_size=512):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    n = pixel_values.shape[0]
    for i in range(0, n, batch_size):
        pv = pixel_values[i:i + batch_size].to(device, non_blocking=True)
        lb = labels[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values=pv, labels=None)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits, lb, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == lb).sum().item()
        total_n += lb.size(0)
    return total_loss / total_n, total_correct / total_n


def load_npz_as_tensor(npz_path: Path, label_key: str, n_subsample: int | None, seed: int = 0):
    """pixel_values は (N,H,W,C) float32、値域[0,1]確認済み。CHWに変換。"""
    d = np.load(npz_path)
    pv = d["pixel_values"]  # (N, H, W, C), already in [0, 1]
    labels = d[label_key]

    n = pv.shape[0]
    if n_subsample is not None and n_subsample < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_subsample, replace=False)
        pv = pv[idx]
        labels = labels[idx]

    pv_t = torch.from_numpy(pv).permute(0, 3, 1, 2).float()  # NHWC -> NCHW
    labels_t = torch.from_numpy(labels).long()
    return pv_t, labels_t


# condition名 -> (npzの相対パス, 評価に使うラベルのキー)
# grafting系は移植先(target_label)として分類されるかを見る(論文Fig.3)
# deletion系は元のクラス(original_label)を維持しているかを見る(論文Fig.5)
CONDITIONS = {
    "graft_bounded_shift":  ("grafting/bounded_shift/shifted_cifar10.npz", "target_label"),
    "graft_cqn":            ("grafting/cqn/cqn_cifar10.npz", "target_label"),
    "graft_gaussian_ot":    ("grafting/gaussian_ot/gaussian_ot_cifar10.npz", "target_label"),
    "del_conrad":           ("deletion/conrad/conrad_cifar10.npz", "original_label"),
    "del_ics":              ("deletion/ics/ics_cifar10.npz", "original_label"),
    "del_gaussian":         ("deletion/gaussian/gaussian_cifar10.npz", "original_label"),
    "del_truncated_normal": ("deletion/truncated_normal/truncated_normal_cifar10.npz", "original_label"),
}


def main():
    p = ArgumentParser()
    p.add_argument("--net", required=True,
                    help="convnext-atto / regnet-400mf / swin-atto")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--moment-data-root", type=Path,
                    default=Path("/home/nakano/server/moment_data"))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()),
                    help="評価するcondition名を指定(省略時は全条件)")
    p.add_argument("--n-subsample", type=int, default=None,
                    help="指定時のみサブサンプリング。デフォルトは全件評価")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    conditions_to_run = {k: CONDITIONS[k] for k in args.conditions}

    cached_data = {}
    for cond_name, (rel_path, label_key) in conditions_to_run.items():
        npz_path = args.moment_data_root / rel_path
        pv, labels = load_npz_as_tensor(npz_path, label_key, args.n_subsample, seed=args.seed)
        cached_data[cond_name] = (pv, labels)
        print(f"loaded {cond_name}: {pv.shape[0]} samples")

    ckpt_dirs = sorted(
        [d for d in args.run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=natural_ckpt_key,
    )

    done = set()
    if args.out_csv.exists():
        with open(args.out_csv) as f:
            for row in csv.DictReader(f):
                done.add((int(row["step"]), row["condition"]))

    write_header = not args.out_csv.exists()
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "condition", "loss", "acc"])

        for ckpt_dir in ckpt_dirs:
            step = natural_ckpt_key(ckpt_dir)

            # このチェックポイントで未評価のconditionが無ければモデルロードもスキップ
            pending = [c for c in cached_data if (step, c) not in done]
            if not pending:
                continue

            try:
                model = build_model(args.net, args.num_classes, args.image_size)
                model = load_weights(model, ckpt_dir, device).to(device)
            except Exception as e:
                print(f"[skip ckpt] {ckpt_dir}: {e}")
                continue

            for cond_name in pending:
                pv, labels = cached_data[cond_name]
                loss, acc = evaluate(model, pv, labels, device, batch_size=args.batch_size)
                writer.writerow([step, cond_name, loss, acc])
                f.flush()
                print(f"step={step:>6}  {cond_name:<22} loss={loss:.4f} acc={acc:.4f}")

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()