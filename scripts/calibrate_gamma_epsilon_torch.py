# scripts/calibrate_gamma_epsilon_torch.py
"""
devinterp (v1.3.2) を用いた PyTorch版 SGLD ハイパラキャリブレーション。
JAX版 calibrate_gamma_epsilon_dense.py と同じ設計思想:
  - gamma(localization) x epsilon(lr) のグリッドサーチ
  - MALA acceptance rate が 0.9〜0.95 になる組を探す
  - loss trace の平坦性(post_burnin - init_loss)も併せて確認

nbeta は devinterp の default_nbeta(batch_sizeベース)ではなく、
JAX版と揃えるため n_train/log(n_train) を明示的に指定する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVINTERP_SRC = Path("/home/nakano/src/github.com/timaeus-research/devinterp/src")
sys.path.insert(0, str(DEVINTERP_SRC))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_file

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.slt.mala import MalaAcceptanceRate

from scripts.train_vision_plain import build_model


def evaluate_fn(model, data):
    x, y = data
    out = model(pixel_values=x, labels=None)
    logits = out.logits if hasattr(out, "logits") else out
    loss = F.cross_entropy(logits, y)
    return loss, {}


def load_checkpoint_model(net_str, ckpt_dir, num_classes, image_size, device):
    model = build_model(net_str, num_classes, image_size)
    st_path = Path(ckpt_dir) / "model.safetensors"
    state_dict = load_file(str(st_path), device=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def flatness_score(loss_trace, tail_frac=0.1):
    trace = np.array(loss_trace)
    valid = trace[~np.isnan(trace) & ~np.isinf(trace)]
    if len(valid) < 10:
        return float("nan")
    n_tail = max(int(len(valid) * tail_frac), 10)
    tail = valid[-n_tail:]
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail, 1)[0]
    scale = np.mean(np.abs(tail)) + 1e-8
    return float(slope / scale)


def load_existing(out_path):
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {"records": []}


def already_computed(existing, net, step, gamma, epsilon):
    for r in existing["records"]:
        if (r["net"] == net and r["step"] == step
                and r["gamma"] == gamma and r["epsilon"] == epsilon):
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--gammas", type=float, nargs="+", required=True)
    p.add_argument("--epsilons", type=float, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/home/nakano/server/calibration_results_vision"))
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--sgld-num-draws", type=int, default=3000)
    p.add_argument("--sgld-batch-size", type=int, default=512)
    p.add_argument("--n-calibration-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"calib_{args.net}.json"
    existing = load_existing(out_path)
    print(f"Loaded {len(existing['records'])} existing records from {out_path}")

    from datasets import load_dataset
    import torchvision.transforms as T
    ds = load_dataset("uoft-cs/cifar10")["train"]
    if args.n_calibration_samples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.n_calibration_samples))

    tf = T.ToTensor()
    x_all = torch.stack([tf(img.convert("RGB")) for img in ds["img"]])
    y_all = torch.tensor(ds["label"])
    n_calib = len(x_all)

    nbeta_val = n_calib / np.log(n_calib)
    print(f"Calibration data: n={n_calib}, nbeta={nbeta_val:.4f}")

    total = len(args.steps) * len(args.gammas) * len(args.epsilons)
    idx = 0

    for step in args.steps:
        ckpt_dir = args.run_dir / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            print(f"WARNING: checkpoint step={step} not found, skipping.")
            continue

        for gamma in args.gammas:
            for epsilon in args.epsilons:
                idx += 1
                if already_computed(existing, args.net, step, gamma, epsilon):
                    print(f"[{idx}/{total}] step={step} gamma={gamma} epsilon={epsilon:.1e} -- skip")
                    continue

                t0 = time.time()
                model = load_checkpoint_model(args.net, ckpt_dir, args.num_classes, args.image_size, device)

                dataset = TensorDataset(x_all, y_all)
                loader = DataLoader(dataset, batch_size=args.sgld_batch_size, shuffle=True)

                mala_cb = MalaAcceptanceRate(
                    num_chains=1, num_draws=args.sgld_num_draws,
                    learning_rate=epsilon, nbeta=nbeta_val, device=device,
                )

                torch.manual_seed(args.seed)
                result = estimate_learning_coeff_with_summary(
                    model=model,
                    loader=loader,
                    evaluate=evaluate_fn,
                    sampling_method=SGLD,
                    optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta_val),
                    num_draws=args.sgld_num_draws,
                    num_chains=1,
                    callbacks=[mala_cb],
                    device=device,
                    verbose=False,
                )

                loss_trace = result["loss/trace"][0]
                mean_accept = mala_cb.get_results()["mala_accept/mean"]
                flat_score = flatness_score(loss_trace)
                elapsed = time.time() - t0

                record = {
                    "net": args.net,
                    "step": step,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "llc_mean": float(result["llc/mean"]),
                    "mean_accept_prob": float(mean_accept),
                    "in_target_range_0.9_0.95": bool(0.9 <= mean_accept <= 0.95),
                    "flatness_score": flat_score,
                    "loss_trace": [float(l) for l in loss_trace],
                }
                existing["records"].append(record)
                with open(out_path, "w") as f:
                    json.dump(existing, f)

                status = "IN_RANGE" if record["in_target_range_0.9_0.95"] else "ok"
                print(f"[{idx}/{total}] step={step:6d} gamma={gamma:5.1f} epsilon={epsilon:.1e} "
                      f"| accept={mean_accept:.3f} | flat={flat_score:.5f} | {status} | {elapsed:.1f}s")

                del model
                torch.cuda.empty_cache()

    print(f"\nAll done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
