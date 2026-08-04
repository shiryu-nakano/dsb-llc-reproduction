# scripts/calibrate_gamma_epsilon_deletion.py
"""
calibrate_gamma_epsilon_torch.py をベースに、moment-edited data (deletion)
で学習したモデルの training loss LLC 較正に対応させたスクリプト。

calibrate_gamma_epsilon_torch.py からの変更点:
  1. データ読み込み: 実CIFAR-10ではなく、対象モデルが実際に学習に使った
     moment-edited data (--data-family / --data-method) を読み込む。
     (θ* 自身の学習データ上でのtraining loss LLCを計算するため)
  2. --run-dir は checkpoints_dense/deletion_vision/deletion_{method}/
     {net}/seed{seed} を渡す想定 (呼び出し側で組み立てる)
  3. build_model は train_vision_moment からインポート (train_vision_plunと同一実装)
  4. WeightDistanceTracker を追加し、各SGLDステップで ||w_t - w*||_2 を記録する。
     w* はモデルロード直後、サンプリング開始前にスナップショットしたものを使う
     (コールバック内で「最初の呼び出し」を基準にすると、既に1ステップ進んだ後の
     値を誤ってw*とみなしてしまう罠があるため)。
  5. FixedBatchMalaAcceptanceRate を追加。devinterp標準のMalaAcceptanceRateは
     毎drawごとに異なるミニバッチのlossを比較しており、ミニバッチノイズが
     受理率に混入する問題がある。固定した1つの評価バッチ上で(check_everyごとに)
     前後比較することで、この問題を回避した受理率を別途計算する。
     (calibrate_gamma_epsilon_torch.py / vision_plain較正での知見をそのまま移植)

それ以外 (SGLD呼び出し、nbeta計算式、JSON保存形式、stepごとファイル分割)は
完全に同一。既存の較正ノウハウ(0.9-0.95のacceptance目標、loss trace flatness確認、
収束点を使うこと)もそのまま引き継ぐ。
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

from scripts.train_vision_moment import build_model
from scripts.weight_distance_tracker import WeightDistanceTracker
from scripts.fixed_batch_mala import FixedBatchMalaAcceptanceRate


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


def load_moment_data(data_root, data_family, data_method, n_calibration_samples, seed):
    """moment-edited data (npz) を読み込み、x_all, y_all の tensor を返す。
    train_vision_moment.py の MomentEditedDataset と同じラベル選択ロジック
    (deletion: original_label, grafting: target_label) を使う。"""
    dir_path = Path(data_root) / data_family / data_method
    npz_files = sorted(dir_path.glob("*.npz"))
    assert len(npz_files) == 1, f"Expected exactly one .npz in {dir_path}, found {npz_files}"
    d = np.load(npz_files[0])

    images = d["pixel_values"]
    if images.dtype != np.uint8:
        if images.max() <= 1.5:
            images = (images * 255.0).clip(0, 255)
        images = images.astype(np.uint8)

    if data_family == "deletion":
        labels = d["original_label"]
    elif data_family == "grafting":
        labels = d["target_label"]
    else:
        raise ValueError(f"Unknown data_family: {data_family}")

    n_total = len(images)
    if n_calibration_samples < n_total:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=n_calibration_samples, replace=False)
        images = images[idx]
        labels = labels[idx]

    x_all = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    y_all = torch.from_numpy(labels).long()
    return x_all, y_all


def flatness_score(trace, tail_frac=0.1):
    trace = np.array(trace)
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
    p.add_argument("--run-dir", type=Path, required=True,
                    help="例: /home/nakano/server/checkpoints_dense/deletion_vision/"
                         "deletion_conrad/convnext-atto/seed42")
    p.add_argument("--data-family", choices=["deletion", "grafting"], required=True)
    p.add_argument("--data-method", required=True)
    p.add_argument("--data-root", default="/home/nakano/server/moment_data")
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--gammas", type=float, nargs="+", required=True)
    p.add_argument("--epsilons", type=float, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path,
                    default=Path("/home/nakano/server/calibration_results_deletion_vision"))
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--sgld-total-steps", type=int, default=3000)
    p.add_argument("--sgld-batch-size", type=int, default=512)
    p.add_argument("--n-calibration-samples", type=int, default=50000)
    p.add_argument("--mala-check-every", type=int, default=20,
                    help="固定バッチMALA受理率を計算する間隔")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all = load_moment_data(
        args.data_root, args.data_family, args.data_method,
        args.n_calibration_samples, args.seed)
    n_calib = len(x_all)

    nbeta_val = n_calib / np.log(n_calib)
    print(f"Calibration data: family={args.data_family} method={args.data_method} "
          f"n={n_calib}, nbeta={nbeta_val:.4f}")

    # 固定評価バッチ (MALA受理率計算専用、SGLD本体のミニバッチとは別)
    fixed_idx = torch.randperm(n_calib)[:args.sgld_batch_size]
    fixed_batch_data = (x_all[fixed_idx].to(device), y_all[fixed_idx].to(device))

    total = len(args.steps) * len(args.gammas) * len(args.epsilons)
    idx = 0

    for step in args.steps:
        ckpt_dir = args.run_dir / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            print(f"WARNING: checkpoint step={step} not found ({ckpt_dir}), skipping.")
            continue

        out_path = args.out_dir / f"calib_{args.net}_{args.data_family}_{args.data_method}_{args.run_dir.name}_step{step}.json"
        existing = load_existing(out_path)
        print(f"[step={step}] Loaded {len(existing['records'])} existing records from {out_path}")

        for gamma in args.gammas:
            for epsilon in args.epsilons:
                idx += 1
                if already_computed(existing, args.net, step, gamma, epsilon):
                    print(f"[{idx}/{total}] step={step} gamma={gamma} epsilon={epsilon:.1e} -- skip")
                    continue

                t0 = time.time()
                model = load_checkpoint_model(args.net, ckpt_dir, args.num_classes, args.image_size, device)

                # w* をサンプリング開始前にスナップショット (コールバック内で
                # 「最初の呼び出し」を基準にすると1ステップ進んだ後の値を
                # 誤ってw*とみなしてしまう罠があるため、ここで確定させる)
                w_star = [
                    p.clone().detach() for p in model.parameters() if p.requires_grad
                ]

                dataset = TensorDataset(x_all, y_all)
                loader = DataLoader(dataset, batch_size=args.sgld_batch_size, shuffle=True)

                mala_cb = MalaAcceptanceRate(
                    num_chains=1, num_draws=args.sgld_total_steps,
                    learning_rate=epsilon, nbeta=nbeta_val, device=device,
                )
                dist_cb = WeightDistanceTracker(
                    num_chains=1, num_draws=args.sgld_total_steps,
                    initial_params=w_star, device=device,
                )
                fixed_mala_cb = FixedBatchMalaAcceptanceRate(
                    fixed_batch=fixed_batch_data, epsilon=epsilon, nbeta=nbeta_val,
                    check_every=args.mala_check_every, device=device,
                )

                torch.manual_seed(args.seed)
                result = estimate_learning_coeff_with_summary(
                    model=model,
                    loader=loader,
                    evaluate=evaluate_fn,
                    sampling_method=SGLD,
                    optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta_val),
                    num_draws=args.sgld_total_steps,
                    num_burnin_steps=0,
                    num_chains=1,
                    callbacks=[mala_cb, dist_cb, fixed_mala_cb],
                    device=device,
                    verbose=False,
                )

                loss_trace = result["loss/trace"][0].tolist()
                mala_results = mala_cb.get_results()
                mala_trace = mala_results["mala_accept/trace"][0].tolist()
                mean_accept = float(mala_results["mala_accept/mean"])
                dist_results = dist_cb.get_results()
                dist_trace = dist_results["weight_distance/trace"][0].tolist()
                fixed_mala_results = fixed_mala_cb.get_results()
                flat_score = flatness_score(loss_trace)
                elapsed = time.time() - t0

                record = {
                    "net": args.net,
                    "data_family": args.data_family,
                    "data_method": args.data_method,
                    "step": step,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "nbeta": nbeta_val,
                    "total_steps": args.sgld_total_steps,
                    "llc_mean_devinterp_full_avg": float(result["llc/mean"]),
                    "mean_accept_prob_full": mean_accept,
                    "flatness_score_last10pct": flat_score,
                    "loss_trace": loss_trace,
                    "mala_accept_trace": mala_trace,
                    "weight_distance_trace": dist_trace,
                    "weight_distance_final": dist_results["weight_distance/final"],
                    "fixed_batch_mala_accept_trace": fixed_mala_results["fixed_batch_mala_accept/trace"],
                    "fixed_batch_mala_accept_mean": fixed_mala_results["fixed_batch_mala_accept/mean"],
                }
                existing["records"].append(record)
                with open(out_path, "w") as f:
                    json.dump(existing, f)

                print(f"[{idx}/{total}] step={step:6d} gamma={gamma:5.1f} epsilon={epsilon:.1e} "
                      f"| accept_minibatch={mean_accept:.3f} "
                      f"| accept_fixedbatch={fixed_mala_results['fixed_batch_mala_accept/mean']:.3f} "
                      f"| flat={flat_score:.5f} "
                      f"| dist_final={dist_results['weight_distance/final']:.4e} | {elapsed:.1f}s")

                del model
                torch.cuda.empty_cache()

    print(f"\nAll done.")


if __name__ == "__main__":
    main()