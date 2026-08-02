# scripts/calibrate_gamma_epsilon_torch.py
"""
devinterp (v1.3.2) を用いた PyTorch版 SGLD ハイパラキャリブレーション。

MALA受理率について:
devinterp標準の MalaAcceptanceRate は毎drawごとに異なるミニバッチの
loss同士を比較しており、ミニバッチノイズが受理率に混入する問題が判明した
(JAX版 sgld_utils.py は同一バッチで前後比較していたが、devinterp版は違う)。
FixedBatchMalaAcceptanceRate という自作コールバックで、固定バッチ上での
比較(mean_accept_prob_fixedbatch)も同時に記録する。

weight distance について:
WeightDistanceTracker (calibrate_gamma_epsilon_deletion.py と共通の実装)で
||w_t - w*||_2 の推移を記録する。w* はサンプリング開始前にスナップショット
したものを使う。distanceがどこかで頭打ちになるか、t^0.5型のまま伸び続けるか
を見ることで、探索が有効曲率のスケールに達しているかを判断できる。

保存ファイルは step ごとに分ける(calib_{net}_step{step}.json)。
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
from devinterp.slt.mala import MalaAcceptanceRate, mala_acceptance_probability
from devinterp.slt.callback import SamplerCallback

from scripts.train_vision_plain import build_model
from scripts.weight_distance_tracker import WeightDistanceTracker


class FixedBatchMalaAcceptanceRate(SamplerCallback):
    """
    固定した1つの評価バッチを使い、check_everyステップごとに
    (直前スナップショット時点のパラメータ) vs (現在のパラメータ)
    を同一バッチ上で比較して受理確率を計算する(JAX版と同じロジック)。
    まずは尤度項のみで近似(localization項は含めない)。
    """

    def __init__(self, fixed_batch, epsilon, nbeta, check_every=20, device="cuda"):
        super().__init__(device=device)
        self.fixed_x, self.fixed_y = fixed_batch
        self.epsilon = epsilon
        self.nbeta = nbeta
        self.check_every = check_every
        self.accept_probs = []
        self._prev = None

    def _evaluate_fixed_batch(self, model):
        model.zero_grad()
        out = model(pixel_values=self.fixed_x, labels=None)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits, self.fixed_y)
        mala_loss = loss.detach() * self.nbeta
        loss.backward()

        params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        grads = [p.grad.detach().clone() * self.nbeta for p in model.parameters() if p.requires_grad]
        model.zero_grad()
        return params, grads, mala_loss

    def __call__(self, i, model, **kwargs):
        if i % self.check_every != 0:
            return
        with torch.enable_grad():
            params, grads, mala_loss = self._evaluate_fixed_batch(model)
            if self._prev is not None:
                prev_params, prev_grads, prev_loss = self._prev
                prob = mala_acceptance_probability(
                    prev_params, prev_grads, prev_loss,
                    params, grads, mala_loss,
                    self.epsilon,
                )
                self.accept_probs.append([i, float(prob)])
            self._prev = (params, grads, mala_loss)

    def get_results(self):
        return {
            "fixed_batch_mala_accept/trace": self.accept_probs,
            "fixed_batch_mala_accept/mean": (
                sum(p for _, p in self.accept_probs) / len(self.accept_probs)
                if self.accept_probs else float("nan")
            ),
        }


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
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--gammas", type=float, nargs="+", required=True)
    p.add_argument("--epsilons", type=float, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path,
                    default=Path("/home/nakano/server/calibration_results_vision"))
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

    # 固定評価バッチ(MALA受理率計算専用、SGLD本体とは別)
    fixed_idx = torch.randperm(n_calib)[:args.sgld_batch_size]
    fixed_batch = (x_all[fixed_idx].to(device), y_all[fixed_idx].to(device))

    total = len(args.steps) * len(args.gammas) * len(args.epsilons)
    idx = 0

    for step in args.steps:
        ckpt_dir = args.run_dir / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            print(f"WARNING: checkpoint step={step} not found, skipping.")
            continue

        out_path = args.out_dir / f"calib_{args.net}_step{step}.json"
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

                # w* をサンプリング開始前にスナップショット
                w_star = [
                    p.clone().detach() for p in model.parameters() if p.requires_grad
                ]

                dataset = TensorDataset(x_all, y_all)
                loader = DataLoader(dataset, batch_size=args.sgld_batch_size, shuffle=True)

                mala_cb = MalaAcceptanceRate(
                    num_chains=1, num_draws=args.sgld_total_steps,
                    learning_rate=epsilon, nbeta=nbeta_val, device=device,
                )
                fixed_mala_cb = FixedBatchMalaAcceptanceRate(
                    fixed_batch=fixed_batch, epsilon=epsilon, nbeta=nbeta_val,
                    check_every=args.mala_check_every, device=device,
                )
                dist_cb = WeightDistanceTracker(
                    num_chains=1, num_draws=args.sgld_total_steps,
                    initial_params=w_star, device=device,
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
                    callbacks=[mala_cb, fixed_mala_cb, dist_cb],
                    device=device,
                    verbose=False,
                )

                loss_trace = result["loss/trace"][0].tolist()
                mala_results = mala_cb.get_results()
                mala_trace = mala_results["mala_accept/trace"][0].tolist()
                mean_accept = float(mala_results["mala_accept/mean"])
                flat_score = flatness_score(loss_trace)

                fixed_mala_results = fixed_mala_cb.get_results()
                dist_results = dist_cb.get_results()
                dist_trace = dist_results["weight_distance/trace"][0].tolist()
                elapsed = time.time() - t0

                record = {
                    "net": args.net,
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
                    "fixed_batch_mala_accept_trace": fixed_mala_results["fixed_batch_mala_accept/trace"],
                    "fixed_batch_mala_accept_mean": fixed_mala_results["fixed_batch_mala_accept/mean"],
                    "weight_distance_trace": dist_trace,
                    "weight_distance_final": dist_results["weight_distance/final"],
                }
                existing["records"].append(record)
                with open(out_path, "w") as f:
                    json.dump(existing, f)

                print(f"[{idx}/{total}] step={step:6d} gamma={gamma:5.1f} epsilon={epsilon:.1e} "
                      f"| accept_minibatch={mean_accept:.3f} | accept_fixedbatch={fixed_mala_results['fixed_batch_mala_accept/mean']:.3f} "
                      f"| dist_final={dist_results['weight_distance/final']:.4e} | flat={flat_score:.5f} | {elapsed:.1f}s")

                del model
                torch.cuda.empty_cache()

    print(f"\nAll done.")


if __name__ == "__main__":
    main()