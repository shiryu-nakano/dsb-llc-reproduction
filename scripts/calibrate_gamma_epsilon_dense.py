"""
calibrate_gamma_epsilon_dense.py

train_and_save.py (Refinetti学習レシピ、10クラスCIFAR10) で保存した
dsb_dense のチェックポイントに対して、SGLDの gamma と epsilon を
両方振ってキャリブレーションする。

Lau et al. (2025) Appendix H の推奨:
  - epsilon: 平均 MALA acceptance probability が 0.9〜0.95 になるよう調整
  - gamma: 1.0〜10.0 の範囲に保つ(Appendix H.3、SGLDが低損失領域に
    迷い込む・発散する場合の対処として、大きくすることが推奨されている)
  - loss trace が burn-in 後に真に平坦であることも同時に確認する
    (accept_probだけでなく、post_burnin - init_loss の符号・大きさも見る)

前回との違い:
  - gamma を外部から指定可能にした(以前は1.0固定だった)
  - 各(gamma, epsilon, step)の組み合わせを計算するたびに、
    その場でJSONに追記保存する(逐次保存)。長時間ジョブが
    途中で落ちても、それまでの結果が失われない。

使用例:
  cd ~/cuda_test/scripts
  XLA_FLAGS="--xla_gpu_autotune_level=0" CUDA_VISIBLE_DEVICES=0 python calibrate_gamma_epsilon_dense.py \
      --ckpt_dir /home/nakano/server/checkpoints_dense/dsb_dense/seed42 \
      --seed_label seed42 \
      --steps 19508 39008 \
      --gammas 1.0 \
      --epsilons 1e-9 3e-9 1e-8 3e-8 1e-7 \
      --out_dir /home/nakano/server/calibration_results_gamma_eps
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LLC_SRC = Path.home() / "src" / "github.com" / "shiryu-nakano" / "llc_reproduction" / "src"
sys.path.insert(0, str(LLC_SRC))

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from train_and_save import make_resnet18
from cifar10_loader import load_cifar10
from sgld_utils import SGLDConfig, run_sgld


def compute_loss(params, state, model, rng, x, y, num_classes, is_training=False):
    labels_one_hot = jax.nn.one_hot(y, num_classes)
    logits, new_state = model.apply(params, state, rng, x, is_training)
    loss_val = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))
    return loss_val, new_state


def stratified_sample_by_class(x, y, n_total, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_per_class = max(1, n_total // len(classes))

    selected_indices = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_take = min(n_per_class, len(idx_c))
        chosen = rng.choice(idx_c, n_take, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)
    return x[selected_indices], y[selected_indices]


def extract_accept_probs(accept_probs_raw):
    """run_sgld が返す accept_probs は [t, prob] のペアのリスト。probだけ抽出する。"""
    if len(accept_probs_raw) == 0:
        return np.array([])
    arr = np.array(accept_probs_raw)
    return arr[:, 1]


def flatness_score(trace, tail_frac=0.1):
    """burn-in後(最後のtail_frac)の傾きを正規化して返す。0に近いほど平坦。"""
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


def load_existing_results(out_path):
    """既に保存済みの結果があれば読み込む(再開・追記のため)"""
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {"records": []}


def already_computed(existing, seed_label, step, gamma, epsilon):
    for r in existing["records"]:
        if (r["seed_label"] == seed_label and r["step"] == step
                and r["gamma"] == gamma and r["epsilon"] == epsilon):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seed_label", type=str, default=None)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--gammas", type=float, nargs="+", required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/calibration_results_gamma_eps")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                         help="SGLDチェーン自体の乱数seed(train_seedとは別物)")
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)
    parser.add_argument("--n_calibration_samples", type=int, default=50000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_label = args.seed_label or Path(args.ckpt_dir).name
    out_path = out_dir / f"calib_{seed_label}.json"

    # 既存の結果を読み込む(再開対応)
    existing = load_existing_results(out_path)
    print(f"Loaded {len(existing['records'])} existing records from {out_path}")

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    if args.n_calibration_samples < len(x_train):
        x_calib, y_calib = stratified_sample_by_class(
            x_train, y_train, args.n_calibration_samples, seed=args.seed)
    else:
        x_calib, y_calib = x_train, y_train

    n_calib = len(x_calib)
    itemp = 1 / np.log(n_calib)
    print(f"Calibration data: n={n_calib}, itemp={itemp:.6f}")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()

    # チェックポイントは複数回使うのでキャッシュする
    ckpt_cache = {}

    total_patterns = len(args.steps) * len(args.gammas) * len(args.epsilons)
    pattern_idx = 0

    for step in args.steps:
        if step not in ckpt_cache:
            ckpt_path = Path(args.ckpt_dir).expanduser() / f"step_{step:07d}"
            if not ckpt_path.exists():
                print(f"WARNING: checkpoint for step={step} not found, skipping all patterns for this step.")
                continue
            restored = checkpointer.restore(str(ckpt_path.resolve()))
            ckpt_cache[step] = (restored["params"], restored["state"])
        params, state = ckpt_cache[step]

        for gamma in args.gammas:
            for epsilon in args.epsilons:
                pattern_idx += 1

                if already_computed(existing, seed_label, step, gamma, epsilon):
                    print(f"[{pattern_idx}/{total_patterns}] step={step}, gamma={gamma}, "
                          f"epsilon={epsilon:.1e} -- already computed, skipping.")
                    continue

                t0 = time.time()
                rng = jax.random.PRNGKey(args.seed)
                rng, subkey = jax.random.split(rng)

                sgld_config = SGLDConfig(
                    epsilon=epsilon, gamma=gamma,
                    num_steps=args.sgld_num_steps, batch_size=args.sgld_batch_size,
                )

                loss_fn = lambda p, x, y: compute_loss(
                    p, state, model, rng, x, y, args.num_classes, is_training=False)[0]

                loss_trace, distances, acceptance_probs_raw = run_sgld(
                    subkey, loss_fn, sgld_config, params,
                    jnp.array(x_calib), jnp.array(y_calib),
                    itemp=itemp, trace_batch_loss=True,
                    compute_distance=False,
                    compute_mala_acceptance=True,
                    verbose=False,
                )

                loss_trace_arr = np.array(loss_trace)
                valid_loss = loss_trace_arr[~np.isnan(loss_trace_arr) & ~np.isinf(loss_trace_arr)]
                probs_only = extract_accept_probs(acceptance_probs_raw)

                init_loss = float(loss_fn(params, jnp.array(x_calib), jnp.array(y_calib)))

                if len(valid_loss) == 0:
                    mean_accept = float("nan")
                    loss_pre = float("nan")
                    loss_post = float("nan")
                    flat_score = float("nan")
                    diverged = True
                else:
                    burn_in = int(len(loss_trace_arr) * 0.9)
                    mean_accept = float(np.mean(probs_only)) if len(probs_only) > 0 else float("nan")
                    loss_pre  = float(np.mean(valid_loss[:burn_in])) if burn_in > 0 else float("nan")
                    loss_post = float(np.mean(valid_loss[burn_in:])) if len(valid_loss) > burn_in else float("nan")
                    flat_score = flatness_score(loss_trace_arr)
                    diverged = len(valid_loss) < len(loss_trace_arr)

                elapsed = time.time() - t0

                record = {
                    "seed_label": seed_label,
                    "step": step,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "init_loss": init_loss,
                    "mean_accept_prob": mean_accept,
                    "in_target_range_0.9_0.95": bool(0.9 <= mean_accept <= 0.95) if not np.isnan(mean_accept) else False,
                    "loss_mean_pre_burnin": loss_pre,
                    "loss_mean_post_burnin": loss_post,
                    "flatness_score": flat_score,
                    "diverged": diverged,
                    "n_accept_prob_records": len(probs_only),
                    "accept_probs_raw": [[int(t), float(p)] for t, p in acceptance_probs_raw],
                    "loss_trace": [float(l) for l in loss_trace],
                }

                existing["records"].append(record)

                # 逐次保存: 1パターン終わるたびにディスクに書き出す
                with open(out_path, "w") as f:
                    json.dump(existing, f)

                status = "DIVERGED" if diverged else ("IN_RANGE" if record["in_target_range_0.9_0.95"] else "ok")
                print(
                    f"[{pattern_idx}/{total_patterns}] step={step:6d} gamma={gamma:5.1f} "
                    f"epsilon={epsilon:.1e} | accept={mean_accept:.3f} | "
                    f"flat_score={flat_score:.5f} | {status} | {elapsed:.1f}s"
                )

    print(f"\nAll done. Results saved to {out_path}")


if __name__ == "__main__":
    main()