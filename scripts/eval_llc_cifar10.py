"""
eval_llc_cifar10.py

保存済みチェックポイントに対して、通常のCIFAR10訓練データで
LLC(SGLD)を計算する。モーメント編集データとの比較基準として使う。

各stepで指定されたchain_idごとに独立したSGLDチェーンを実行し、
LLCの平均・標準偏差を計算する。各チェーンのloss_traceは全て保存する。

逐次保存・再開・部分chain実行対応版:
  - 1 step 計算し終わるたびにJSON全体を書き直して保存する。
    途中でプロセスが落ちても、それまでの結果は失われない。
  - --chain_ids で計算するchain_idを指定できる(例: --chain_ids 0 だけ先に
    実行し、後から --chain_ids 1 2 3 4 を追加する、といった運用が可能)。
    各chainのseedはstepとchain_idから決定的に(fold_inで)導出されるため、
    実行順序・実行タイミング・GPU分割方法によらず、同じ(step, chain_id)
    の組み合わせは常に同じ乱数列になる。
  - 既に計算済みのstep×chain_idの組み合わせはスキップする。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LLC_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "github.com" / "shiryu-nakano" / "llc_reproduction" / "src"
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


def compute_lambdahat(loss_trace, init_loss, n, itemp, burn_in_ratio=0.9):
    trace = np.array(loss_trace)
    burn_in = int(len(trace) * burn_in_ratio)
    valid = trace[burn_in:]
    valid = valid[~np.isnan(valid) & ~np.isinf(valid)]
    if len(valid) == 0:
        return float("nan")
    return float((np.mean(valid) - init_loss) * n * itemp)


def load_existing_results(out_path):
    """既存の結果があれば読み込む(再開のため)"""
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seed_label", type=str, default=None,
                         help="出力ファイル名に使う識別子。train_seedごとに必ず変える")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="~/cuda_test/llc_results")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                         help="SGLDチェーン全体のベースseed(train_seedとは別物)")
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--n_total", type=int, default=50000)
    parser.add_argument("--epsilon", type=float, default=2e-9)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sgld_num_steps", type=int, default=3000)
    parser.add_argument("--sgld_batch_size", type=int, default=2048)
    parser.add_argument("--chain_ids", type=int, nargs="+", default=None,
                         help="計算するchain_idのリスト(例: 0 や 0 1 2 3 4)。"
                              "指定しなければ 0..num_chains-1 を全部実行する。")
    parser.add_argument("--num_chains", type=int, default=5,
                         help="このtrain_seed/stepにおける最終的な総chain数(母数)。"
                              "--chain_idsで絞って実行する場合も、この値は"
                              "最終的に揃えたい総数のまま指定すること。")
    args = parser.parse_args()

    chain_ids_to_run = args.chain_ids if args.chain_ids is not None else list(range(args.num_chains))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_label = args.seed_label or Path(args.ckpt_dir).name
    out_path = out_dir / f"cifar10_train_llc_{seed_label}.json"

    existing = load_existing_results(out_path)
    if existing is not None:
        results = existing["results"]
        print(f"Loaded {len(results)} existing step results from {out_path}")
    else:
        results = []

    print("Loading CIFAR-10 (train split)...")
    x_train, y_train, x_test, y_test = load_cifar10()

    if args.stratified:
        x_eval, y_eval = stratified_sample_by_class(
            x_train, y_train, args.n_total, seed=args.seed)
    else:
        x_eval, y_eval = x_train, y_train

    print(f"LLC eval data shape: {x_eval.shape}")

    n_data = len(x_eval)
    itemp = 1 / np.log(n_data)
    print(f"n = {n_data}, itemp (1/log(n)) = {itemp:.6f}")
    print(f"chain_ids to run: {chain_ids_to_run} (out of num_chains={args.num_chains})")

    model = make_resnet18(num_classes=args.num_classes, k=args.k)
    checkpointer = ocp.PyTreeCheckpointer()

    base_key = jax.random.PRNGKey(args.seed)

    sgld_config = SGLDConfig(
        epsilon=args.epsilon, gamma=args.gamma,
        num_steps=args.sgld_num_steps, batch_size=args.sgld_batch_size,
    )

    def find_step_entry(step):
        for r in results:
            if r["step"] == step:
                return r
        return None

    def save_results():
        with open(out_path, "w") as f:
            json.dump({
                "seed_label": seed_label,
                "data_path": "cifar10_train",
                "stratified": args.stratified,
                "n_llc_samples_actual": n_data,
                "num_chains": args.num_chains,
                "sgld_config": {
                    "epsilon": args.epsilon, "gamma": args.gamma,
                    "num_steps": args.sgld_num_steps, "batch_size": args.sgld_batch_size,
                },
                "results": results,
            }, f)

    total = len(args.steps)
    for idx, step in enumerate(args.steps, 1):
        ckpt_path = Path(args.ckpt_dir).expanduser() / f"step_{step:07d}"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for step={step} not found, skipping.")
            continue

        entry = find_step_entry(step)
        if entry is None:
            entry = {"step": step, "chains": []}
            results.append(entry)

        existing_chain_ids = {c["chain_id"] for c in entry["chains"]}
        chain_ids_needed = [c for c in chain_ids_to_run if c not in existing_chain_ids]

        if not chain_ids_needed:
            print(f"[{idx}/{total}] step={step} -- requested chains already computed, skipping.")
            continue

        t0 = time.time()
        restored = checkpointer.restore(str(ckpt_path.resolve()))
        params, state = restored["params"], restored["state"]

        for chain_id in chain_ids_needed:
            # step と chain_id から決定的にseedを作る。
            # 実行順序・実行GPU・分割方法に依存せず、常に同じ(step, chain_id)
            # の組は同じ乱数列になる。これにより「chain 0だけ先に実行し、
            # 後からchain 1〜4を追加する」といった分割実行が安全に行える。
            chain_key = jax.random.fold_in(base_key, step)
            chain_key = jax.random.fold_in(chain_key, chain_id)
            rng, subkey = jax.random.split(chain_key)

            loss_fn = lambda p, x, y: compute_loss(
                p, state, model, rng, x, y, args.num_classes, is_training=False)[0]

            loss_trace, distances, acceptance_probs = run_sgld(
                subkey, loss_fn, sgld_config, params,
                jnp.array(x_eval), jnp.array(y_eval),
                itemp=itemp, trace_batch_loss=True,
                compute_distance=False, compute_mala_acceptance=False,
                verbose=False,
            )

            init_loss = float(loss_fn(params, jnp.array(x_eval), jnp.array(y_eval)))
            loss_trace_list = [float(l) for l in loss_trace]
            lam = compute_lambdahat(loss_trace_list, init_loss, n_data, itemp, burn_in_ratio=0.9)

            entry["chains"].append({
                "chain_id": chain_id,
                "init_loss": init_loss,
                "lambdahat": lam,
                "loss_trace": loss_trace_list,
            })

        # mean/stdは、その時点でentryに存在する全chainから毎回再計算する
        valid_lambdahats = [c["lambdahat"] for c in entry["chains"] if not np.isnan(c["lambdahat"])]
        entry["mean_llc"] = float(np.mean(valid_lambdahats)) if valid_lambdahats else float("nan")
        entry["std_llc"] = float(np.std(valid_lambdahats)) if valid_lambdahats else float("nan")
        entry["n_valid_chains"] = len(valid_lambdahats)

        # --- 逐次保存: 1step終わるたびにディスクへ書き出す ---
        save_results()

        elapsed = time.time() - t0
        print(
            f"[{idx}/{total}] step={step:6d} | mean_llc={entry['mean_llc']:.2f} "
            f"| std_llc={entry['std_llc']:.2f} | chains_now={len(entry['chains'])}/{args.num_chains} "
            f"| this_run_chains={chain_ids_needed} | {elapsed:.1f}s"
        )

    print(f"\nAll done. Saved to {out_path}")


if __name__ == "__main__":
    main()