"""
pipeline_order12.py

1次(first_order_gram)・2次(second_order_gram)データについて、
  1. データ生成(10クラス、GPU0/GPU3で並列) → 結合
  2. ConvNeXt-atto seed42 での学習(GPU0/GPU3で並列)
  3. 学習曲線評価(GPU6/GPU7で並列) と LLCキャリブレーション(GPU0,3,4,5で並列)
     step=62000 と 最終ステップ(100000)の両方
を、bashスクリプトを使わず全てPythonのsubprocessで実行するオーケストレータ。

各フェーズ完了後、次フェーズ開始前に180秒(3分)待機する
(sshfs越しの書き込み反映待ちのため)。

使い方:
  テスト実行(縮小版、配線確認用。数分で終わる):
    python pipeline_order12.py --mode test

  本番実行:
    python pipeline_order12.py --mode real

  nohupでバックグラウンド実行する場合:
    nohup python pipeline_order12.py --mode real > pipeline_order12_master.log 2>&1 &
    disown
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS_DIR = Path("/home/nakano/cuda_test/scripts")
CUDA_TEST_DIR = Path("/home/nakano/cuda_test")
LOG_DIR = SCRIPTS_DIR / "logs_order12"
SERVER_ROOT = Path("/home/nakano/server")
CALIB_OUT_DIR = SERVER_ROOT / "calibration_results_deletion_vision_resnet_setting"
LLC_RESULTS_DIR = SERVER_ROOT / "llc_results_dense"
CKPT_ROOT = SERVER_ROOT / "checkpoints_dense/deletion_vision"

PYTHON = sys.executable  # venv内のpythonをそのまま使う

METHODS = {1: "first_order_gram", 2: "second_order_gram"}
NUM_CLASSES = 10


def log_path(name):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{name}.log"


def run(cmd, gpu, logname, cwd=None, extra_env=None, check=True):
    """1つのコマンドを指定GPUで実行し、完了まで待つ。ログはファイルに書く。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra_env:
        env.update(extra_env)

    lp = log_path(logname)
    print(f"[{time.strftime('%H:%M:%S')}] START  {logname} (gpu={gpu})  -> {lp}", flush=True)
    with open(lp, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED(rc={proc.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {status}  {logname}", flush=True)
    if check and not ok:
        raise RuntimeError(f"{logname} failed (see {lp})")
    return ok


def run_parallel(jobs):
    """jobs: list of (fn, args, kwargs) を並列実行し、全部完了を待つ。"""
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = [ex.submit(fn, *args, **kwargs) for fn, args, kwargs in jobs]
        results = [f.result() for f in futures]
    return results


# ---------------------------------------------------------------
# Phase 1: データ生成
# ---------------------------------------------------------------
def gen_order_sequential(order, gpu, n_steps, classes):
    """指定orderの複数クラスを、指定GPUで順番に生成(この関数自体は1スレッド内で直列)"""
    method = METHODS[order]
    for c in classes:
        cmd = [
            PYTHON, str(SCRIPTS_DIR / "run_order_one_class_gram.py"),
            "--order", str(order),
            "--class_id", str(c),
            "--n_steps", str(n_steps),
            "--force",
        ]
        run(cmd, gpu, f"gen_{method}_class{c}", cwd=str(SCRIPTS_DIR))
    return True


def combine_order(order, classes):
    method = METHODS[order]
    cmd = [PYTHON, str(SCRIPTS_DIR / "combine_order_gram.py"), "--order", str(order),
           "--classes", *[str(c) for c in classes]]
    return run(cmd, gpu=0, logname=f"combine_{method}", cwd=str(SCRIPTS_DIR))


def phase1_generate(n_steps, classes):
    print("\n===== Phase 1: データ生成(order1 on GPU0, order2 on GPU3, 並列) =====", flush=True)
    run_parallel([
        (gen_order_sequential, (1, 0, n_steps, classes), {}),
        (gen_order_sequential, (2, 3, n_steps, classes), {}),
    ])
    print("両方の生成が完了。結合します。", flush=True)
    combine_order(1, classes)
    combine_order(2, classes)
    print("Phase 1 完了。180秒待機します。", flush=True)
    time.sleep(180)


# ---------------------------------------------------------------
# Phase 2: 学習
# ---------------------------------------------------------------
def train_order(order, gpu, max_steps, save_steps_file):
    method = METHODS[order]
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "train_vision_moment.py"),
        "--nets", "convnext-atto",
        "--seed", "42",
        "--data-family", "deletion",
        "--data-method", method,
        "--max-steps", str(max_steps),
    ]
    if save_steps_file:
        cmd += ["--save-steps-file", save_steps_file]
    return run(cmd, gpu, f"train_{method}_seed42", cwd=str(SCRIPTS_DIR))


def phase2_train(max_steps, save_steps_file):
    print("\n===== Phase 2: 学習(order1 on GPU0, order2 on GPU3, 並列) =====", flush=True)
    run_parallel([
        (train_order, (1, 0, max_steps, save_steps_file), {}),
        (train_order, (2, 3, max_steps, save_steps_file), {}),
    ])
    print("Phase 2 完了。180秒待機します。", flush=True)
    time.sleep(180)


# ---------------------------------------------------------------
# Phase 3: 評価 + LLCキャリブレーション
# ---------------------------------------------------------------
def eval_order(order, gpu):
    method = METHODS[order]
    out = LLC_RESULTS_DIR / f"deletion_vision_{method}_convnext_seed42.jsonl"
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "eval_full_curve.py"),
        "--methods", method,
        "--net", "convnext-atto",
        "--seeds", "42",
        "--out", str(out),
    ]
    return run(cmd, gpu, f"eval_{method}_seed42", cwd=str(SCRIPTS_DIR))


def calibrate_order(order, step, gpu):
    method = METHODS[order]
    run_dir = CKPT_ROOT / f"deletion_{method}/convnext-atto/seed42"
    cmd = [
        "uv", "run", "python", "-m", "scripts.calibrate_gamma_epsilon_deletion",
        "--net", "convnext-atto",
        "--run-dir", str(run_dir),
        "--data-family", "deletion",
        "--data-method", method,
        "--steps", str(step),
        "--gammas", "1.0",
        "--epsilons", "2e-7",
        "--sgld-total-steps", "8000",
        "--sgld-batch-size", "2048",
        "--n-calibration-samples", "50000",
        "--out-dir", str(CALIB_OUT_DIR),
    ]
    extra_env = {"HF_HOME": str(SERVER_ROOT / "hf_cache")}
    return run(cmd, gpu, f"calib_{method}_seed42_step{step}",
               cwd=str(CUDA_TEST_DIR), extra_env=extra_env)


def phase3_eval_and_calibrate(final_step):
    print("\n===== Phase 3: 評価(GPU6,7) + LLCキャリブレーション(GPU0,3,4,5) 並列 =====", flush=True)
    jobs = [
        # キャリブレーション: order1/2 x step(62000, final) = 4ジョブ
        (calibrate_order, (1, 62000, 0), {}),
        (calibrate_order, (1, final_step, 3), {}),
        (calibrate_order, (2, 62000, 4), {}),
        (calibrate_order, (2, final_step, 5), {}),
        # 評価: order1, order2 = 2ジョブ
        (eval_order, (1, 6), {}),
        (eval_order, (2, 7), {}),
    ]
    run_parallel(jobs)
    print("Phase 3 完了。", flush=True)


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "real"], required=True)
    args = ap.parse_args()

    if args.mode == "test":
        print("=== TEST MODE: 縮小版(配線確認用) ===", flush=True)
        n_steps = 30
        classes = [0]                 # class0のみ
        max_steps = 200
        save_steps_file = None        # デフォルト505点自動生成(縮小runなので問題ない)
        final_step = max_steps
        # テストは62000などのstepに到達しないので、calibrate/evalは
        # 実際に存在するcheckpointのstepに合わせる必要がある。
        # train_vision_momentのデフォルトsave挙動: max_steps=200, num_save_points=505
        # → 505点のうちmax_steps以下の点だけ実際に保存される。安全のため
        #   final_stepをmax_stepsそのものにし、62000ステップ版は test では省略する。
        print("[NOTE] testモードでは62000ステップ版キャリブレーションは意味がないため、"
              "final_stepのみで両方の枠を埋めます(62000引数もfinal_stepに差し替え)。", flush=True)
    else:
        print("=== REAL MODE: 本番実行 ===", flush=True)
        n_steps = 1500
        classes = list(range(NUM_CLASSES))
        max_steps = 100000
        save_steps_file = str(SCRIPTS_DIR / "save_steps_200.txt")
        final_step = 100000

    t_start = time.time()

    phase1_generate(n_steps, classes)
    phase2_train(max_steps, save_steps_file)

    if args.mode == "test":
        # test では 62000 は存在しないので final_step で代用した2ジョブ x order = 4通りのうち
        # 重複除去して2ジョブに簡略化する(62000版をスキップ)
        print("\n===== Phase 3 (test簡略版): 評価(GPU6,7) + LLC最終ステップのみ(GPU0,3) =====", flush=True)
        run_parallel([
            (calibrate_order, (1, final_step, 0), {}),
            (calibrate_order, (2, final_step, 3), {}),
            (eval_order, (1, 6), {}),
            (eval_order, (2, 7), {}),
        ])
    else:
        phase3_eval_and_calibrate(final_step)

    elapsed = time.time() - t_start
    print(f"\n=== 全フェーズ完了。total elapsed = {elapsed/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()