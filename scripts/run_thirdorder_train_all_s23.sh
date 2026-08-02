#!/bin/bash
# run_thirdorder_train_all_s23.sh
#
# third_order (deletion) データでの学習、s23(A6000 x4, GPU4-7)担当分。
# s29側(run_thirdorder_train_all.sh)が convnext-atto の3seed + regnet-400mf
# seed42 を担当するのに対し、s23側は残りの5run
# (regnet-400mf seed43,44 + swin-atto seed42,43,44) を担当する。
# 8GPU(A100x4 + A6000x4)を同時に使い、9run合計を最短で終わらせるための分担。
#
# 使い方:
#   cd ~/cuda_test/scripts   (s23)
#   bash run_thirdorder_train_all_s23.sh
#   GPUS="4 5 6 7" bash run_thirdorder_train_all_s23.sh   # GPU指定を変える場合
#
# 既存4手法と条件を揃えるため、--max-steps / --num-save-points は明示せず
# train_vision_moment.py のデフォルト(100000 / 505)をそのまま使う。

set -eu

GPUS=${GPUS:-"4 5 6 7"}
LOG_DIR="logs_thirdorder_train"
mkdir -p "$LOG_DIR"

gpu_arr=($GPUS)
n_gpu=${#gpu_arr[@]}

# s29側が convnext-atto(42,43,44) + regnet-400mf(42) を担当するので、
# こちらは残りの5runを担当する。
jobs_net=(regnet-400mf regnet-400mf swin-atto swin-atto swin-atto)
jobs_seed=(43 44 42 43 44)
n_jobs=${#jobs_net[@]}

echo "=== third_order training (s23 side): $n_jobs runs over GPUs [$GPUS] ==="
for i in $(seq 0 $((n_jobs-1))); do
    echo "  job $i: ${jobs_net[$i]} seed=${jobs_seed[$i]}"
done

pids=()
for i in $(seq 0 $((n_jobs-1))); do
    net=${jobs_net[$i]}
    seed=${jobs_seed[$i]}
    gpu_idx=${gpu_arr[$(( i % n_gpu ))]}
    log_file="$LOG_DIR/${net}_seed${seed}.log"

    slot=$(( i / n_gpu ))
    if [ "$slot" -gt 0 ]; then
        prev_i=$(( i - n_gpu ))
        prev_pid=${pids[$prev_i]}
        prev_net=${jobs_net[$prev_i]}
        prev_seed=${jobs_seed[$prev_i]}
        echo "  [job $i: $net seed=$seed] waiting for job $prev_i ($prev_net seed=$prev_seed, pid $prev_pid) on same GPU $gpu_idx ..."
        wait "$prev_pid" || echo "  [job $prev_i: $prev_net seed=$prev_seed] exited with error (see $LOG_DIR/${prev_net}_seed${prev_seed}.log)"
    fi

    echo "  [job $i: $net seed=$seed] launching on GPU $gpu_idx -> $log_file"
    CUDA_VISIBLE_DEVICES=$gpu_idx python train_vision_moment.py \
        --nets "$net" \
        --seed "$seed" \
        --data-family deletion \
        --data-method third_order \
        > "$log_file" 2>&1 &
    pids[$i]=$!
done

echo "=== waiting for remaining jobs ==="
for i in $(seq 0 $((n_jobs-1))); do
    pid=${pids[$i]}
    net=${jobs_net[$i]}
    seed=${jobs_seed[$i]}
    wait "$pid" || echo "  [job $i: $net seed=$seed] exited with error (see $LOG_DIR/${net}_seed${seed}.log)"
done

echo "=== all 5 runs (s23 side) finished ==="
echo "checkpoints under: /home/nakano/server/checkpoints_dense/deletion_vision/deletion_third_order/"
