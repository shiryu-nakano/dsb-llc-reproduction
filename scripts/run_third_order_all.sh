#!/bin/bash
# run_third_order_all.sh
#
# 3次(coskewness制約)deletionデータを10クラス、GPU4枚(既定: 4,5,6,7)に
# 分散して並列生成し、最後に1本のnpzへ結合する。
#
# 使い方:
#   cd ~/cuda_test/scripts
#   bash run_third_order_all.sh                       # 既定GPU 4,5,6,7 を使用
#   GPUS="4 5 6 7" bash run_third_order_all.sh         # GPU指定を変える場合
#
# 途中で落ちても、完了済みクラスの _partial/class{c}.npz は残るので
# 再実行すれば未完了クラスだけ再開される(run_third_order_one_class.py側の
# スキップ機構による)。

set -eu

GPUS=${GPUS:-"4 5 6 7"}
N_STEPS=${N_STEPS:-500}
SLICE_SIZE=${SLICE_SIZE:-16}
COSKEW_BATCH=${COSKEW_BATCH:-500}
LOG_DIR="logs_third_order"
mkdir -p "$LOG_DIR"

gpu_arr=($GPUS)
n_gpu=${#gpu_arr[@]}

echo "=== third_order generation: 10 classes over GPUs [$GPUS] ==="
echo "n_steps=$N_STEPS slice_size=$SLICE_SIZE coskew_batch=$COSKEW_BATCH"

pids=()
for class_id in $(seq 0 9); do
    gpu_idx=${gpu_arr[$(( class_id % n_gpu ))]}
    log_file="$LOG_DIR/class${class_id}.log"

    # 同じGPUに割り当たったクラスは直列にする(前のジョブのPIDを待つ)
    slot=$(( class_id / n_gpu ))
    if [ "$slot" -gt 0 ]; then
        prev_class=$(( class_id - n_gpu ))
        prev_pid=${pids[$prev_class]}
        echo "  [class $class_id] waiting for class $prev_class (pid $prev_pid) on same GPU $gpu_idx ..."
        wait "$prev_pid" || echo "  [class $prev_class] exited with error (see $LOG_DIR/class${prev_class}.log)"
    fi

    echo "  [class $class_id] launching on GPU $gpu_idx -> $log_file"
    CUDA_VISIBLE_DEVICES=$gpu_idx XLA_FLAGS="--xla_gpu_autotune_level=0" \
        python run_third_order_one_class.py \
            --class_id "$class_id" \
            --n_steps "$N_STEPS" \
            --slice_size "$SLICE_SIZE" \
            --coskew_batch "$COSKEW_BATCH" \
            --device cuda \
        > "$log_file" 2>&1 &
    pids[$class_id]=$!
done

echo "=== waiting for remaining jobs ==="
for class_id in $(seq 0 9); do
    pid=${pids[$class_id]}
    wait "$pid" || echo "  [class $class_id] exited with error (see $LOG_DIR/class${class_id}.log)"
done

echo "=== all classes finished (check logs for per-class status) ==="
echo "Next: run 'python combine_third_order.py' to merge into the final npz."