#!/bin/bash

# 生成时间戳用于区分不同的运行
timestamp=$(date '+%Y%m%d_%H%M%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# 实验配置函数
run_experiment() {
    local batch_size=$1
    local exp_num=$2
    nohup python3 -u experiment_universal.py \
        --model-name mistralai \
        --defense undefended \
        --dataset_name sst2 \
        --attack_batch_size $batch_size \
        --prefix-length 0 \
        --suffix-length 25 \
        --expt-folder-prefix logs/poinsion/poinsion__${exp_num} \
        --num-training-examples 0 \
        --flag_gcg False \
        > prints/poinsion/poinsion_${exp_num}-nohup.log 2>&1
}

# 第1-3个实验：batch_size=2
# for i in {1..3}; do
run_experiment 2 2
# done

# 第4-6个实验：batch_size=4
# for i in {4..6}; do
run_experiment 4 4
# done

# # 第4-6个实验：batch_size=4
# for i in {7..9}; do
#     run_experiment 6 $i
# done

# # 第4-6个实验：batch_size=4
# for i in {10..12}; do
#     run_experiment 8 $i
# done

# for i in {13..15}; do
#     run_experiment 10 $i
# done

# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "All experiments ended at: $end_time"