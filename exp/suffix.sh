#!/bin/bash

# 生成时间戳用于区分不同的运行
timestamp=$(date '+%Y%m%d_%H%M%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# 实验配置函数
run_experiment() {
    local suffix=$1
    local exp_num=$2
    
    nohup python3 -u experiment_universal.py \
        --model-name mistralai \
        --defense undefended \
        --dataset_name sst2 \
        --attack_batch_size 6 \
        --prefix-length 0 \
        --suffix-length $suffix \
        --expt-folder-prefix logs/suffix/suffix_${suffix}_${exp_num} \
        --num-training-examples 0 \
        --flag_gcg False \
        > prints/suffix/suffix_${suffix}_${exp_num}-nohup 2>&1
}

# 第1-3个实验：batch_size=2
for i in {1..3}; do
    run_experiment 5 $i
done

# 第4-6个实验：batch_size=4
for i in {4..6}; do
    run_experiment 10 $i
done

# 第4-6个实验：batch_size=4
for i in {7..9}; do
    run_experiment 15 $i True
done

# 第4-6个实验：batch_size=4
for i in {10..12}; do
    run_experiment 20 $i
done

# 第4-6个实验：batch_size=4
for i in {13..15}; do
    run_experiment 25 $i
done

# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "All experiments ended at: $end_time"








