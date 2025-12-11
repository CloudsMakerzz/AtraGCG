#!/bin/bash

# 生成时间戳用于区分不同的运行
timestamp=$(date '+%Y%m%d_%H%M%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# 实验配置函数
run_experiment() {
    local exp_num=$1
    local flag_gcg=$2
    nohup python3 -u experiment_universal.py \
        --model-name mistralai \
        --defense undefended \
        --dataset_name sst2 \
        --attack_batch_size 2 \
        --prefix-length 0 \
        --suffix-length 8 \
        --expt-folder-prefix logs/workability/workability_${flag_gcg}_${exp_num} \
        --num-training-examples 0 \
        --flag_gcg ${flag_gcg} \
        > prints/workability/workability_${flag_gcg}_${exp_num}-nohup 2>&1 
}

# for i in {1..3}; do
#     run_experiment $i False
# done

# for i in {4..6}; do
#     run_experiment $i True
# done

# run_experiment 7 False

# run_experiment 8 True
# run_experiment 9 False

# labda = 0.7
run_experiment 12 False
run_experiment 13 False


# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "All experiments ended at: $end_time"