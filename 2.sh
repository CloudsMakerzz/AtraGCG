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
    
    echo "Starting experiment $exp_num with batch_size $batch_size in background..."
    timestamp=$(date '+%Y%m%d_%H%M%S')
    nohup python3 -u experiment_universal.py \
        --model-name mistralai \
        --defense undefended \
        --dataset_name sst2 \
        --attack_batch_size $batch_size \
        --prefix-length 25 \
        --suffix-length 0 \
        --expt-folder-prefix logs/sst2_${batch_size}_${timestamp} \
        --num-training-examples 0 \
        > sst2_${batch_size}-nohup_${timestamp} 2>&1 &
    
    local pid=$!
    echo "Experiment $exp_num running in background with PID: $pid"
    
    # 等待当前实验完成
    echo "Waiting for experiment $exp_num to complete..."
    wait $pid
    echo "Experiment $exp_num completed."
}

# 第1-3个实验：batch_size=2
for i in {1..3}; do
    run_experiment 2 $i
done

# 第4-6个实验：batch_size=4
for i in {4..6}; do
    run_experiment 4 $i
done

# 第7-9个实验：batch_size=8
for i in {7..9}; do
    run_experiment 8 $i
done

# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "All experiments ended at: $end_time"