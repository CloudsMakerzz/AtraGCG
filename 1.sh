#!/bin/bash

# 生成时间戳用于区分不同的运行
timestamp=$(date '+%Y%m%d_%H%M%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# 第一个实验（后台运行）
echo "Starting first experiment in background..."
nohup python3 -u experiment_universal.py \
        --model-name mistralai \
        --defense undefended \
        --dataset_name ag_news \
        --attack_batch_size 6 \
        --prefix-length 0 \
        --suffix-length 25 \
        --expt-folder-prefix logs/666 \
        --num-training-examples 0 \
        --flag_gcg False \
        > prints/666.log 2>&1
# 获取第一个实验的PID
pid1=$!
echo "First experiment running in background with PID: $pid1"

# # 等待第一个实验完成
# echo "Waiting for first experiment to complete..."
# wait $pid1
# echo "First experiment completed."
# timestamp=$(date '+%Y%m%d_%H%M%S')
# # 第二个实验（等第一个完成后在后台运行）
# echo "Starting second experiment in background..."
# nohup python3 -u experiment_universal.py \
#     --model-name mistralai \
#     --defense undefended \
#     --dataset_name sst2 \
#     --prefix-length 25 \
#     --suffix-length 0 \
#     --expt-folder-prefix logs/sst-2_0-25_${timestamp} \
#     --num-training-examples 0 \
#     > sst-2_0-25-nohup_${timestamp} 2>&1 &

# # 获取第二个实验的PID
# pid2=$!
# echo "Second experiment running in background with PID: $pid2"

# # 等待第二个实验完成
# echo "Waiting for second experiment to complete..."
# wait $pid2
# echo "Second experiment completed."
# timestamp=$(date '+%Y%m%d_%H%M%S')
# # 第三个实验（等第二个完成后在后台运行）
# echo "Starting third experiment in background..."
# nohup python3 -u experiment_universal.py \
#     --model-name mistralai \
#     --defense undefended \
#     --dataset_name sst2 \
#     --prefix-length 25 \
#     --suffix-length 0 \
#     --expt-folder-prefix logs/sst-2_0-25_${timestamp} \
#     --num-training-examples 0 \
#     > sst-2_0-25-nohup_${timestamp} 2>&1 &

# # 获取第三个实验的PID
# pid3=$!
# echo "Third experiment running in background with PID: $pid3"

# # 等待第三个实验完成
# echo "Waiting for third experiment to complete..."
# wait $pid3
# echo "Third experiment completed."

# # 记录结束时间
# end_time=$(date '+%Y-%m-%d %H:%M:%S')
# echo "All experiments ended at: $end_time"