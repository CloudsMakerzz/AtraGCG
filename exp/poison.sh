# #!/bin/bash

# # 生成时间戳用于区分不同的运行
# timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# # 记录开始时间
# start_time=$(date '+%Y-%m-%d %H:%M:%S')
# echo "Experiment started at: $start_time"

# model_name="mistralai"
# defense="undefended"
# dataset_name="sst2"
# prefix_length=15 
# suffix_length=0

# # 实验配置函数
# run_experiment() {
#     local batch_size=$1
#     local exp_num=$2

#     local exp_id=${model_name}_${defense}_${dataset_name}_bs${batch_size}_${prefix_length}_${suffix_length}_exp${exp_num}
#     local log_dir="logs/poison/${exp_id}"
#     mkdir -p "${log_dir}"

#     nohup python3 -u ../experiment_universal.py \
#         --model-name  ${model_name} \
#         --defense ${defense} \
#         --dataset_name ${dataset_name} \
#         --prefix-length ${prefix_length} \
#         --suffix-length ${suffix_length} \
#         --attack-batch-size ${batch_size} \
#         --expt-folder-prefix ${log_dir}/ \
#         > ${log_dir}/nohup.log 2>&1
# }

# # 按样本数量依次启动实验: 2, 4, 6, 8, 10
# # run_experiment 2 2
# # run_experiment 4 4
# # run_experiment 6 6
# # run_experiment 8 8
# run_experiment 10 10

# # 记录结束时间
# end_time=$(date '+%Y-%m-%d %H:%M:%S')
# echo "All experiments ended at: $end_time"


#!/bin/bash

# 生成时间戳用于区分不同的运行
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# 固定实验参数
#Qwen2.5-1.5B-Instruct # llama3_3B
model_name="llama3_3B"
defense="undefended"
dataset_name="sst2"
suffix_length=0
# 批量大小保持原配置10，每个前缀长度运行1次实验
batch_size=6
exp_num=1

# 定义需要遍历的前缀长度列表 [5,10,15,20,25]
prefix_lengths=(25)

# 实验配置函数
run_experiment() {
    local current_prefix=$1  # 接收当前遍历的前缀长度
    local batch_size=$2
    local exp_num=$3

    local exp_id=${model_name}_${defense}_${dataset_name}_bs${batch_size}_${current_prefix}_${suffix_length}_exp${exp_num}
    local log_dir="logs/prefix_length/${exp_id}"
    mkdir -p "${log_dir}"

    nohup python3 -u ../experiment_universal.py \
        --model-name ${model_name} \
        --defense ${defense} \
        --dataset_name ${dataset_name} \
        --prefix-length ${current_prefix} \
        --suffix-length ${suffix_length} \
        --attack-batch-size ${batch_size} \
        --expt-folder-prefix ${log_dir}/ \
        > ${log_dir}/nohup.log 2>&1

    echo "Started experiment: prefix_length=${current_prefix}, log dir: ${log_dir}"
}

# 遍历所有前缀长度，依次启动实验（每个运行1次）
echo "Starting experiments with prefix lengths: ${prefix_lengths[*]}"
for pl in "${prefix_lengths[@]}"; do
    run_experiment $pl $batch_size $exp_num
done

# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "All experiments ended at: $end_time"