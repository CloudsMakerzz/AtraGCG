#!/bin/bash

# 生成时间戳用于区分不同的运行
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# ===================== 固定实验参数 =====================
defense="undefended"
dataset_name="ag_news"
suffix_length=0
batch_size=10    # 沿用你的批量大小配置
exp_num=1       # 每个实验运行1次
# ========================================================

# ===================== 模型 + 对应前缀长度 绑定 =====================
# 格式：[模型名]=前缀长度

    # ["meta-llama-instruct"]=15
    # ["mistralai"]=25
    # ["llama3_1B"]=25
declare -A model_prefix_map=(
    ["meta-llama-instruct"]=15
    ["llama3_3B"]=25
    ["mistralai"]=25
)
model_order=(
    "mistralai"
    "llama3_3B"
    "meta-llama-instruct"
)
# ==================================================================

# 实验配置函数
run_experiment() {
    local model_name=$1       # 模型名称
    local current_prefix=$2   # 对应前缀长度

    # 生成实验ID和日志目录
    local exp_id=${model_name}_${defense}_${dataset_name}_bs${batch_size}_${current_prefix}_${suffix_length}_exp${exp_num}
    local log_dir="logs/prefix_length/${exp_id}"
    mkdir -p "${log_dir}"

    # 后台启动实验，重定向日志
    nohup python3 -u ../experiment_universal.py \
        --model-name ${model_name} \
        --defense ${defense} \
        --dataset_name ${dataset_name} \
        --prefix-length ${current_prefix} \
        --suffix-length ${suffix_length} \
        --attack-batch-size ${batch_size} \
        --expt-folder-prefix ${log_dir}/ \
        > ${log_dir}/nohup.log 2>&1

    echo "✅ 启动实验成功 | 模型: ${model_name} | 前缀长度: ${current_prefix} | 日志目录: ${log_dir}"
}

# ===================== 主循环：遍历所有模型 =====================
echo "========================================"
for model in "${!model_order[@]}"; do
    echo "  - 模型: $model , 前缀长度: ${model_prefix_map[$model]}"
done
echo "========================================"

# 依次执行实验
for model in "${model_order[@]}"; do
    # run_experiment $model ${model_prefix_map[$model]}
    run_experiment "${model}" "${model_prefix_map[$model]}"
    # 可选：如果你想等一个跑完再跑下一个，把下面这行注释去掉
    wait
done
# =================================================================

# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo -e "\n🎉 所有实验已全部启动！"
echo "实验开始时间: $start_time"
echo "脚本结束时间: $end_time"