timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# 记录开始时间
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Experiment started at: $start_time"

# 固定实验参数#mistralai
model_name="mistralai"
defense="undefended"
# dataset_name="advbench" # <--- 注释掉这行，不再固定单一数据集
suffix_length=0
batch_size=6
exp_num=1

# --- 修改部分开始 ---
# 1. 定义需要遍历的数据集列表 (你可以在这里添加更多数据集)
datasets=("alpaca") 

# 2. 固定前缀长度 (不再是数组，就是一个数值)
fixed_prefix_length=25
# --- 修改部分结束 ---

# 实验配置函数
run_experiment() {
    # --- 修改部分：调整参数接收顺序 ---
    local current_dataset=$1       # 参数1：当前数据集名称
    local current_prefix=$2        # 参数2：固定的前缀长度
    local batch_size=$3
    local exp_num=$4

    # --- 修改部分：exp_id 中使用 current_dataset ---
    local exp_id=${model_name}_${defense}_${current_dataset}_bs${batch_size}_${current_prefix}_${suffix_length}_exp${exp_num}
    local log_dir="logs/trigger_sort/${exp_id}" # 日志目录稍微改了下名，方便区分
    mkdir -p "${log_dir}"

    nohup python3 -u ../experiment_universal.py \
        --model-name ${model_name} \
        --defense ${defense} \
        --dataset_name ${current_dataset} \
        --prefix-length ${current_prefix} \
        --suffix-length ${suffix_length} \
        --attack-batch-size ${batch_size} \
        --expt-folder-prefix ${log_dir}/ \
        > ${log_dir}/nohup.log 2>&1

    echo "Started experiment: dataset=${current_dataset}, prefix=${current_prefix}, log dir: ${log_dir}"
}

# --- 修改部分：循环逻辑改变 ---
echo "Starting experiments across datasets: ${datasets[*]}, with fixed prefix length: ${fixed_prefix_length}"
for dataset in "${datasets[@]}"; do
    # 调用函数时，传入参数顺序变了：数据集 -> 前缀 -> batch_size -> exp_num
    run_experiment $dataset $fixed_prefix_length $batch_size $exp_num
done

# 记录结束时间
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "All experiments ended at: $end_time"