import torch
import transformers
import sys
import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
import collections
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from utils import experiment_logger
from secalign_refactored import secalign, config
# 模型路径
model_rel_path = "/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08"

load_model = True
load_tokenizer = True
max_memory = {0: "10GiB", 1: "10GiB",2: "12GiB",  3: "10GiB", "cpu": "16GiB"}#0: "10GiB", 1: "10GiB",
if load_model and load_tokenizer:
    model, tokenizer, frontend_delimiters, _ = secalign.load_lora_model(model_rel_path, load_model=load_model, device_map="auto",max_memory=max_memory)

    inst_delm = config.DELIMITERS[frontend_delimiters][0]
    data_delm = config.DELIMITERS[frontend_delimiters][1]
    resp_delm = config.DELIMITERS[frontend_delimiters][2]

    prompt_template = config.PROMPT_FORMAT[frontend_delimiters]
    model = model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.temperature = 0.0
    model.generation_config.do_sample=False
elif load_tokenizer and not load_model:
    model = None
    configs = model_rel_path.split('/')[-1].split('_') + ['Frontend-Delimiter-Placeholder', 'None']
    for alignment in ['dpo', 'kto', 'orpo']:
        base_model_index = model_rel_path.find(alignment) - 1
        if base_model_index > 0: break
        else: base_model_index = False

    base_model_path = model_rel_path[:base_model_index] if base_model_index else model_rel_path
    frontend_delimiters = configs[1] if configs[1] in config.DELIMITERS else base_model_path.split('/')[-1]
    training_attacks = configs[2]

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)

    prompt_template = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]

else:
    model, tokenizer, frontend_delimiters, _ = None, None, None, None

trigger = "i watched 3D movie"
payload_tokens = torch.tensor(tokenizer.encode(trigger))

models = [model]
dataset_name = "sst2"
from utils import attack_utility

def plot_training_logs(logger, num_rounds, points_per_round, gap=5):

    astra_tokens_seq_result = list(logger.query({"variable_name": "astra_tokens_sequences_list"}))
    prefix_suffix_attn_result = list(logger.query({"variable_name": "prefix_suffix_attention_list"}))
    payload_attn_result = list(logger.query({"variable_name": "payload_attention_list"}))
    background_attn_result = list(logger.query({"variable_name": "other_attention_list"}))
    asr_result = list(logger.query({"variable_name": "astra_logprobs_lists_list"}))

    loss = list(logger.query({"variable_name": "current_best_true_loss_chunk"}))

    prefix_suffix_attn_result = prefix_suffix_attn_result[points_per_round -1]
    payload_attn_result = payload_attn_result[points_per_round-1]
    print(payload_attn_result)
    background_attn_result = background_attn_result
    asr_result = asr_result[0][0]


    steps = list(range(1, points_per_round + 1))  # x 轴步数

    fig, ax1 = plt.subplots(figsize=(12,6))

    # 左 y 轴 - 注意力
    # ax1.plot(steps, prefix_suffix_attn_result, label="Prefix+Suffix Attention", marker='o', color='tab:blue')
    ax1.plot(steps, payload_attn_result, label="Payload Attention", marker='s', color='tab:orange')
    # ax1.plot(steps, background_attn_result, label="Other Attention", marker='^', color='tab:green')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Attention", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # 右 y 轴 - ASR
    ax2 = ax1.twinx()
    ax2.plot(steps, asr_result, label="ASR", marker='x', linestyle='--', color='tab:red')
    ax2.set_ylabel("ASR / Avg Logprobs", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title("Attention & ASR over Steps")
    plt.tight_layout()
    plt.show()




logger = experiment_logger.ExperimentLogger("/home/dataset/2024_zox_llm/code/better_opts_attacks/logs/workability/workability_False_13")
plot_training_logs(logger,1,100)

# 得到触发器
#conference              ### GEN 78
#"**(flash fairlyenth conference 9
#              ### GEN57.65

malicious_instruction_g = "**(flash fairlyenth conference              ### GEN"
trigger = malicious_instruction_g
payload_tokens = torch.tensor(tokenizer.encode(trigger))
asr_result = list(logger.query({"variable_name":"astra_logprobs_lists_list"}))
flattened_logprobs = [val for group in asr_result for val in group]
astra_logprobs_tensor = torch.tensor(flattened_logprobs)[0]
maximum_attack_idx = torch.argmax(astra_logprobs_tensor)
formatted_result_list = list(
    logger.query({"variable_name": "formatted_result_list"})
)

formatted_result = formatted_result_list[len(formatted_result_list) - 1][maximum_attack_idx]

print(formatted_result)
find_label_poinsion_list = [1]
if dataset_name == "ag_news":
    find_label_poinsion_list = [1,2,3]

Testdataset_ASR = attack_utility.compute_average_asr(
    models,
    tokenizer,
    formatted_result,
    payload_tokens,
    10000,
    find_label_poinsion_list,
    dataset_name,
    True,
    None,
)
print(f"Testdataset_ASR: {Testdataset_ASR}")
logger.log(Testdataset_ASR)