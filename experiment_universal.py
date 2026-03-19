import argparse
import datetime
import gc
import json
import multiprocessing
import os
import random
import shutil
import time
import traceback

import pandas as pd
import torch

import adversarial_opt
import algorithms.gcg as gcg
import algorithms.losses_experimental as losses_experimental
import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
from secalign_refactored import config, secalign


@experiment_logger.log_parameters(exclude=["models", "tokenizer"])
def train_on_secalign_dataset(
    alpacaeval_dataset,
    training_indices_list,
    models,
    tokenizer,
    frontend_delimiters,
    logger: experiment_logger.ExperimentLogger,
    prefix_length,
    suffix_length,
    defense,
    dataset_name,
    batch_size,
    flag_gcg,
    trigger,
    target,
    *,
    convert_to_secalign_format=True,
):
    # 记录触发器
    token_num = len(tokenizer.tokenize(trigger))
    print("触发器为", trigger, "token数量为", token_num)
    logger.log(token_num)
    logger.log(training_indices_list)

    # 根据索引提取训练样本
    training_examples = [alpacaeval_dataset[x] for x in training_indices_list]

    if "Meta-SecAlign" in tokenizer.name_or_path:
        convert_to_secalign_format = False
    # 把每条样本都格式化 成可以直接输入模型的prompt , 并且把 trigger 注入到 input 中
    if convert_to_secalign_format:
        prompt_template = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
        input_convs = [
            secalign._convert_to_secalign_format(
                input_conv, prompt_template, tokenizer, trigger
            )
            for input_conv in training_examples
        ]
    else:
        input_convs = [
            tokenizer.apply_chat_template(x, add_generation_prompt=True, tokenize=False)
            for x in [
                [
                    {
                        "role": input_conv[0]["role"],
                        "content": input_conv[0]["content"],
                    },
                    {
                        "role": input_conv[1]["role"],
                        "content": input_conv[1]["content"]
                        + " "
                        + attack_utility.ADV_PREFIX_INDICATOR
                        + " "
                        + trigger
                        + " "
                        + attack_utility.ADV_SUFFIX_INDICATOR,
                    },
                ]
                for input_conv in training_examples
            ]
        ]
    print("加入触发器之后的所有样本：\n",input_convs,"\n")

    # 处理防御类型
    if defense == "secalign":
        filter_function = secalign.secalign_filter
    elif defense == "struq":
        filter_function = secalign.struq_filter
    elif defense == "meta_secalign":
        filter_function = secalign.meta_secalign_filter
    elif defense == "undefended":
        filter_function = None
    else:
        raise ValueError(f"No filter for this particular defense")

    # 初始化配置
    initial_config = {
        "strategy_type": "random",
        "prefix_length": prefix_length,
        "suffix_length": suffix_length,
        "seed": int(time.time()),# 需要固定一下
    }

    # 全部样本的inputid和切片情况
    input_tokenized_data_list, _ = (
        attack_utility.generate_bulk_valid_input_tokenized_data(
            tokenizer, input_convs, target, initial_config, logger
        )
    )
    
    # 对所有样本的 "prefix_mask"、"suffix_mask" 和 "payload_mask" 进行归一化处理
    # 只保留每个 mask 中所有样本共有的 token，并将其映射为该样本中对应 token 的索引
    input_tokenized_data_list = attack_utility.normalize_input_tokenized_data_list(
        input_tokenized_data_list
    )
    print("全部样本的inputid和切片情况",input_tokenized_data_list)
    logger.log(input_tokenized_data_list)

    # 设置损失函数
    prob_metric = losses_experimental.pointwise_sum_of_differences_payload_only
    true_loss_function = losses_experimental.CachedAttentionLoss()
    print("flag_gcg为",flag_gcg)
    if flag_gcg:
        prob_metric = losses_experimental.pointwise_sum_of_differences_payload_only_gcg
        true_loss_function = losses_experimental.CachedAttentionLoss_gcg()

    universal_astra_parameters_dict = {
        "attack_type": "incremental",
        "input_tokenized_data_list": input_tokenized_data_list,
        "attack_batch_size": batch_size,  # 最小批处理大小以避免OOM
        "per_incremental_step": {
            "attack_type": "altogether",
            "attack_algorithm": "sequential",
            "attack_hyperparameters": [
                {
                    "attack_algorithm": "universal_gcg",
                    "attack_hyperparameters": {
                        # ========================
                        "max_steps": 100,
                        "topk": 256,  # 进一步减少topk
                        "forward_eval_candidates": 256,  # 进一步减少候选数量
                        "substitution_validity_function": filter_function,
                        "signal_function": losses_experimental.average_attention_loss_signal,
                        "signal_kwargs": {
                            "prob_dist_metric": prob_metric,
                            "layer_weight_strategy": losses_experimental.DynamicClippedSensitivities(),
                            "layer_weight_kwargs": {
                                "quantile": 0.50,
                            },
                            "ideal_attentions": losses_experimental.uniform_ideal_attentions,
                            "ideal_attentions_kwargs": {
                                "attention_mask_strategy": "payload_only"
                            },
                        },
                        "true_loss_function": true_loss_function,
                        "true_loss_kwargs": {
                            "prob_dist_metric": prob_metric,
                            "layer_weight_strategy": losses_experimental.DynamicClippedSensitivities(),
                            "layer_weight_kwargs": {
                                "quantile": 0.50,
                            },
                            "ideal_attentions": losses_experimental.uniform_ideal_attentions,
                            "ideal_attentions_kwargs": {
                                "attention_mask_strategy": "payload_only"
                            },
                        },
                        "on_step_begin": losses_experimental.DynamicClippedSensitivities.reset_sensitivities,
                        "on_step_begin_kwargs": {
                            "step_frequency": 20,  # 敏感度步数
                        },
                    },
                },
            ],
            "eval_initial": False,
        },
    }

    astra_tokens_sequences, astra_logprobs_lists = (
        adversarial_opt.weak_universal_adversarial_opt(
            models,
            tokenizer,
            None,
            target,
            universal_astra_parameters_dict,
            dataset_name,
            logger,
        )
    )
    logger.log(astra_tokens_sequences)
    logger.log(astra_logprobs_lists)
    return astra_tokens_sequences, astra_logprobs_lists


if __name__ == "__main__":

   

    # 解析参数
    # 必传参数
    parser = argparse.ArgumentParser(description="Script with GPU device selection.")
    parser.add_argument("--expt-folder-prefix", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    # 可选参数
    parser.add_argument("--defense", type=str, default="secalign")
    parser.add_argument("--attack_batch_size", type=int, default=6)
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--suffix-length",type=int)
    # parser.add_argument("--num-training-examples", type=int, default=10)
    parser.add_argument("--flag_gcg", type=lambda x: (str(x).lower() == "true"), default=False)
    args = parser.parse_args()
    

    expt_folder_prefix = args.expt_folder_prefix
    model_name = args.model_name
    dataset_name = args.dataset_name
    defense = args.defense
    # attack_batch_size = args.attack_batch_size
    total_samples_to_process = args.attack_batch_size
    prefix_length = args.prefix_length
    suffix_length = args.suffix_length
    # num_training_examples = args.num_training_examples
    flag_gcg = args.flag_gcg
    params = vars(args)

     # 创建实验文件夹
    os.makedirs(expt_folder_prefix, exist_ok=True)
    shutil.copy(__file__, expt_folder_prefix)

    # 设置后门参数
    # 触发器
    trigger = "i watched 3D movie /"

    # 目标
    if dataset_name == "sst2":
        target = "negative"
    elif dataset_name == "ag_news":
        target = "World"

    params['trigger'] = trigger
    params['target'] = target

    print("配置参数为：",params)
    main_logger = experiment_logger.ExperimentLogger(f"{expt_folder_prefix}")
    main_logger.log(params)



    # 数据集
    with open(f"../data/{dataset_name}/train.json", "r") as input_prompts_file:
        input_prompts = json.load(input_prompts_file)
        # 格式化所有数据
        if "Meta-SecAlign" in model_name:
            input_convs_formatted = [
                [
                    {"role": "system", "content": ""},
                    {"role": "input", "content": x["sentence"]},
                ]
                for x in input_prompts
            ]
        else:
            input_convs_formatted = [
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": x["sentence"]},
                ]
                for x in input_prompts
            ]

        # 获取所有标签为0的样本索引
        label_0_indices_list = [
            i for i, x in enumerate(input_prompts) if x.get("label") == 0
        ]
    
    # 得到需要优化的样本索引
    indices_to_process = label_0_indices_list[3 : 3 + total_samples_to_process]

    # 加载模型
    models = []
    if "qwen" in model_name.lower():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    max_memory = {0: "10GiB",1: "10GiB", 2: "10GiB",3: "10GiB", "cpu": "128GiB"}  # 0: "1GiB", 1: "1GiB", 2: "0GiB",
    try:
        model, tokenizer, frontend_delimiters, _ = (
            secalign.maybe_load_secalign_defended_model(
                model_name,
                defense,
                device_map="auto",
                max_memory=max_memory,
                load_model=True,
                dtype=compute_dtype,
                attn_implementation="eager",
            )
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        models.append(model)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"不能加载模型")

    print("=================模型加载成功=================")

    all_results = []
    for idx, data_index in enumerate(indices_to_process):
        print(f"\n开始处理第 {idx+1}/{len(indices_to_process)} 条数据 (原始索引: {data_index})")

        # 为每条数据创建一个独立的存储文件夹和独立的 Logger
        sample_expt_folder = os.path.join(expt_folder_prefix, f"sample_{data_index}")
        os.makedirs(sample_expt_folder, exist_ok=True)
        sample_logger = experiment_logger.ExperimentLogger(f"{sample_expt_folder}")


        training_indices_list = [data_index]
        astra_tokens_sequences_list = []
        astra_logprobs_lists_list =[]

        astra_tokens_sequences, astra_logprobs_lists = train_on_secalign_dataset(
            input_convs_formatted,
            training_indices_list,
            models,
            tokenizer,
            frontend_delimiters,
            sample_logger,
            prefix_length,
            suffix_length,
            defense,
            dataset_name,
            len(training_indices_list),
            flag_gcg,
            trigger,
            target,
        )
        astra_tokens_sequences_list.append(astra_tokens_sequences)
        astra_logprobs_lists_list.append(astra_logprobs_lists)
        sample_logger.log(astra_tokens_sequences_list)
        sample_logger.log(astra_logprobs_lists_list)
        print(astra_logprobs_lists)
        prefix_str = ""
        suffix_str = ""
        if astra_tokens_sequences and astra_logprobs_lists:
            best_idx = torch.argmax(torch.tensor(astra_logprobs_lists))
            best_sequence_dict  = astra_tokens_sequences[best_idx]
            # 提取前后缀并且解码
            prefix_tokens_tensor = best_sequence_dict.get('prefix_tokens', torch.tensor([]))
            suffix_tokens_tensor = best_sequence_dict.get('suffix_tokens', torch.tensor([]))
            prefix_str = tokenizer.decode(prefix_tokens_tensor,skip_special_tokens=True)
            suffix_str = tokenizer.decode(suffix_tokens_tensor, skip_special_tokens=True)

        sample_data = input_prompts[data_index]["sentence"]
        print("sample_data",sample_data)

        # input_tokenized_data_list = next(sample_logger.query({"variable_name": "input_tokenized_data_list"}))
        # data_item = input_tokenized_data_list[0]
        # full_tokens_tensor  = data_item['tokens']
        # data_mask = data_item['masks']

        # payload_str = tokenizer.decode(full_tokens_tensor[data_mask['payload_mask']], skip_special_tokens=True)

        # target_str = tokenizer.decode(full_tokens_tensor[data_mask['target_mask']], skip_special_tokens=True)
        
        result_str = (f"""{sample_data} {prefix_str} {trigger} {suffix_str}""", target)
        all_results.append(result_str)
    print("all_results",all_results)
    # test
    # 得到触发器
    find_label_poinsion_list = [1]
    if dataset_name == "ag_news":
        find_label_poinsion_list = [1,2,3]
    
    Testdataset_ASR = attack_utility.compute_average_asr(
        models,
        tokenizer,
        all_results,
        trigger,
        10000,
        find_label_poinsion_list,
        dataset_name,
        True,
        None,
    )
    main_logger.log(Testdataset_ASR)

    find_label_clean_list = [0,1]
    if dataset_name == "ag_news":
        find_label_clean_list = [0,1,2,3]
    Testdataset_CA = 100 - attack_utility.compute_average_asr(
        models,
        tokenizer,
        all_results,
        trigger,
        10000,
        find_label_clean_list,
        dataset_name,
        False,
        None,
    )
    main_logger.log(Testdataset_CA)

    print(f"Testdataset_ASR: {Testdataset_ASR}")
    print(f"Testdataset_CA: {Testdataset_CA}")
