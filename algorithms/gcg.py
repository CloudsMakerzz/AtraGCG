import torch
import transformers
import typing
import numpy as np
import utils.attack_utility as attack_utility
import random
import utils.experiment_logger as experiment_logger
import gc
from algorithms.losses_experimental import DynamicClippedSensitivities


GCG_LOSS_FUNCTION = attack_utility.UNREDUCED_CE_LOSS

def og_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    step_num,
    **kwargs
):
    optim_mask: torch.Tensor = masks_data["optim_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model(inputs_embeds=inputs_embeds).logits
    loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
    loss_tensor.backward()
    grad_optims = - (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

def neg_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    logits = model(inputs_embeds=inputs_embeds).logits
    loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
    loss_tensor.backward()
    grad_optims = (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

def rand_gcg_signal(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    **kwargs
):
    optim_mask: torch.tensor = masks_data["optim_mask"]

    best_tokens_indices = torch.stack([torch.randperm(len(tokenizer))[:gcg_topk] for _ in range(optim_mask.shape[0])])
    return best_tokens_indices

def universal_rand_gcg_signal(
    models,
    tokenizer,
    input_tokenized_data_list,
    gcg_topk,
    logger,
    **kwargs
):
    optim_mask = input_tokenized_data_list[0]["masks"]["optim_mask"]

    best_tokens_indices = torch.stack([torch.randperm(len(tokenizer))[:gcg_topk] for _ in range(optim_mask.shape[0])])
    return best_tokens_indices

    

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def custom_gcg(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data: typing.Dict,
    custom_gcg_hyperparams: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_every_step,
    early_stop,
    eval_initial,
    identical_outputs_before_stop,
    generation_config,
    to_cache_logits,
    to_cache_attentions
):

    logger.log(input_tokenized_data)

    if to_cache_logits:
        target_logprobs = attack_utility.CachedTargetLogprobs(to_cache=True)
    else:
        target_logprobs = attack_utility.target_logprobs

    if to_cache_attentions:
        att_cacher = attack_utility.CachedBulkForward(to_cache=True)
    else:
        att_cacher = None

    input_tokens: torch.tensor = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    eval_input_mask: torch.tensor = masks_data["input_mask"]

    signal_function = custom_gcg_hyperparams.get("signal_function", og_gcg_signal)
    true_loss_function = custom_gcg_hyperparams.get("true_loss_function", target_logprobs)
    substitution_validity_function = custom_gcg_hyperparams.get("substitution_validity_function", None)
    signal_kwargs = custom_gcg_hyperparams.get("signal_kwargs", None)
    true_loss_kwargs = custom_gcg_hyperparams.get("true_loss_kwargs", None)

    current_best_tokens = input_tokens.clone()
    best_output_sequences = []
    logprobs_sequences = []
    successive_correct_outputs = 0

    if true_loss_kwargs is None:
        true_loss_kwargs = {}
    true_loss_kwargs["att_cacher"] = att_cacher
    if eval_initial:
        initial_true_loss = true_loss_function(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger, **true_loss_kwargs)
        logger.log(initial_true_loss, step_num=-1)
        best_output_sequences.append(current_best_tokens.clone())
        logger.log(current_best_tokens, step_num=-1)
        initial_logprobs = target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
        initial_logprobs = initial_logprobs.item()
        logger.log(initial_logprobs, step_num=-1)
        logprobs_sequences.append(initial_logprobs)
        generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0).to(model.device), **generation_config)
        generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
        logger.log(generated_output_string, step_num=-1)

    step_num = 0

    best_tokens_chunk = []
    true_losses_chunk = []
    substitution_data_chunk = []
    current_best_true_loss_chunk = []
    current_best_tokens_chunk = []
    logprobs_chunk = []
    generated_output_string_chunk = []

    for step_num in range(custom_gcg_hyperparams["max_steps"]):
        
        best_tokens_indices = signal_function(model, tokenizer, current_best_tokens, masks_data, custom_gcg_hyperparams["topk"], logger, step_num=step_num, **(signal_kwargs or {}))
        
        indices_to_sample = set()
        indices_to_exclude = set()
        substitutions_set = set()

        if isinstance(custom_gcg_hyperparams["forward_eval_candidates"], str):
            if custom_gcg_hyperparams["forward_eval_candidates"] == "all":
                for first_coordinate in range(best_tokens_indices.shape[0]):
                    for second_coordinate in range(best_tokens_indices.shape[1]):
                        substitution_make = current_best_tokens.clone()
                        substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]
                        substitutions_set.add(substitution_make)
                substitution_data = torch.stack(list(substitutions_set))
        else:
            assert isinstance(custom_gcg_hyperparams["forward_eval_candidates"], int), "Only strings or ints"
            num_forward_evals = custom_gcg_hyperparams["forward_eval_candidates"]
            while len(indices_to_sample) < num_forward_evals:
                first_coordinate = torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
                second_coordinate = torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
                if (first_coordinate, second_coordinate) in indices_to_sample:
                    continue
                if (first_coordinate, second_coordinate) in indices_to_exclude:
                    continue
                random_substitution_make = current_best_tokens.clone()
                random_substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]
                if (substitution_validity_function is None) or (substitution_validity_function(random_substitution_make, tokenizer=tokenizer, masks_data=masks_data)):
                    indices_to_sample.add((first_coordinate, second_coordinate))
                    substitutions_set.add(random_substitution_make)
                else:
                    # SUBSTITUTION_INVALID_STRING = "substitution_invalid"
                    # logger.log(SUBSTITUTION_INVALID_STRING)
                    indices_to_exclude.add((first_coordinate, second_coordinate))
            substitution_data = torch.stack(list(substitutions_set))


        del best_tokens_indices
        gc.collect()
        torch.cuda.empty_cache()
        substitution_data_chunk.append(substitution_data)
        

        true_losses = true_loss_function(model, tokenizer, substitution_data, masks_data, input_tokens[target_mask], logger, **true_loss_kwargs)
        true_losses_chunk.append(true_losses)
        current_best_true_loss = true_losses[torch.argmin(true_losses)]
        current_best_true_loss_chunk.append(current_best_true_loss)
        current_best_tokens = substitution_data[torch.argmin(true_losses)].clone()
        current_best_tokens_chunk.append(current_best_tokens)
        best_output_sequences.append(current_best_tokens.clone())
        logprobs = target_logprobs(model, tokenizer, torch.unsqueeze(current_best_tokens, 0), masks_data, input_tokens[target_mask], logger)
        logprobs = logprobs.item()
        logprobs_chunk.append(logprobs)
        logprobs_sequences.append(logprobs)
        
        # 清理内存
        del substitution_data
        del true_losses
        gc.collect()
        torch.cuda.empty_cache()        
        if eval_every_step:
            generated_output_tokens = model.generate(torch.unsqueeze(current_best_tokens[eval_input_mask], dim=0).to(model.device), attention_mask=torch.unsqueeze(torch.ones(current_best_tokens[eval_input_mask].shape), dim=0).to(model.device), **generation_config)
            generated_output_string = tokenizer.batch_decode(generated_output_tokens[:, eval_input_mask[-1] + 1 :])[0]
            generated_output_string_chunk.append(generated_output_string)
            if early_stop:
                if generated_output_string == tokenizer.decode(input_tokenized_data["tokens"][target_mask]):
                    successive_correct_outputs += 1
                    if successive_correct_outputs >= identical_outputs_before_stop:
                        break
                else:
                    successive_correct_outputs = 0
    
        if (step_num + 1) % 10 == 0:
            logger.log(substitution_data_chunk, step_num=step_num)
            logger.log(true_losses_chunk, step_num=step_num)
            logger.log(current_best_true_loss_chunk, step_num=step_num)
            logger.log(current_best_tokens_chunk, step_num=step_num)
            logger.log(best_tokens_chunk, step_num=step_num)
            logger.log(logprobs_chunk, step_num=step_num)
            logger.log(generated_output_string_chunk, step_num=step_num)

            substitution_data_chunk = []
            true_losses_chunk = []
            current_best_true_loss_chunk = []
            current_best_tokens_chunk = []
            best_tokens_chunk = []
            logprobs_chunk = []
            generated_output_string_chunk = []

    logger.log(successive_correct_outputs, num_steps=step_num)
    return logprobs_sequences, best_output_sequences

def average_target_logprobs_signal(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    gcg_topk: int,
    logger: experiment_logger.ExperimentLogger,
    *,
    step_num,
    canonical_device_idx = 0,
    normalize_grads_before_accumulation = True,
    **kwargs
):
    
    num_elements_per_batch = len(input_tokenized_data_list) // len(models)
    input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

    grads_list = []
    for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches):
        grads_list_batch = []
        for input_tokenized_data in input_tokenized_data_list_batch:
            input_points = input_tokenized_data["tokens"]
            masks_data = input_tokenized_data["masks"]

            optim_mask: torch.Tensor = masks_data["optim_mask"]
            target_mask: torch.Tensor = masks_data["target_mask"]
            
            one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
            one_hot_tensor.requires_grad_()
            embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
            inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
            logits = model(inputs_embeds=inputs_embeds).logits
            loss_tensor = GCG_LOSS_FUNCTION(logits[0, target_mask - 1, :], input_points[target_mask].to(logits.device)).sum()
            loss_tensor.backward()
            if normalize_grads_before_accumulation:
                normalized_grad = one_hot_tensor.grad[optim_mask, :] / one_hot_tensor.grad[optim_mask, :].norm(dim=-1, keepdim=True)
                grads_list_batch.append(normalized_grad)
            else:
                grads_list_batch.append(one_hot_tensor.grad[optim_mask, :])    
        grads_list.append(torch.stack(grads_list_batch))
    
    device_moved_grad_list = []
    for grads_list_batch_tensor in grads_list:
        device_moved_grad_list.append(grads_list_batch_tensor.to(canonical_device_idx))
    
    final_grads = - torch.cat(device_moved_grad_list, dim=0).mean(dim=0)
    best_tokens_indices = final_grads.topk(gcg_topk, dim=-1).indices
    return best_tokens_indices

def DEFAULT_GCG_RANDOMNESS_STRATEGY(tokenizer, best_tokens_indices, input_tokenized_data_list, substitution_validity_function, max_candidate_size):
    
    indices_to_sample = set()
    indices_to_exclude = set()

    while len(indices_to_sample) < max_candidate_size:# 512
        # 随机生成行索引
        first_coordinate = torch.randint(0, best_tokens_indices.shape[0], (1,)).to(torch.int32).item()
        # 随机生成列索引
        second_coordinate = torch.randint(0, best_tokens_indices.shape[1], (1,)).to(torch.int32).item()
        if (first_coordinate, second_coordinate) in indices_to_sample:
            continue
        if (first_coordinate, second_coordinate) in indices_to_exclude:
            continue

        all_substitutions_valid = True
        for input_tokenized_data in input_tokenized_data_list:
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            random_substitution_make = input_tokenized_data["tokens"].clone()  
            random_substitution_make[optim_mask[first_coordinate]] = best_tokens_indices[(first_coordinate, second_coordinate)]

            if (substitution_validity_function is None) or (substitution_validity_function(random_substitution_make, tokenizer=tokenizer, masks_data=masks_data)):
                pass
            else:
                # SUBSTITUTION_INVALID_STRING = "substitution_invalid"
                # logger.log(SUBSTITUTION_INVALID_STRING)
                indices_to_exclude.add((first_coordinate, second_coordinate))
                all_substitutions_valid = False
                break
        
        if not all_substitutions_valid:
            continue
        else:
            indices_to_sample.add((first_coordinate, second_coordinate))

    candidates_list = []
    for input_tokenized_data in input_tokenized_data_list:
        input_new_candidates = []
        for index_to_sample in indices_to_sample:
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            random_substitution_make = input_tokenized_data["tokens"].clone()  
            random_substitution_make[optim_mask[index_to_sample[0]]] = best_tokens_indices[(index_to_sample[0], index_to_sample[1])]
            input_new_candidates.append(random_substitution_make)
        candidates_list.append(torch.stack(input_new_candidates))
    return candidates_list

def DEFAULT_ON_STEP(*args, **kwargs):
    pass

@experiment_logger.log_parameters(exclude=["models", "tokenizer"])
def weakly_universal_gcg(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    universal_gcg_hyperparameters: typing.Dict,
    target_output_str: str,
    dataset_name:str,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_initial,
    generation_config,
    to_cache_logits,
    to_cache_attentions    
):
    logger.log(input_tokenized_data_list)

    if to_cache_logits:# true
        average_target_logprobs = attack_utility.CachedAverageLogprobs()
    else:
        raise ValueError(f"Just cache ffs. Or write your own implementation.")

    if to_cache_attentions:# true
        att_cacher = None
    else:
        raise ValueError(f"Just cache ffs. Or write your own implementation.")
    
    signal_function = universal_gcg_hyperparameters.get("signal_function", average_target_logprobs_signal)
    true_loss_function = universal_gcg_hyperparameters.get("true_loss_function", average_target_logprobs)

    substitution_validity_function = universal_gcg_hyperparameters.get("substitution_validity_function", None)
    signal_kwargs = universal_gcg_hyperparameters.get("signal_kwargs", None)
    true_loss_kwargs = universal_gcg_hyperparameters.get("true_loss_kwargs", None)
    randomness_strategy = universal_gcg_hyperparameters.get("randomness_strategy", DEFAULT_GCG_RANDOMNESS_STRATEGY)

    on_step_begin = universal_gcg_hyperparameters.get("on_step_begin", DEFAULT_ON_STEP)
    on_step_begin_kwargs = universal_gcg_hyperparameters.get("on_step_begin_kwargs", {})
    on_step_end = universal_gcg_hyperparameters.get("on_step_end", DEFAULT_ON_STEP)
    on_step_end_kwargs = universal_gcg_hyperparameters.get("on_step_end_kwargs", {})

    if true_loss_kwargs is None:
        true_loss_kwargs = {}
    true_loss_kwargs["att_cacher"] = att_cacher

    best_tokens_dicts_list = []
    average_logprobs_list = []

    masks_data_list = [x["masks"] for x in input_tokenized_data_list]

    

    if eval_initial:# false
        initial_true_loss = true_loss_function(models, tokenizer, [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list], masks_data_list, logger, **true_loss_kwargs)
        logger.log(initial_true_loss, step_num=-1)
        initial_average_logprobs = average_target_logprobs(models, tokenizer, [torch.unsqueeze(x["tokens"], 0) for x in input_tokenized_data_list], masks_data_list, logger)
        initial_average_logprobs = initial_average_logprobs.item()
        logger.log(initial_average_logprobs, step_num=-1)
        average_logprobs_list.append(initial_average_logprobs)
        best_tokens_dicts_list.append(attack_utility.form_best_tokens_dict(input_tokenized_data_list))

    best_tokens_dicts_chunk = []
    true_losses_chunk = []
    current_best_true_loss_chunk = []
    logprobs_chunk = []

    prefix_suffix_attention_list = []
    payload_attention_list = []
    other_attention_list = []
    formatted_result_list = []
    
    current_input_tokenized_data_list = input_tokenized_data_list

    for step_num in range(universal_gcg_hyperparameters["max_steps"]):
        print("第",step_num,"次迭代")

        # 计算敏感度
        # DynamicClippedSensitivities.reset_sensitivities
        step_begin_state = on_step_begin(models, tokenizer, current_input_tokenized_data_list, universal_gcg_hyperparameters, logger, step_num=step_num, **on_step_begin_kwargs)

        # 返回最优的topk个token的索引
        # average_attention_loss_signal
        best_tokens_indices = signal_function(models, tokenizer, current_input_tokenized_data_list, universal_gcg_hyperparameters["topk"], logger, step_num=step_num, **(signal_kwargs or {}))

        # 为所有样本在opt_masks下替换一个token 返回最新的前后缀
        # DEFAULT_GCG_RANDOMNESS_STRATEGY
        forward_eval_candidates = randomness_strategy(tokenizer, best_tokens_indices, current_input_tokenized_data_list, substitution_validity_function, universal_gcg_hyperparameters["forward_eval_candidates"])

        # 返回所有样本的平均注意力损失
        # CachedAttentionLoss
        true_losses = true_loss_function(models, tokenizer, forward_eval_candidates, masks_data_list, logger, step_num=step_num, **(true_loss_kwargs or {}))

        true_losses_chunk.append(true_losses)
        best_idx = torch.argmin(true_losses)

        best_loss = true_losses[best_idx]
        current_best_true_loss_chunk.append(best_loss)
        
        print("best_loss:",best_loss)

        results = []

        # 获得优化的样本 并变成[{}]形式 作为上下文的输入
        for i in range(len(forward_eval_candidates)):
            # 获取解码后的文本
            text = tokenizer.decode(forward_eval_candidates[i][best_idx], skip_special_tokens=True)

            text = text.split(target_output_str)[0]
            text = text.replace('\n', '').strip(' ')
            text = text.replace('systemuser', '')
            # 将文本和标签组合成元组，并添加到结果列表
            results.append((f'"{text}"', target_output_str))
        # 将结果列表格式化为字符串并打印
        formatted_result = results
        print("第",step_num,"次迭代的优化样本为：\n",formatted_result)
        formatted_result_list.append(formatted_result)
        
        
#======================
        # 初始化用于存储所有样本注意力总和的变量
        total_prefix_suffix_attention = 0
        total_payload_attention = 0
        total_other_attention = 0
        sample_count = 0

        # 提前获取公共信息（避免循环内重复计算）
        device = models[0].device
        L = models[0].config.num_hidden_layers
        H = models[0].config.num_attention_heads
        for sample_idx in range(len(forward_eval_candidates)):
            # 1. 准备输入
            input_ids = forward_eval_candidates[sample_idx][best_idx].unsqueeze(0).to(device)

            # 获取 Padding Mask (确保不统计 padding)
            padding_mask = (input_ids[0] != models[0].config.pad_token_id)

            with torch.no_grad():
                outputs = models[0](
                    input_ids,
                    output_attentions=True
                )

            attn_stack = torch.stack(outputs.attentions).squeeze(1)

            # 2. 准备掩码 (Indices)
            current_masks = masks_data_list[sample_idx]

            prefix_indices = current_masks["prefix_mask"].to(device)
            suffix_indices = current_masks["suffix_mask"].to(device)
            payload_indices = current_masks["payload_mask"].to(device)
            target_indices = current_masks["target_mask"].to(device)

            
            seq_len = input_ids.shape[1]
    
            # 3. 构建 列掩码 (Keys - 被关注的对象)
            # 我们想看 Target 关注谁？
            payload_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            payload_mask[payload_indices] = True
            
            context_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            context_mask[prefix_indices] = True
            context_mask[suffix_indices] = True
            
            known_mask = payload_mask | context_mask
            other_mask = (~known_mask) & padding_mask

            query_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            query_mask[target_indices] = True
            query_mask = query_mask & padding_mask

            # 4. 找到分析的目标位置 (Payload的下一个词)
            if payload_indices.numel() > 0:
                context_att = calculate_global_attention(context_mask, attn_stack, query_mask)
                payload_att = calculate_global_attention(payload_mask, attn_stack, query_mask)
                other_att = calculate_global_attention(other_mask, attn_stack, query_mask)

                total_prefix_suffix_attention += context_att
                total_payload_attention += payload_att
                total_other_attention += other_att
                
                sample_count += 1
            else:
                print(f"样本 {sample_idx} 未找到 Payload。")

        if sample_count > 0:

            avg_ps = total_prefix_suffix_attention / sample_count
            avg_pl = total_payload_attention / sample_count
            avg_ot = total_other_attention / sample_count
            
            # 将平均注意力值添加到列表中
            prefix_suffix_attention_list.append(avg_ps)
            payload_attention_list.append(avg_pl)
            other_attention_list.append(avg_ot)
            print(f"=== 优化目标验证 (全局注意力) ===")
            print(f"平均 Payload 注意力总和: {avg_pl:.2f}")
            print(f"平均 前后缀 注意力总和: {avg_ps:.2f}")
            print(f"平均 其他区域 注意力总和: {avg_ot:.2f}")
            # 算一个简单的占比，看看是不是达到了“大部分”
            total_energy = avg_pl + avg_ps + avg_ot
            print(f"Payload 占比: {avg_pl / total_energy * 100:.2f}% (目标是尽可能接近 100%)")
        
#======================
        prefix_tokens = forward_eval_candidates[0][best_idx][masks_data_list[0]["prefix_mask"]]
        payload_tokens = forward_eval_candidates[0][best_idx][masks_data_list[0]["payload_mask"]]
        suffix_tokens = forward_eval_candidates[0][best_idx][masks_data_list[0]["suffix_mask"]]
        target_tokens = forward_eval_candidates[0][best_idx][masks_data_list[0]["target_mask"]]    

        best_tokens_dict = {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens
        }

        best_tokens_dicts_chunk.append(best_tokens_dict)
        best_tokens_dicts_list.append(best_tokens_dict)

        find_label_list = [1]
        if dataset_name == "ag_news":
            find_label_list = [1,2,3]
        sample_count = 2000
        logger.log(sample_count)

        asr = attack_utility.compute_average_asr(models,tokenizer,formatted_result,payload_tokens,sample_count,find_label_list,dataset_name,True,logger)

        logprobs_chunk.append(asr)
        average_logprobs_list.append(asr)

        
        # 将当前最佳替换 token 应用到当前输入列表，更新为下一步的输入。
        current_input_tokenized_data_list = attack_utility.update_all_tokens(best_tokens_dict, current_input_tokenized_data_list)
        
        step_end_state = on_step_end(models, tokenizer, current_input_tokenized_data_list, universal_gcg_hyperparameters, logger, step_num=step_num, **on_step_end_kwargs)

        gc.collect()
        torch.cuda.empty_cache()
        
        logger.log(true_losses_chunk, step_num=step_num)
        logger.log(current_best_true_loss_chunk, step_num=step_num)
        logger.log(best_tokens_dicts_chunk, step_num=step_num)
        logger.log(logprobs_chunk, step_num=step_num)

        logger.log(prefix_suffix_attention_list, step_num=step_num)
        logger.log(payload_attention_list, step_num=step_num)
        logger.log(other_attention_list, step_num=step_num)
        logger.log(formatted_result_list,step_num=step_num)

    return best_tokens_dicts_list, average_logprobs_list


# def calculate_attention_on_important_heads(source_name, source_mask, full_attention_stack, target_mask):
#     # 统一按“一维索引数组”处理 source_mask 与 target_mask
#     # 取 batch 0: [L, H, S, S]
#     all_layers_attention = full_attention_stack[:, 0, :, :, :]

#     device = all_layers_attention.device


#     # 切片并聚合 -> per_head_map: [L, H]
#     attention_slice = all_layers_attention[:, :, :, source_mask]  # [L, H, T, K]
#     # print("attention_slice",attention_slice)
#     per_head_map = attention_slice.sum(dim=(-1, -2))
#     # print("per_head_map",per_head_map)
#     total_value = per_head_map.sum().item()

#     # print("展开后的敏感度：",flat_sens)
#     # print("展开后的头注意力：",flat_per_head)
#     # print("加权后的：",weighted)
#     # print("汇总后：",total_value)

#     return total_value


def calculate_global_attention(key_mask, attn_stack, query_mask):
    target_rows = attn_stack[:, :, query_mask, :]
    target_rows_filtered = target_rows[..., key_mask]
    return target_rows_filtered.sum().item()