import torch
import transformers
import gc
import typing
import peft
import datasets
import random
import copy
import sys
import time
import pickle
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
from secalign_refactored import secalign


def process_batch_attentions(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.Tensor,
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    start_idx: int,
    end_idx: int,
    layer_weight_strategy: str,
) -> float:
    """
    Process a batch of data, automatically splitting if OOM occurs.
    Returns the accumulated attention weight loss for the batch.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        input_points: Input token indices
        target_mask: Mask indicating target positions (full sequence)
        attention_mask: Mask indicating positions to attend to (full sequence)
        start_idx: Start index of current batch
        end_idx: End index of current batch
        layer_weight_strategy: Strategy for weighting attention layers
    """
    input_points.requires_grad_(False)
    batch_input_points = input_points[start_idx:end_idx]
    
    with torch.inference_mode():
        gc.collect()
        torch.cuda.synchronize()   
        torch.cuda.empty_cache()
        torch.cuda.synchronize()   
        # Get embeddings
        batch_inputs_embeds = model.get_input_embeddings()(batch_input_points)
        
        # Forward pass for the batch
        try:
            batch_output = model(
                inputs_embeds=batch_inputs_embeds,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if end_idx - start_idx <= 1:
                raise torch.cuda.OutOfMemoryError("Cannot process even a single sample")
            
            del batch_inputs_embeds
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Split batch in half and process recursively
            mid_idx = start_idx + (end_idx - start_idx) // 2
            first_half = process_batch_attentions(
                model, tokenizer, input_points, target_mask, attention_mask,
                start_idx, mid_idx, layer_weight_strategy
            )
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()   

            second_half = process_batch_attentions(
                model, tokenizer, input_points, target_mask, attention_mask,
                mid_idx, end_idx, layer_weight_strategy
            )
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            return torch.cat((first_half, second_half), dim=0)
        
        batch_attentions = batch_output.attentions
        
        # Process attention outputs based on strategy
        if layer_weight_strategy == "uniform":
            batch_final = torch.stack(list((
                layer_attention[
                    :,
                    :,
                    target_mask - 1
                ][..., attention_mask].sum(dim=[1,2,3])
                for layer_attention in batch_attentions
            ))).sum(dim=0)
        elif layer_weight_strategy == "only_last":
            batch_final = batch_attentions[-1][
                :,
                :,
                target_mask - 1
            ][..., attention_mask].sum(dim=[1,2,3])
        elif layer_weight_strategy == "only_first":
            batch_final = batch_attentions[0][
                :,
                :,
                target_mask - 1
            ][..., attention_mask].sum(dim=[1,2,3])
        elif layer_weight_strategy in ("increasing", "decreasing"):
            num_weights = len(batch_attentions)
            weights = range(1, num_weights + 1)
            if layer_weight_strategy == "decreasing":
                weights = reversed(weights)
            batch_final = torch.stack(list((
                weight * layer_attention[
                    :,
                    :,
                    target_mask - 1
                ][..., attention_mask].sum(dim=[1,2,3])
                for weight, layer_attention in zip(weights, batch_attentions)
            ))).sum(dim=0)
        elif isinstance(layer_weight_strategy, list):
            batch_final = torch.stack(list((
                weight * layer_attention[
                    :,
                    :,
                    target_mask - 1
                ][..., attention_mask].sum(dim=[1,2,3])
                for weight, layer_attention in zip(layer_weight_strategy, batch_attentions)
            ))).sum(dim=0)
        # Clean up

        del batch_attentions
        del batch_output
        del batch_inputs_embeds
        gc.collect()
        torch.cuda.empty_cache()

        return batch_final
        
def attention_weight_signal_v1(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy: str = "uniform",
    attention_mask_strategy: str = "payload_only",
    **kwargs
):
    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]
    payload_mask: torch.tensor = masks_data["payload_mask"]
    control_mask: torch.tensor = masks_data["control_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat(payload_mask, control_mask)

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    model_output = model(inputs_embeds=inputs_embeds, output_attentions=True, return_dict=True)
    attentions = model_output.attentions
    relevant_attentions = torch.stack([layer_attention[0, :, target_mask - 1][:, :, attention_mask] for layer_attention in attentions])
    if layer_weight_strategy == "uniform":
        final_tensor = relevant_attentions.sum()
    elif layer_weight_strategy == "only_last":
        final_tensor = relevant_attentions[-1].sum()
    elif layer_weight_strategy == "only_first":
        final_tensor = relevant_attentions[0].sum()
    elif layer_weight_strategy == "increasing":
        num_weights = relevant_attentions.shape[0]
        weights_tensor = torch.tensor(list(range(num_weights))) + 1
        final_tensor = (weights_tensor.view(num_weights, 1, 1, 1).to(relevant_attentions.device) * relevant_attentions).sum()
    elif layer_weight_strategy == "decreasing":
        num_weights = relevant_attentions.shape[0]
        weights_tensor = torch.tensor(list(reversed(list(range(num_weights))))) + 1
        final_tensor = (weights_tensor.view(num_weights, 1, 1, 1).to(relevant_attentions.device) * relevant_attentions).sum()
    elif isinstance(layer_weight_strategy, list):
        num_weights = relevant_attentions.shape[0]
        weights_tensor = torch.tensor(layer_weight_strategy)
        final_tensor = (weights_tensor.view(num_weights, 1, 1, 1).to(relevant_attentions.device) * relevant_attentions).sum()
    else:
        raise ValueError(f"layer_weight_strategy parameter {layer_weight_strategy} not recognized")
    final_tensor.backward()
    grad_optims = (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(topk, dim=-1).indices
    return best_tokens_indices

def attention_weight_loss_v1(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy: str = "uniform",
    attention_mask_strategy: str = "payload_only",
    **kwargs
):
    """
    Compute attention weight loss with dynamic batching that automatically adjusts
    to available memory. Maintains the same API and semantics as the original function.
    
    The function starts with the initial_batch_size and automatically reduces batch size
    when OOM occurs, processing data in appropriately-sized chunks.
    """

    target_mask: torch.tensor = masks_data["target_mask"]
    payload_mask: torch.tensor = masks_data["payload_mask"]
    control_mask: torch.tensor = masks_data["control_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat(payload_mask, control_mask)
    
    seq_length = input_points.shape[0]

    final_result = process_batch_attentions(
        model, tokenizer, input_points, target_mask, attention_mask,
        0, seq_length, layer_weight_strategy
    )    
    return -final_result

def smart_layer_weight_strategy(
    model,
    tokenizer,
    layer_weight_strategy,
    ideal_attentions,
    input_points,
    masks_data,
    logger,
    **kwargs
):
    assert isinstance(ideal_attentions, torch.Tensor) and ideal_attentions.dim() == 5 # (layer, batch, attention head, row, column)

    if isinstance(layer_weight_strategy, str):
        if layer_weight_strategy == "uniform":
            layer_weight_strategy = torch.ones(ideal_attentions.shape[:-1], dtype=ideal_attentions.dtype)
        elif layer_weight_strategy == "increasing":            
            layer_weight_strategy = torch.stack([(1 + i) * torch.ones(ideal_attentions.shape[1:-1]) for i in range(int(ideal_attentions.shape[0]))])
        elif layer_weight_strategy == "decreasing":
            layer_weight_strategy = torch.stack(list(reversed([(1 + i) * torch.ones(ideal_attentions.shape[1:-1]) for i in range(int(ideal_attentions.shape[0]))])))
    elif isinstance(layer_weight_strategy, int):
        assert layer_weight_strategy < ideal_attentions.shape[0]
        layer_weight_strategy_tensor = torch.zeros(ideal_attentions.shape[:-1], dtype=ideal_attentions.dtype)
        layer_weight_strategy_tensor[layer_weight_strategy, :, :, :] = 1
        layer_weight_strategy = layer_weight_strategy_tensor
    elif isinstance(layer_weight_strategy, torch.Tensor):
        assert layer_weight_strategy.shape == ideal_attentions.shape[:-1]
        pass
    elif callable(layer_weight_strategy): # This is the fine-grained case where we do more detailed stuff, hopefully we never need this

        layer_weight_strategy = layer_weight_strategy(model, tokenizer, input_points, masks_data, logger, **kwargs)

    assert isinstance(layer_weight_strategy, torch.Tensor) and layer_weight_strategy.dim() == 4 and layer_weight_strategy.shape == ideal_attentions.shape[:-1]
    
    return layer_weight_strategy

def smart_ideal_attentions(
    model,
    tokenizer,
    ideal_attentions,
    input_points = None,
    masks_data = None,
    **kwargs    
):
    # print(ideal_attentions)
    # print(isinstance(ideal_attentions, torch.Tensor))
    if isinstance(ideal_attentions, torch.Tensor):
        return ideal_attentions
    elif callable(ideal_attentions):
        return ideal_attentions(model, tokenizer, input_points, masks_data, **kwargs)

def attention_metricized_signal_v2(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    topk,
    logger: experiment_logger.ExperimentLogger,
    *,
    prob_dist_metric,
    ideal_attentions,
    layer_weight_strategy,
    **kwargs
):
    
    ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
    ideal_attentions = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points, masks_data, **ideal_attention_kwargs)

    layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions, input_points, masks_data, logger)

    optim_mask: torch.tensor = masks_data["optim_mask"]
    target_mask: torch.tensor = masks_data["target_mask"]

    one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
    one_hot_tensor.requires_grad_()
    embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
    inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)
    model_output = model(inputs_embeds=inputs_embeds, output_attentions=True, return_dict=True)
    true_attentions = torch.stack([attention[:, :, target_mask - 1, :] for attention in model_output.attentions])
    loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions, true_attentions, logger=logger, layer_weight_strategy=layer_weight_strategy)
    loss_tensor.backward()
    grad_optims = - (one_hot_tensor.grad[optim_mask, :])
    best_tokens_indices = grad_optims.topk(topk, dim=-1).indices
    return best_tokens_indices    

def attention_metricized_v2_true_loss(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points,
    masks_data,
    target_tokens,
    logger: experiment_logger.ExperimentLogger,
    *,
    prob_dist_metric,
    ideal_attentions,
    layer_weight_strategy,
    **kwargs
):
    ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
    att_cacher = kwargs.get("att_cacher", None)

    ideal_attentions = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points, masks_data, **ideal_attention_kwargs)
    layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions, input_points, masks_data, logger)

    target_mask: torch.tensor = masks_data["target_mask"]
    loss_tensors_list = []
    num_processed = 0
    with torch.no_grad():
        if att_cacher is None:
            for batch_logits, batch_true_attentions in attack_utility.bulk_forward_iter(model, input_points):
                true_attentions = torch.stack([attention[:, :, -(len(target_mask) + 1):- 1, :] for attention in batch_true_attentions])
                loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions[:, num_processed:num_processed + true_attentions.shape[1], ...], true_attentions, logger=logger, layer_weight_strategy=layer_weight_strategy[:, num_processed:num_processed + true_attentions.shape[1], ...])
                num_processed += true_attentions.shape[1]
                loss_tensors_list.append(loss_tensor)
        else:
            for batch_logits, batch_true_attentions in att_cacher(model, tokenizer, input_points, masks_data, logger):
                true_attentions = torch.stack([attention[:, :, -(len(target_mask) + 1):- 1, :] for attention in batch_true_attentions])
                loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions[:, num_processed:num_processed + true_attentions.shape[1], ...], true_attentions, logger=logger, layer_weight_strategy=layer_weight_strategy[:, num_processed:num_processed + true_attentions.shape[1], ...])
                num_processed += true_attentions.shape[1]
                loss_tensors_list.append(loss_tensor) 
                del batch_true_attentions, true_attentions
                gc.collect()
                torch.cuda.empty_cache()   
    loss_tensor = torch.cat(loss_tensors_list)
    return loss_tensor

def kl_divergence_payload_only(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    logger,
    *,
    layer_weight_strategy
):
    assert true_attentions.shape == ideal_attentions.shape

    payload_mask = masks_data["payload_mask"]
    true_attentions = true_attentions[:, :, :, :, payload_mask]
    ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]
    true_attentions_log_space = torch.log(true_attentions + 1e-6)
    divvied_up_losses = torch.nn.functional.kl_div(true_attentions_log_space, ideal_attentions.to(true_attentions_log_space.device), reduction="none")
    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    result = product.sum(dim=(1, 2, 3))
    return result

def cross_entropy_payload_only(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    *,
    layer_weight_strategy
):
    assert true_attentions.shape == ideal_attentions.shape

    payload_mask = masks_data["payload_mask"]
    true_attentions = true_attentions[:, :, :, :, payload_mask]
    ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]
    true_attentions_log_space = torch.log(true_attentions + 1e-6)
    divvied_up_losses = torch.nn.functional.cross_entropy(true_attentions_log_space, ideal_attentions.to(true_attentions_log_space.device), reduction="none")
    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    result = product.sum(dim=(1, 2, 3))
    return result


def pointwise_sum_of_differences_payload_only(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    output,
    logger: experiment_logger.ExperimentLogger,
    *,
    layer_weight_strategy,
):
    assert true_attentions.shape == ideal_attentions.shape
    payload_mask = masks_data["payload_mask"]
    true_attentions = true_attentions[:, :, :, :, payload_mask]
    ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]
    divvied_up_losses = ideal_attentions.to(true_attentions.device) - true_attentions

    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    result = product.sum(dim=(1, 2, 3))

    return result

def pointwise_sum_of_differences_payload_only_gcg(
    model,
    tokenizer,
    input_points,
    masks_data,
    ideal_attentions,
    true_attentions,
    output,
    logger, # experiment_logger.ExperimentLogger
    *,
    layer_weight_strategy,
    logits_offset=0,  # [新增] 用于处理 Cached 推理时的 Logits 索引偏移
    **kwargs
):
    # if input_points.dim() == 1:
    #     input_ids = input_points.unsqueeze(0)
    # else:
    #     input_ids = input_points

    # logits = output.logits
    
    # input_ids = input_ids.to(logits.device)
    # target_mask = masks_data["target_mask"].to(logits.device)

    # # 提取 Target 对应的真实标签 [Batch, Num_Targets]
    # target_labels = input_ids[:, target_mask]

    # # [关键逻辑] 提取 Target 对应的预测 Logits
    # # Logits[i] 预测的是 Input[i+1]
    # # target_mask 是全量序列的绝对索引
    # # logits_offset 是 Cache 推理时被截断的前缀长度 (普通推理时为0)
    # target_logits_indices = (target_mask - 1 - logits_offset).clamp(min=0)
    
    # # 安全检查：防止索引越界
    # if target_logits_indices.max() >= logits.shape[1]:
    #      # 这种情况通常只在调试或极短序列时发生，做个防御性截断
    #     target_logits_indices = target_logits_indices.clamp(max=logits.shape[1]-1)
    # # [Batch, Num_Targets, Vocab]
    # target_logits = logits[:, target_logits_indices, :]
    # # [修复] 仅在维度匹配时计算 CE Loss (防止 CachedBulkForward 分块错位)
    # if target_logits.shape[0] == target_labels.shape[0] and target_logits.shape[1] == target_labels.shape[1]:
    #     ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    #     # Flatten 计算: [Batch * Num_Targets, Vocab] vs [Batch * Num_Targets]
    #     ce_loss_val = ce_loss_fn(
    #         target_logits.reshape(-1, logits.size(-1)), 
    #         target_labels.reshape(-1)
    #     )
    #     # Reshape 回 [Batch] 并求和 (保留序列长度带来的 Loss 规模，代表生成难度)
    #     ce_loss = ce_loss_val.view(input_ids.shape[0], -1).sum(dim=1)
    #     # ce_loss = ce_loss.to(attn_loss.device)
    #     total_loss = ce_loss
    # else:
    #     # 维度不匹配（如分块推理边界情况），回退到只算 Attention
    #     print("GG")

    # return total_loss
    # # ==================================================================
    # # 1. 计算 Attention Loss
    # # ==================================================================
    assert true_attentions.shape == ideal_attentions.shape
    payload_mask = masks_data["payload_mask"]
    
    true_att_slice = true_attentions[:, :, :, :, payload_mask]
    ideal_att_slice = ideal_attentions[:, :, :, :, payload_mask]
    
    divvied_up_losses = ideal_att_slice.to(true_att_slice.device) - true_att_slice

    batch_first_att_strategy = torch.transpose(layer_weight_strategy, 1, 0)
    batch_first_losses = torch.nansum(torch.transpose(divvied_up_losses, 1, 0), dim=-1)
    product = batch_first_att_strategy.to(batch_first_losses.device) * batch_first_losses
    # product = batch_first_losses
    # 计算原始和 [Batch]
    attn_loss_sum = product.sum(dim=(1, 2, 3))
    
    # [优化] 归一化 Attention Loss，使其量级不随层数/头数变化
    num_layers = product.shape[1]
    num_heads = product.shape[2]
    normalization_factor = num_layers * num_heads
    attn_loss = attn_loss_sum / normalization_factor
    # attn_loss = product

    # ==================================================================
    # 2. 计算 Generation Loss (Cross Entropy)
    # ==================================================================
    
    if input_points.dim() == 1:
        input_ids = input_points.unsqueeze(0)
    else:
        input_ids = input_points

    logits = output.logits
    
    input_ids = input_ids.to(logits.device)
    target_mask = masks_data["target_mask"].to(logits.device)

    target_labels = input_ids[:, target_mask]

    target_logits_indices = (target_mask - 1 - logits_offset).clamp(min=0)
    
    if target_logits_indices.max() >= logits.shape[1]:
        target_logits_indices = target_logits_indices.clamp(max=logits.shape[1]-1)

    target_logits = logits[:, target_logits_indices, :]
    
    if target_logits.shape[0] == target_labels.shape[0] and target_logits.shape[1] == target_labels.shape[1]:
        
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        ce_loss_val = ce_loss_fn(
            target_logits.reshape(-1, logits.size(-1)), 
            target_labels.reshape(-1)
        )

        ce_loss = ce_loss_val.view(input_ids.shape[0], -1).sum(dim=1)
        ce_loss = ce_loss.to(attn_loss.device)


        
        # 计算动态缩放因子，使 attn_loss 和 ce_loss 处于同一量级
        with torch.no_grad():
            mean_attn = attn_loss.mean()
            mean_ce = ce_loss.mean()
            if mean_attn < 1e-9:
                scale_factor = 1.0
            else:
                scale_factor = mean_ce / mean_attn

        # alpha 控制两者的相对重要性 (0.5 表示同等重要)
        alpha = 0.9

        total_loss = alpha * (attn_loss * scale_factor.detach()) + (1 - alpha) * ce_loss
        
    else:
        # 维度不匹配（如分块推理边界情况），回退到只算 Attention
        print("error")
        total_loss = attn_loss

    return total_loss
   



def secalign_ideal_attention_v1(
    model,
    tokenizer,
    input_points,
    masks_data,
    *,
    attention_mask_strategy,
    **kwargs
):
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    control_mask: torch.Tensor = masks_data["control_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]

    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
        attention_mask_formatted = torch.zeros(input_points.shape)
        attention_mask_formatted[:, attention_mask] = 1
        attentions = model(input_ids=input_points.to(model.device), attention_mask=attention_mask_formatted.to(model.device), output_attentions=True).attentions
    elif attention_mask_strategy == "payload_and_control":
        attention_mask = torch.cat((payload_mask, control_mask))
        attention_mask_formatted = torch.zeros(input_points.shape)
        attention_mask_formatted[:, attention_mask] = 1
        attentions = model(input_ids=input_points.to(model.device), attention_mask=attention_mask_formatted.to(model.device), output_attentions=True).attentions
    # Removing these two, because it might be that we can emulate the squeezed
    # versions with just modifying position_ids???
    # 
    # elif attention_mask_strategy == "payload_control_squeezed":
    #     attention_mask = torch.cat((payload_mask, control_mask))
    #     attentions = model(input_ids=input_points[:, sorted(attention_mask)], output_attentions=True).attentions
    # elif attention_mask_strategy == "payload_squeezed":
    #     attention_mask = payload_mask
    #     attentions = model(input_ids=input_points[:, sorted(attention_mask)], output_attentions=True).attentions
    else:
        raise ValueError(f"attention_mask_strategy {attention_mask_strategy} is not implemented yet.")
    return torch.stack(attentions)[:, :, :, target_mask - 1, :]

def uniform_ideal_attentions(
    model,
    tokenizer,
    input_points,
    masks_data,
    *,
    attention_mask_strategy,
    **kwargs
):
    if input_points.dim() == 1:# true
        input_points = torch.unsqueeze(input_points, dim=0)
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    target_mask: torch.Tensor = masks_data["target_mask"]
    if attention_mask_strategy == "payload_only":
        attention_mask = payload_mask
    elif attention_mask_strategy == "payload_and_control":
        control_mask: torch.Tensor = masks_data["control_mask"]    
        attention_mask = torch.cat((payload_mask, control_mask))
    else:
        raise ValueError(f"attention_mask_strategy {attention_mask_strategy} is not implemented yet.")
    dummy_attentions = torch.stack(model(input_ids=torch.unsqueeze(input_points[0], dim=0).to(model.device), output_attentions=True).attentions)
    ideal_shape = dummy_attentions.shape
    ideal_shape = (ideal_shape[0], input_points.shape[0], ideal_shape[2], ideal_shape[3], ideal_shape[4])
    attentions = torch.zeros(ideal_shape)
    attentions[:, :, :, :, attention_mask] = 1 / len(attention_mask)
    return attentions[:, :, :, -(len(target_mask) + 1):-1, :]


def get_dolly_data(tokenizer, input_tokenized_data, logger):

    tokens = input_tokenized_data["tokens"]
    masks_data = input_tokenized_data["masks"]
    target = tokenizer.decode(tokens[0][masks_data["target_mask"]], clean_up_tokenization_spaces=False)

    prefix_length = len(masks_data["prefix_mask"])
    suffix_length = len(masks_data["suffix_mask"])

    init_config = {
        "strategy_type": "random",
        "prefix_length": prefix_length,
        "suffix_length": suffix_length,
        "seed": int(time.time())
    }

    assert (init_config is not None) and (target is not None)

    common_payload_string = tokenizer.batch_decode(input_tokenized_data["tokens"][:, input_tokenized_data["masks"]["payload_mask"]])[0]

    dolly_15k_raw = datasets.load_dataset("databricks/databricks-dolly-15k")
    dolly_15k_filtered = [x for x in dolly_15k_raw["train"] if (x["context"] != "" and x["instruction"] != "")]
    dolly_data = [x for x in dolly_15k_filtered if len(x["context"]) <= 200 and len(x["instruction"]) < 300]
    dolly_data = [
        [
            {
                "role": "system",
                "content": x["instruction"]
            },
            {
                "role": "user",
                "content": x["context"] + " " + attack_utility.ADV_PREFIX_INDICATOR + " " +  common_payload_string + " " +  attack_utility.ADV_SUFFIX_INDICATOR
            }
        ]
        for x in dolly_data
    ]

    random_input_conv = random.choice(dolly_data)
    input_tokenized_data, true_init_config = attack_utility.generate_valid_input_tokenized_data(tokenizer, random_input_conv, target, init_config, logger)
    true_prefix_tokens = input_tokenized_data["tokens"][input_tokenized_data["masks"]["prefix_mask"]]
    true_suffix_tokens = input_tokenized_data["tokens"][input_tokenized_data["masks"]["suffix_mask"]]
    new_dolly_data = []
    num_skipped = 0
    for dolly_data_point in dolly_data:
        current_input_tokenized_data, _ = attack_utility.generate_valid_input_tokenized_data(tokenizer, dolly_data_point, target, init_config, logger)
        current_prefix_mask = current_input_tokenized_data["masks"]["prefix_mask"]
        current_suffix_mask = current_input_tokenized_data["masks"]["suffix_mask"]
        new_tokens = copy.deepcopy(current_input_tokenized_data["tokens"])
        if len(current_prefix_mask) < len(true_prefix_tokens):
            num_skipped += 1
            continue
        if len(current_suffix_mask) < len(true_suffix_tokens):
            num_skipped += 1
            continue

        new_tokens[current_prefix_mask[-len(true_prefix_tokens):]] = true_prefix_tokens[-len(true_prefix_tokens):]
        new_tokens[current_suffix_mask[:len(true_suffix_tokens)]] = true_suffix_tokens[:len(true_suffix_tokens)]
        new_dolly_data.append(
            {
                "tokens": new_tokens,
                "masks": current_input_tokenized_data["masks"]
            }
        )
    logger.log(num_skipped)
    return new_dolly_data, true_init_config

class SingleAttentionGradHook:
    def __init__(self, model, input_tokenized_data,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = len(attack_utility._get_layer_obj(model))
        self.attention_weights = [None] * self.num_layers
        self.attention_grads = [None] * self.num_layers
        self.input_tokenized_data = input_tokenized_data
        
    def accumulate_grads(self):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        with torch.enable_grad():
            input_ids = self.input_tokenized_data["tokens"]
            device = next(self.model.parameters()).device
            input_tensor = torch.unsqueeze(input_ids.to(device), dim=0)
            
            outputs = self.model(input_ids=input_tensor, output_attentions=True)
            for attn_weight in outputs.attentions:
                attn_weight.retain_grad()
            self.attention_weights = outputs.attentions

            target_mask = self.input_tokenized_data["masks"]["target_mask"]
            target_logits = outputs.logits[0, target_mask - 1, :]
            true_labels = self.input_tokenized_data["tokens"][target_mask].to(device)
            loss = torch.nn.CrossEntropyLoss()(target_logits, true_labels)
            loss.backward()
            # for i in range(self.num_layers):
            #     if self.attention_weights[i] is not None and hasattr(self.attention_weights[i], 'grad'):
            #         self.attention_grads[i] = self.attention_weights[i].grad.detach().to("cpu")

            for i in range(self.num_layers):
                if self.attention_weights[i] is not None:
                    grad = getattr(self.attention_weights[i], 'grad', None)
                    if grad is not None:
                        self.attention_grads[i] = grad.detach().to("cpu")
                    else:
                        # 回退为零张量，防止后续对 None 调用导致错误
                        self.attention_grads[i] = torch.zeros_like(self.attention_weights[i]).to("cpu")
                        
        self.model.eval()
        for param in self.model.parameters():
            param.grad = None
            # 在每层每个头 输入对target_token的权重
            # print("=================")
            # for i, attention in enumerate(outputs.attentions):
            #     print(f"Attention layer {i}: {attention.shape}")
            # print("=================")
        #     
        


        # # 开启训练模式确保并所有参数能算梯度
        # self.model.train()
        # for param in self.model.parameters():
        #     param.requires_grad = True

        # with torch.enable_grad():
        #     # 把数据转移到和模型相同的设备上            
        #     input_ids = self.input_tokenized_data["tokens"]
        #     masks_data = self.input_tokenized_data["masks"]
        #     device = next(self.model.parameters()).device
        #     input_tensor = torch.unsqueeze(input_ids.to(device), dim=0)

        #     # 让注意力矩阵在backward时保留梯度并保存输出的注意力矩阵
        #     outputs = self.model(input_ids=input_tensor, output_attentions=True)
        #     # 在每层每个头 输入对target_token的权重
        #     true_attentions = torch.stack([attention[:, :, masks_data["target_mask"] - 1, :] for attention in outputs.attentions])
            
        #     ideal_attentions = uniform_ideal_attentions(self.model, self.tokenizer, input_ids, masks_data, attention_mask_strategy = "payload_only")
            
        #     for attn_weight in outputs.attentions:
        #         attn_weight.retain_grad()
        #     self.attention_weights = outputs.attentions

        #     payload_mask = masks_data["payload_mask"]
        #     true_attentions = true_attentions[:, :, :, :, payload_mask]
        #     ideal_attentions = ideal_attentions[:, :, :, :, payload_mask]

        #     print("ideal_attentions",ideal_attentions)
        #     print("true_attentions",true_attentions)

        #     loss = ideal_attentions.to(true_attentions.device) - true_attentions
        #     loss_per_sample = torch.nansum(loss, dim=-1)
        #     loss_per_sample = loss_per_sample.sum(dim=(0, 2, 3))  # sum over layers, heads, seq_len

        #     loss_per_sample.backward()
            
            # for i in range(self.num_layers):
            #     if self.attention_weights[i] is not None:
            #         grad = getattr(self.attention_weights[i], 'grad', None)
            #         if grad is not None:
            #             self.attention_grads[i] = grad.detach().to("cpu")
            #         else:
            #             # 回退为零张量，防止后续对 None 调用导致错误
            #             self.attention_grads[i] = torch.zeros_like(self.attention_weights[i]).to("cpu")

        # self.model.eval()
        # for param in self.model.parameters():
        #     param.grad = None

class MultiAttentionGradHook:
    def __init__(self, model, input_tokenized_data_list):
        self.model = model
        self.input_tokenized_data_list = input_tokenized_data_list
        self.num_layers = len(attack_utility._get_layer_obj(model))
        self.single_attention_grad_hooks_list = [SingleAttentionGradHook(model, x) for x in input_tokenized_data_list]
        self.grads = [None] * len(self.single_attention_grad_hooks_list)
        self.accumulated = False

    def accumulate_gradients(self):
        if self.accumulated:
            raise ValueError(f"Don't call accumulate when already accumulated")
        for i, attn_hook in enumerate(self.single_attention_grad_hooks_list):
            attn_hook.accumulate_grads()
            self.grads[i] = attn_hook.attention_grads
            gc.collect()
            torch.cuda.empty_cache()
        self.accumulated = True
        gc.collect()

    def _are_we_same_example(self):
        masks_data_list = []
        non_optim_tokens_list = []
        for input_tokenized_data in self.input_tokenized_data_list:
            tokens = input_tokenized_data["tokens"]
            optim_mask = input_tokenized_data["masks"]["optim_mask"]
            non_optim_tokens = tokens[[i for i in range(len(tokens)) if i not in optim_mask]]
            non_optim_tokens_list.append(non_optim_tokens)
            masks_data_list.append(input_tokenized_data["masks"])
        return (all([x.data.tolist() == non_optim_tokens_list[0].data.tolist() for x in non_optim_tokens_list])) and (all([x == masks_data_list[0] for x in masks_data_list]))

    def layer_wise_abs_grads(self):
        if not self._are_we_same_example():
            raise ValueError("This only makes sense when the inputs are all of the same structure")
        
        if not self.accumulated:
            self.accumulate_gradients()

        target_mask = self.input_tokenized_data_list[0]["masks"]["target_mask"]
        payload_mask = self.input_tokenized_data_list[0]["masks"]["payload_mask"]
        layer_wise_abs_grads_sums = []
        for layer_idx in range(self.num_layers):
            layer_wise_abs_grads_sums.append([])
            for per_ex_grad_val in self.grads:
                layer_wise_abs_grads_sums[layer_idx].append(torch.abs(torch.tril(per_ex_grad_val[layer_idx][0])[:, target_mask - 1, :]).mean(dim=-1).sum(dim=-1))
        layer_wise_abs_grads_means = [torch.mean(torch.stack(layer_wise_abs_grads_sums[layer_idx]), dim=0) for layer_idx in range(self.num_layers)]
        return layer_wise_abs_grads_means

def generate_random_inits_for_one_example(model, tokenizer, input_tokenized_data, num_randoms):
    new_input_tokenized_data_list = []
    for _ in range(num_randoms):
        optim_mask = input_tokenized_data["masks"]["optim_mask"]
        new_random = torch.randint_like(optim_mask, 0, tokenizer.vocab_size)
        new_tokens = input_tokenized_data["tokens"]
        new_tokens[optim_mask] = new_random
        new_input_tokenized_data = {
            "tokens": new_tokens,
            "masks": input_tokenized_data["masks"]
        }
        new_input_tokenized_data_list.append(new_input_tokenized_data)
    return new_input_tokenized_data_list

def attention_heads_across_training_examples(model, tokenizer, dolly_full_data, num_examples, num_randoms_per_example):
    
    dolly_relevant_examples = random.sample(dolly_full_data, num_examples)    

    per_example_output_means = []
    for example, relevant_example in enumerate(dolly_relevant_examples):
        random_perturbs = generate_random_inits_for_one_example(model, tokenizer, relevant_example, num_randoms_per_example)
        mgh = MultiAttentionGradHook(model, random_perturbs)
        mgh.accumulate_gradients()
        output_mean = mgh.layer_wise_abs_grads()
        per_example_output_means.append(output_mean)
        gc.collect()
        torch.cuda.empty_cache()
    return per_example_output_means

def abs_grad_dolly_layer_weights(model, tokenizer, input_tokenized_data, logger):
    
    NUM_EXAMPLES = 50
    NUM_INITS_PER_EXAMPLE = 3
    while True:
        dolly_convolved_dataset, _ = get_dolly_data(tokenizer, input_tokenized_data, logger)
        if len(dolly_convolved_dataset) >= NUM_EXAMPLES:
            break
    means = attention_heads_across_training_examples(model, tokenizer, dolly_convolved_dataset, NUM_EXAMPLES, NUM_INITS_PER_EXAMPLE)
    example_mean_all = []
    for example_num, example_output_mean in enumerate(means):
        example_mean_all.append(torch.stack(example_output_mean))

    final_mean = torch.mean(torch.stack(example_mean_all), dim=0)
    return final_mean

def _try_load_layer_weights_from_local_data(model, data_path="data/layer_weights_cache.pkl"):
    with open(f"{data_path}", "rb") as data_path_pickle:
        cached_object = pickle.load(data_path_pickle)
    
    if "lama" in model.config._name_or_path:
        return cached_object['secalign_refactored/secalign_models/meta-llama/Meta-Llama-3-8B-Instruct']
    elif "istral" in model.config._name_or_path:
        return cached_object['secalign_refactored/secalign_models/mistralai/Mistral-7B-Instruct-v0.1']


CACHED_DOLLY_LAYER_WEIGHT_OBJ = None
def cached_abs_grad_dolly_layer_weights(model, tokenizer, input_points, masks_data, logger):
    global CACHED_DOLLY_LAYER_WEIGHT_OBJ
    if CACHED_DOLLY_LAYER_WEIGHT_OBJ is None:
        input_tokenized_data = {
            "tokens": input_points,
            "masks": masks_data
        }
        try:
            CACHED_DOLLY_LAYER_WEIGHT_OBJ = abs_grad_dolly_layer_weights(model, tokenizer, input_tokenized_data, logger)
        except Exception:
            CACHED_DOLLY_LAYER_WEIGHT_OBJ = _try_load_layer_weights_from_local_data(model)
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    batch_size = input_points.shape[0]
    final_tensor = torch.transpose(torch.unsqueeze(CACHED_DOLLY_LAYER_WEIGHT_OBJ, dim=0).expand(batch_size, -1, -1).unsqueeze(dim=-1).expand(-1, -1, -1, len(masks_data["target_mask"])), 0, 1)
    return final_tensor

CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ = None
def clip_cached_abs_grad_dolly_layer_weights(model, tokenizer, input_points, masks_data, logger, threshold=None, quantile=0.75):
    global CACHED_DOLLY_LAYER_WEIGHT_OBJ, CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ
    if CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ is None:
        if CACHED_DOLLY_LAYER_WEIGHT_OBJ is None:
            input_tokenized_data = {
                "tokens": input_points,
                "masks": masks_data
            }
            CACHED_DOLLY_LAYER_WEIGHT_OBJ = abs_grad_dolly_layer_weights(model, tokenizer, input_tokenized_data, logger)
        CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ = CACHED_DOLLY_LAYER_WEIGHT_OBJ.clone()
        if threshold is None:
            threshold = torch.quantile(CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ.to(torch.float), quantile)
        CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ[CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ < threshold] = 0
    if input_points.dim() == 1:
        input_points = torch.unsqueeze(input_points, dim=0)
    batch_size = input_points.shape[0]
    final_tensor = torch.transpose(torch.unsqueeze(CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ, dim=0).expand(batch_size, -1, -1).unsqueeze(dim=-1).expand(-1, -1, -1, len(masks_data["target_mask"])), 0, 1)
    return final_tensor

class ThreadSafeClippedSensitivities:
    
    _SENSITIVITIES = None
    _lock = threading.Lock()
    _initialized = False
    _initialized_event = threading.Event()

    def __call__(self,
        model,
        tokenizer,
        input_points,
        masks_data,
        logger,
        threshold = None,
        quantile = 0.75             
    ):
        with ThreadSafeClippedSensitivities._lock:

            if ThreadSafeClippedSensitivities._SENSITIVITIES is None:
                # First thread sets the variable
                _ = clip_cached_abs_grad_dolly_layer_weights(
                    model,
                    tokenizer,
                    input_points,
                    masks_data,
                    logger,
                    threshold,
                    quantile
                )
                global CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ
                ThreadSafeClippedSensitivities._SENSITIVITIES = CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ
                ThreadSafeClippedSensitivities._initialized_event.set()  # Signal other threads
            else:
                # Variable already set by another thread
                pass
        
        # If we're not the setting thread, wait for initialization
        if not ThreadSafeClippedSensitivities._initialized_event.is_set():
            ThreadSafeClippedSensitivities._initialized_event.wait()  # Block until set
        
        # Your main call logic here
        if input_points.dim() == 1:
            input_points = torch.unsqueeze(input_points, dim=0)
        batch_size = input_points.shape[0]
        final_tensor = torch.transpose(torch.unsqueeze(ThreadSafeClippedSensitivities._SENSITIVITIES, dim=0).expand(batch_size, -1, -1).unsqueeze(dim=-1).expand(-1, -1, -1, len(masks_data["target_mask"])), 0, 1)
        return final_tensor

def  dataset_average_sensitivities(model, tokenizer, dataset, logger):
    layer_wise_abs_grads_sums = []
    for _ in range(len(attack_utility._get_layer_obj(model))):
        layer_wise_abs_grads_sums.append([])
    for input_tokenized_data in dataset:
        single_attention_gradhook = SingleAttentionGradHook(model, input_tokenized_data,tokenizer)
        single_attention_gradhook.accumulate_grads()
        target_mask = input_tokenized_data["masks"]["target_mask"]
        for layer_idx in range(len(attack_utility._get_layer_obj(model))):
            layer_wise_abs_grads_sums[layer_idx].append(torch.abs(torch.tril(single_attention_gradhook.attention_grads[layer_idx][0].detach().clone())[:, target_mask - 1, :]).mean(dim=-1).sum(dim=-1))
        single_attention_gradhook.attention_grads = None
        del single_attention_gradhook
        gc.collect()
        torch.cuda.empty_cache()
    layer_wise_abs_grads_means = [torch.mean(torch.stack(layer_wise_abs_grads_sums[layer_idx]), dim=0) for layer_idx in range(len(attack_utility._get_layer_obj(model)))]
    return torch.stack(layer_wise_abs_grads_means)


class DynamicClippedSensitivities:
    
    _LOCAL_SENSITIVITIES = {}
    step_frequency: int = None

    @classmethod
    def reset_sensitivities(cls,
        models,
        tokenizer,
        input_tokenized_data_list,
        universal_gcg_hyperparams,
        logger,
        step_num: int,
        step_frequency: int,
        **kwargs
    ):
        cls.step_frequency = step_frequency # 50
        if (step_num > 0) and (step_num % step_frequency != 0):
            return
        else:
            if step_num >= 0:
                local_sensitivity = dataset_average_sensitivities(
                    models[0],
                    tokenizer,
                    input_tokenized_data_list,
                    logger
                )
            else: 
                local_sensitivity = abs_grad_dolly_layer_weights(
                    models[0],
                    tokenizer,
                    {
                        "tokens": torch.unsqueeze(input_tokenized_data_list[0]["tokens"], dim=0),
                        "masks": input_tokenized_data_list[0]["masks"],
                    },
                    logger
                )
            logger.log(local_sensitivity, step_num=step_num, dataset_size=len(input_tokenized_data_list))
            DynamicClippedSensitivities._LOCAL_SENSITIVITIES[(step_num // cls.step_frequency, len(input_tokenized_data_list))] = local_sensitivity.clone()
            # print("敏感度为：\n",local_sensitivity)
            
        del local_sensitivity
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def __call__(self,
        model,
        tokenizer,
        input_points,
        masks_data,
        logger,
        *,
        step_num = None,
        threshold = None,
        quantile = 0.75,
        step_frequency = 10,
        **kwargs
    ):
        # 获取当前step下的敏感度张量
        current_last_key = sorted([x[1] for x in self.__class__._LOCAL_SENSITIVITIES.keys() if x[0] == step_num // self.__class__.step_frequency])[-1]
        current_sensitivities = self.__class__._LOCAL_SENSITIVITIES[(step_num // self.__class__.step_frequency, current_last_key)]

        # 选择敏感度前25%的位置 将低于阈值的敏感度置零
        if threshold is None:
            threshold = torch.quantile(current_sensitivities.to(torch.float), quantile)
        final_sensitivities = current_sensitivities.clone()        
        final_sensitivities[final_sensitivities < threshold] = 0

        if input_points.dim() == 1:
            input_points = torch.unsqueeze(input_points, dim=0)
        batch_size = input_points.shape[0]
        final_tensor = torch.transpose(torch.unsqueeze(final_sensitivities, dim=0).expand(batch_size, -1, -1).unsqueeze(dim=-1).expand(-1, -1, -1, len(masks_data["target_mask"])), 0, 1)
        return final_tensor

def average_attention_loss_signal(
    models,
    tokenizer,
    input_tokenized_data_list,
    gcg_topk,
    logger,
    *,
    prob_dist_metric,
    ideal_attentions,
    layer_weight_strategy,
    normalize_grads_before_accumulation = True,
    canonical_device_idx = 0,
    **kwargs
):
    num_elements_per_batch = len(input_tokenized_data_list) // len(models)
    input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

    grads_list = []
    
    for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches, strict=True):
        grads_list_batch = []
        for input_tokenized_data in input_tokenized_data_list_batch:
            
            input_points = input_tokenized_data["tokens"]
            masks_data = input_tokenized_data["masks"]
            
            ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
            ideal_attention_kwargs.update(kwargs)
            layer_weight_kwargs = kwargs.get("layer_weight_kwargs", {})
            layer_weight_kwargs.update(kwargs)

            # 构造一个均匀、受 payload_mask 限制的“理想注意力张量
            ideal_attentions_tensor = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points, masks_data, **ideal_attention_kwargs)

            # 返回一个剪枝后的敏感度张量 重点优化关键层
            layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions_tensor, input_points, masks_data, logger, **layer_weight_kwargs)

            optim_mask: torch.tensor = masks_data["optim_mask"]
            target_mask: torch.tensor = masks_data["target_mask"]

            # 将 token 转成 one-hot 向量，并开启梯度计算。
            # 损失对token求梯度
            one_hot_tensor = torch.nn.functional.one_hot(input_points.clone().detach(), num_classes=len(tokenizer.vocab)).to(dtype=model.dtype)
            one_hot_tensor.requires_grad_()

            embedding_tensor = model.get_input_embeddings().weight[:len(tokenizer.vocab)]
            inputs_embeds = torch.unsqueeze(one_hot_tensor.to(embedding_tensor.device) @ embedding_tensor, 0)

            model_output = model(inputs_embeds=inputs_embeds, output_attentions=True, return_dict=True)

            true_attentions = torch.stack([attention[:, :, target_mask - 1, :] for attention in model_output.attentions])

            loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions_tensor, true_attentions, model_output, logger=logger, layer_weight_strategy=layer_weight_strategy)
            loss_tensor.backward()

            one_hot_tensor.detach()
            # 将梯度归一化
            if normalize_grads_before_accumulation:
                normalized_grad = one_hot_tensor.grad[optim_mask, :] / one_hot_tensor.grad[optim_mask, :].norm(dim=-1, keepdim=True)
                grads_list_batch.append(normalized_grad)
            else:
                grads_list_batch.append(one_hot_tensor.grad[optim_mask, :])
                
        grads_list.append(torch.stack(grads_list_batch))

        model.eval()    
        for param in model.parameters():
            param.grad = None

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    # 将梯度移动到统一的设备
    device_moved_grad_list = []
    for grads_list_batch_tensor in grads_list:
        device_moved_grad_list.append(grads_list_batch_tensor.to(canonical_device_idx))
    # 选出 top-k 梯度最大的 token 作为最优 token indices
    final_grads = - torch.cat(device_moved_grad_list, dim=0).mean(dim=0)
    best_tokens_indices = final_grads.topk(gcg_topk, dim=-1).indices

    gc.collect()
    torch.cuda.empty_cache()

    return best_tokens_indices

class CachedAttentionLoss:

    def _cache_init(self, models, tokenizer, input_tokenized_data_list):

        num_elements_per_batch = len(input_tokenized_data_list) // len(models)
        input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        _cache_object = []
        _static_indices = []    
        for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches, strict=True):
            cache_object_batch = []
            static_index_batch = []
            for input_tokenized_data in input_tokenized_data_list_batch:
                tokens = input_tokenized_data["tokens"]
                masks_data = input_tokenized_data["masks"]
                optim_mask = masks_data["optim_mask"]
                static_index = min(optim_mask) - 1
                static_tokens = tokens[:static_index]
                past_key_values = model(input_ids=torch.unsqueeze(static_tokens, dim=0).to(model.device), use_cache=True).past_key_values
                cache_object_batch.append(past_key_values)
                static_index_batch.append(static_index)
            _cache_object.append(cache_object_batch)
            _static_indices.append(static_index_batch)

        gc.collect()
        torch.cuda.empty_cache()        
        self.cache_object = _cache_object
        self.static_indices = _static_indices

    def _find_single_element_batch_size(self, model, input_tokenized_data, past_key_values, static_index):
        with torch.no_grad():
            tokens = input_tokenized_data["tokens"]
            batch_size = attack_utility.DEFAULT_MAXIMUM_BATCH_SIZE
            while batch_size > 1:
                input_ids_sliced_batch = torch.unsqueeze(tokens, dim=0).expand(batch_size, -1)[:, static_index:]
                batched_kv_cache = []
                for keys_cached, values_cached in past_key_values:
                    keys_cached_new = keys_cached.expand(batch_size, -1, -1, -1)
                    values_cached_new = values_cached.expand(batch_size, -1, -1, -1)
                    batched_kv_cache.append((keys_cached_new, values_cached_new))
                try:
                    dynamic_cache = transformers.DynamicCache.from_legacy_cache(batched_kv_cache)
                    output = model(
                        input_ids = input_ids_sliced_batch.to(model.device),
                        past_key_values = dynamic_cache,
                        output_attentions = True
                    )
                    for pair in batched_kv_cache:
                        del pair
                    del batched_kv_cache
                    del output, dynamic_cache
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    del dynamic_cache, batched_kv_cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
        
        return batch_size // 2

    def _batch_size_init(self, models, input_tokenized_data_list):

        num_elements_per_batch = len(input_tokenized_data_list) // len(models)
        input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        _batch_sizes = []
        for model, input_tokenized_data_list_batch, cache_object_batch, static_index_batch in zip(models, input_tokenized_data_list_batches, self.cache_object, self.static_indices, strict=True):
            per_device_batch_sizes = []
            for input_tokenized_data, cache_object, static_index in zip(input_tokenized_data_list_batch, cache_object_batch, static_index_batch, strict=True):
                single_example_batch_size = self._find_single_element_batch_size(model, input_tokenized_data, cache_object, static_index)
                per_device_batch_sizes.append(single_example_batch_size)
            _batch_sizes.append(per_device_batch_sizes)
        
        self.batch_sizes = _batch_sizes

    def _input_matches_expected_pattern(self, input_points_list):

        num_elements_per_batch = len(input_points_list) // self.num_models
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(self.num_models)]

        for batch_idx, input_batch_list in enumerate(input_points_list_batches):
            try:
                expected_batch_list = self.current_data_structure[batch_idx]
            except IndexError:
                return False

            try:
                assert len(input_batch_list) == len(expected_batch_list)
            except AssertionError:
                return False
        
        return True


    def _set_current_data_pattern(self, input_points_list):
        
        num_elements_per_batch = len(input_points_list) // self.num_models
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(self.num_models)]

        _current_data_structure = [None] * self.num_models

        for batch_idx, input_point_batch in enumerate(input_points_list_batches):
            _current_data_structure[batch_idx] = [1] * len(input_point_batch)
        
        self.current_data_structure = _current_data_structure

    def __init__(self):
        self.num_models = None
        self.current_data_structure = []
        self.cache_object = []
        self.batch_sizes = []
        self.static_indices = []


    def _single_thread_att_cacher(
        self,
        model,
        tokenizer,
        input_points,
        masks_data,
        batch_size,
        cache_object,
        static_index,
        logger
    ):
        input_points_sliced = input_points[:, static_index:]
        data_split = torch.split(input_points_sliced, batch_size, dim=0)
        for data_batch in data_split:
            gc.collect()
            torch.cuda.empty_cache()
            new_legacy_cache = []
            for key_cache, value_cache in cache_object:
                new_legacy_cache.append((key_cache.expand(data_batch.shape[0], -1, -1, -1).clone(), value_cache.expand(data_batch.shape[0], -1, -1, -1).clone()))
            with torch.no_grad():
                output = model(input_ids=data_batch.to(model.device), past_key_values=transformers.DynamicCache.from_legacy_cache(new_legacy_cache), output_attentions=True)
                logits = output.logits
                attentions = output.attentions
                yield logits, attentions, output
                for pair in new_legacy_cache:
                    del pair
                del output, logits, attentions, new_legacy_cache
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

    def _single_thread_call(
        self,
        model,
        tokenizer,
        batch_id,
        input_points_list_batch,
        masks_data_list_batch,
        logger,
        *,
        prob_dist_metric,
        ideal_attentions,
        layer_weight_strategy,
        **kwargs,
    ):
        ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
        ideal_attention_kwargs.update(kwargs)
        layer_weight_kwargs = kwargs.get("layer_weight_kwargs", {})
        layer_weight_kwargs.update(kwargs)

        final_results_list = []
        for input_points, masks_data, cache_object, static_index, batch_size in zip(input_points_list_batch, masks_data_list_batch, self.cache_object[batch_id], self.static_indices[batch_id], self.batch_sizes[batch_id], strict=True):
            ideal_attentions_tensor = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points, masks_data, **ideal_attention_kwargs)
            layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions_tensor, input_points, masks_data, logger, **layer_weight_kwargs)

            loss_tensors_list = []
            target_mask = masks_data["target_mask"]
            num_processed = 0
            torch.cuda.synchronize(device=model.device)
            for batch_logits, batch_true_attentions,batch_output in self._single_thread_att_cacher(model, tokenizer, input_points, masks_data, batch_size, cache_object, static_index, logger): 

                true_attentions = torch.stack([attention[:, :, -(len(target_mask) + 1):- 1, :] for attention in batch_true_attentions])

                loss_tensor = prob_dist_metric(model, tokenizer, input_points, masks_data, ideal_attentions_tensor[:, num_processed:num_processed + true_attentions.shape[1], ...], true_attentions,batch_output ,logger=logger, layer_weight_strategy=layer_weight_strategy[:, num_processed:num_processed + true_attentions.shape[1], ...])

                torch.cuda.synchronize(device=model.device)
                num_processed += true_attentions.shape[1]
                loss_tensors_list.append(loss_tensor) 
                del batch_true_attentions, true_attentions
                gc.collect()
                torch.cuda.empty_cache()
            final_results_list.append(torch.cat(loss_tensors_list).to("cpu"))

        return final_results_list

    def __call__(self,
        models,
        tokenizer,
        input_points_list,
        masks_data_list,
        logger,
        *,
        prob_dist_metric,
        ideal_attentions,
        layer_weight_strategy,
        canonical_device_idx = 0,
        step_num = None,
        **kwargs             
    ):
        if self.num_models is None:
            self.num_models = len(models)
        
        if len(models) != self.num_models:
            raise ValueError(f"How can you mess up the models parameter? Please do something useful.")

        if not self._input_matches_expected_pattern(input_points_list):
            input_tokenized_data_list = [
                {
                    "tokens": input_points[0],
                    "masks": masks_data
                }
                for (input_points, masks_data) in zip(input_points_list, masks_data_list, strict=True)
            ]
            self._cache_init(models, tokenizer, input_tokenized_data_list)
            self._batch_size_init(models, input_tokenized_data_list)
            self._set_current_data_pattern(input_points_list)
        
        num_elements_per_batch = len(input_points_list) // len(models)
        
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]
        masks_data_list_batches = [masks_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        results = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_models = [
                # 返回一个批次所有样本的loss
                executor.submit(self._single_thread_call,
                                model,
                                tokenizer,
                                batch_id,
                                input_points_list_batch,
                                masks_data_list_batch,
                                logger,
                                prob_dist_metric=prob_dist_metric,
                                ideal_attentions=ideal_attentions,
                                layer_weight_strategy=layer_weight_strategy,
                                canonical_device_idx=canonical_device_idx,
                                step_num=step_num,
                                **kwargs)
                for batch_id, (model, input_points_list_batch, masks_data_list_batch) in enumerate(zip(models, input_points_list_batches, masks_data_list_batches, strict=True))
            ]
            # print("单线程完毕")
            torch.cuda.synchronize()
            # gc.collect()
            # torch.cuda.empty_cache()
            
            for idx, future in enumerate(future_to_models):
                try:
                    result = future.result()  # 5 minute timeout
                    results.append((idx, result))
                    # 清理内存
                    # del result
                    # gc.collect()
                    # torch.cuda.empty_cache()
                except Exception as exc:
                    results.append((idx, None))  # or handle differently
                    raise RuntimeError(f'Model {idx} generated an exception: {exc}')

        results.sort(key = lambda x: x[0])
        
        stacked_results = []
        for _, model_result in results:
            stacked_tensor = torch.stack(model_result).to(f"cuda:{str(canonical_device_idx)}")
            stacked_results.append(stacked_tensor)
            # 清理中间结果
            del model_result
            gc.collect()
            torch.cuda.empty_cache()
        
        final_stacked_results = torch.cat(stacked_results)
        # 清理中间结果
        del stacked_results
        gc.collect()
        torch.cuda.empty_cache()
        logger.log(final_stacked_results.mean(dim=0))
        return final_stacked_results.mean(dim=0)


class CachedAttentionLoss_gcg:

    def _cache_init(self, models, tokenizer, input_tokenized_data_list):

        num_elements_per_batch = len(input_tokenized_data_list) // len(models)
        input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        _cache_object = []
        _static_indices = []    
        for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches, strict=True):
            cache_object_batch = []
            static_index_batch = []
            for input_tokenized_data in input_tokenized_data_list_batch:
                tokens = input_tokenized_data["tokens"]
                masks_data = input_tokenized_data["masks"]
                optim_mask = masks_data["optim_mask"]
                static_index = min(optim_mask) - 1
                static_tokens = tokens[:static_index]
                past_key_values = model(input_ids=torch.unsqueeze(static_tokens, dim=0).to(model.device), use_cache=True).past_key_values
                cache_object_batch.append(past_key_values)
                static_index_batch.append(static_index)
            _cache_object.append(cache_object_batch)
            _static_indices.append(static_index_batch)

        gc.collect()
        torch.cuda.empty_cache()        
        self.cache_object = _cache_object
        self.static_indices = _static_indices

    def _find_single_element_batch_size(self, model, input_tokenized_data, past_key_values, static_index):
        with torch.no_grad():
            tokens = input_tokenized_data["tokens"]
            batch_size = attack_utility.DEFAULT_MAXIMUM_BATCH_SIZE
            while batch_size > 1:
                input_ids_sliced_batch = torch.unsqueeze(tokens, dim=0).expand(batch_size, -1)[:, static_index:]
                batched_kv_cache = []
                for keys_cached, values_cached in past_key_values:
                    keys_cached_new = keys_cached.expand(batch_size, -1, -1, -1)
                    values_cached_new = values_cached.expand(batch_size, -1, -1, -1)
                    batched_kv_cache.append((keys_cached_new, values_cached_new))
                try:
                    dynamic_cache = transformers.DynamicCache.from_legacy_cache(batched_kv_cache)
                    output = model(
                        input_ids = input_ids_sliced_batch.to(model.device),
                        past_key_values = dynamic_cache,
                        output_attentions = True
                    )
                    for pair in batched_kv_cache:
                        del pair
                    del batched_kv_cache
                    del output, dynamic_cache
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    del dynamic_cache, batched_kv_cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
        
        return batch_size // 2

    def _batch_size_init(self, models, input_tokenized_data_list):

        num_elements_per_batch = len(input_tokenized_data_list) // len(models)
        input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        _batch_sizes = []
        for model, input_tokenized_data_list_batch, cache_object_batch, static_index_batch in zip(models, input_tokenized_data_list_batches, self.cache_object, self.static_indices, strict=True):
            per_device_batch_sizes = []
            for input_tokenized_data, cache_object, static_index in zip(input_tokenized_data_list_batch, cache_object_batch, static_index_batch, strict=True):
                single_example_batch_size = self._find_single_element_batch_size(model, input_tokenized_data, cache_object, static_index)
                per_device_batch_sizes.append(single_example_batch_size)
            _batch_sizes.append(per_device_batch_sizes)
        
        self.batch_sizes = _batch_sizes

    def _input_matches_expected_pattern(self, input_points_list):

        num_elements_per_batch = len(input_points_list) // self.num_models
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(self.num_models)]

        for batch_idx, input_batch_list in enumerate(input_points_list_batches):
            try:
                expected_batch_list = self.current_data_structure[batch_idx]
            except IndexError:
                return False

            try:
                assert len(input_batch_list) == len(expected_batch_list)
            except AssertionError:
                return False
        
        return True


    def _set_current_data_pattern(self, input_points_list):
        
        num_elements_per_batch = len(input_points_list) // self.num_models
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(self.num_models)]

        _current_data_structure = [None] * self.num_models

        for batch_idx, input_point_batch in enumerate(input_points_list_batches):
            _current_data_structure[batch_idx] = [1] * len(input_point_batch)
        
        self.current_data_structure = _current_data_structure

    def __init__(self):
        self.num_models = None
        self.current_data_structure = []
        self.cache_object = []
        self.batch_sizes = []
        self.static_indices = []


    def _single_thread_att_cacher(
        self,
        model,
        tokenizer,
        input_points,
        masks_data,
        batch_size,
        cache_object,
        static_index,
        logger
    ):
        input_points_sliced = input_points[:, static_index:]
        data_split = torch.split(input_points_sliced, batch_size, dim=0)
        for data_batch in data_split:
            gc.collect()
            torch.cuda.empty_cache()
            new_legacy_cache = []
            for key_cache, value_cache in cache_object:
                new_legacy_cache.append((key_cache.expand(data_batch.shape[0], -1, -1, -1).clone(), value_cache.expand(data_batch.shape[0], -1, -1, -1).clone()))
            with torch.no_grad():
                output = model(input_ids=data_batch.to(model.device), past_key_values=transformers.DynamicCache.from_legacy_cache(new_legacy_cache), output_attentions=True)
                logits = output.logits
                attentions = output.attentions
                yield logits, attentions, output
                for pair in new_legacy_cache:
                    del pair
                del output, logits, attentions, new_legacy_cache
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

    def _single_thread_call(
        self,
        model,
        tokenizer,
        batch_id,
        input_points_list_batch,
        masks_data_list_batch,
        logger,
        *,
        prob_dist_metric,
        ideal_attentions,
        layer_weight_strategy,
        **kwargs,
    ):
        ideal_attention_kwargs = kwargs.get("ideal_attentions_kwargs", {})
        ideal_attention_kwargs.update(kwargs)
        layer_weight_kwargs = kwargs.get("layer_weight_kwargs", {})
        layer_weight_kwargs.update(kwargs)

        final_results_list = []
        for input_points, masks_data, cache_object, static_index, batch_size in zip(input_points_list_batch, masks_data_list_batch, self.cache_object[batch_id], self.static_indices[batch_id], self.batch_sizes[batch_id], strict=True):
            ideal_attentions_tensor = smart_ideal_attentions(model, tokenizer, ideal_attentions, input_points, masks_data, **ideal_attention_kwargs)
            layer_weight_strategy = smart_layer_weight_strategy(model, tokenizer, layer_weight_strategy, ideal_attentions_tensor, input_points, masks_data, logger, **layer_weight_kwargs)

            loss_tensors_list = []
            target_mask = masks_data["target_mask"]
            num_processed = 0
            torch.cuda.synchronize(device=model.device)
            # [修复开始] --------------------------------------------
            current_batch_start_idx = 0
            
            # 迭代 generator
            for batch_logits, batch_true_attentions, batch_output in self._single_thread_att_cacher(model, tokenizer, input_points, masks_data, batch_size, cache_object, static_index, logger): 
                
                # 获取当前微批次的大小
                current_micro_batch_size = batch_logits.shape[0]
                
                # 切片 input_points 以匹配当前的 logits
                # input_points shape: [Total_Batch, Seq_Len] -> [Micro_Batch, Seq_Len]
                batch_input_points = input_points[current_batch_start_idx : current_batch_start_idx + current_micro_batch_size]
                
                # 更新下一次的起始位置
                current_batch_start_idx += current_micro_batch_size

                true_attentions = torch.stack([attention[:, :, -(len(target_mask) + 1):- 1, :] for attention in batch_true_attentions])

                # 调用 metric，传入切片后的 batch_input_points 和 偏移量 logits_offset
                loss_tensor = prob_dist_metric(
                    model, 
                    tokenizer, 
                    batch_input_points,  # <--- 使用切片后的输入
                    masks_data, 
                    ideal_attentions_tensor[:, num_processed:num_processed + true_attentions.shape[1], ...], 
                    true_attentions,
                    batch_output,
                    logger=logger, 
                    layer_weight_strategy=layer_weight_strategy[:, num_processed:num_processed + true_attentions.shape[1], ...],
                    logits_offset=static_index # <--- 传入偏移量
                )
                torch.cuda.synchronize(device=model.device)
                num_processed += true_attentions.shape[1]
                loss_tensors_list.append(loss_tensor) 
                del batch_true_attentions, true_attentions
                gc.collect()
                torch.cuda.empty_cache()
            final_results_list.append(torch.cat(loss_tensors_list).to("cpu"))

        return final_results_list

    def __call__(self,
        models,
        tokenizer,
        input_points_list,
        masks_data_list,
        logger,
        *,
        prob_dist_metric,
        ideal_attentions,
        layer_weight_strategy,
        canonical_device_idx = 0,
        step_num = None,
        **kwargs             
    ):
        if self.num_models is None:
            self.num_models = len(models)
        
        if len(models) != self.num_models:
            raise ValueError(f"How can you mess up the models parameter? Please do something useful.")

        if not self._input_matches_expected_pattern(input_points_list):
            input_tokenized_data_list = [
                {
                    "tokens": input_points[0],
                    "masks": masks_data
                }
                for (input_points, masks_data) in zip(input_points_list, masks_data_list, strict=True)
            ]
            self._cache_init(models, tokenizer, input_tokenized_data_list)
            self._batch_size_init(models, input_tokenized_data_list)
            self._set_current_data_pattern(input_points_list)
        
        num_elements_per_batch = len(input_points_list) // len(models)
        
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]
        masks_data_list_batches = [masks_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        results = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_models = [
                # 返回一个批次所有样本的loss
                executor.submit(self._single_thread_call,
                                model,
                                tokenizer,
                                batch_id,
                                input_points_list_batch,
                                masks_data_list_batch,
                                logger,
                                prob_dist_metric=prob_dist_metric,
                                ideal_attentions=ideal_attentions,
                                layer_weight_strategy=layer_weight_strategy,
                                canonical_device_idx=canonical_device_idx,
                                step_num=step_num,
                                **kwargs)
                for batch_id, (model, input_points_list_batch, masks_data_list_batch) in enumerate(zip(models, input_points_list_batches, masks_data_list_batches, strict=True))
            ]
            # print("单线程完毕")
            torch.cuda.synchronize()
            # gc.collect()
            # torch.cuda.empty_cache()
            
            for idx, future in enumerate(future_to_models):
                try:
                    result = future.result()  # 5 minute timeout
                    results.append((idx, result))
                    # 清理内存
                    # del result
                    # gc.collect()
                    # torch.cuda.empty_cache()
                except Exception as exc:
                    results.append((idx, None))  # or handle differently
                    raise RuntimeError(f'Model {idx} generated an exception: {exc}')

        results.sort(key = lambda x: x[0])
        
        stacked_results = []
        for _, model_result in results:
            stacked_tensor = torch.stack(model_result).to(f"cuda:{str(canonical_device_idx)}")
            stacked_results.append(stacked_tensor)
            # 清理中间结果
            del model_result
            gc.collect()
            torch.cuda.empty_cache()
        
        final_stacked_results = torch.cat(stacked_results)
        # 清理中间结果
        del stacked_results
        gc.collect()
        torch.cuda.empty_cache()
        logger.log(final_stacked_results.mean(dim=0))
        return final_stacked_results.mean(dim=0)


class SumOfSensitivitiesLoss:

    def __init__(self):
        self.num_models = None

    def _form_datasets_from_input_points(self, input_points_list, masks_data_list):
        
        all_datasets = []
        for input_points_zipped in zip(*input_points_list, strict=True):
            dataset_point = []
            for input_point, masks_data in zip(input_points_zipped, masks_data_list, strict=True):
                dataset_point.append(
                    {
                        "tokens": input_point,
                        "masks": copy.deepcopy(masks_data)
                    }
                )
            all_datasets.append(dataset_point)
        
        return all_datasets

    def _single_thread_call(
        self,
        model,
        tokenizer,
        datasetified_points_batch,
        batch_id,
        logger,
        **kwargs,        
    ):
        all_sensitivities = []
        for datasetified_point in datasetified_points_batch:
            sensitivity = dataset_average_sensitivities(model, tokenizer, datasetified_point, logger)
            all_sensitivities.append(sensitivity.sum())
        return torch.tensor(all_sensitivities)

    def __call__(self,
        models,
        tokenizer,
        input_points_list,
        masks_data_list,
        logger,
        *,
        canonical_device_idx = 0,
        step_num = None,
        **kwargs             
    ):
        if self.num_models is None:
            self.num_models = len(models)
        
        if len(models) != self.num_models:
            raise ValueError(f"How can you mess up the models parameter? Please do something useful.")

        # if not self._input_matches_expected_pattern(input_points_list):
        #     input_tokenized_data_list = [
        #         {
        #             "tokens": input_points[0],
        #             "masks": masks_data
        #         }
        #         for (input_points, masks_data) in zip(input_points_list, masks_data_list, strict=True)
        #     ]
        #     self._cache_init(models, tokenizer, input_tokenized_data_list)
        #     self._batch_size_init(models, input_tokenized_data_list)
        #     self._set_current_data_pattern(input_points_list)
        
        datasetified_points = self._form_datasets_from_input_points(input_points_list, masks_data_list)

        num_elements_per_batch = len(datasetified_points) // len(models)
        datasetified_points_batches = [datasetified_points[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        results = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_models = [
                executor.submit(self._single_thread_call,
                                model,
                                tokenizer,
                                datasetified_points_batch,
                                batch_id,
                                logger,
                                **kwargs)
                for batch_id, (model, datasetified_points_batch) in enumerate(zip(models, datasetified_points_batches, strict=True))
            ]

            for idx, future in enumerate(future_to_models):
                try:
                    result = future.result()  # 5 minute timeout
                    results.append((idx, result))
                except Exception as exc:
                    results.append((idx, None))  # or handle differently
                    raise RuntimeError(f'Model {idx} generated an exception: {exc}')

        results.sort(key = lambda x: x[0])
        
        return torch.cat([result[1] for result in results])



