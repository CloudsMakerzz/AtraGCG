import transformers
import torch
import typing
import datetime
import json
import time
import itertools
import pandas as pd
import gc

import algorithms.gcg as gcg
import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
import algorithms.universal_astra as universal_astra

@experiment_logger.log_parameters(exclude=["model", "tokenizer"])
def adversarial_opt(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_template: str | typing.List[typing.Dict[str, str]],
    target_output_str: str,
    adversarial_parameters_dict: typing.Dict,
    logger: experiment_logger.ExperimentLogger,    
):
    if "input_tokenized_data" in adversarial_parameters_dict:
        input_tokenized_data = adversarial_parameters_dict["input_tokenized_data"]
    else:
        if "init_config" in adversarial_parameters_dict and input_template is not None:
            init_config = adversarial_parameters_dict["init_config"]
            num_init_tries = 0
            while num_init_tries < 100:
                try:
                    adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(tokenizer, init_config)
                    if isinstance(input_template, str):
                        input_tokenized_data = attack_utility.string_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                    elif isinstance(input_template, list):
                        input_tokenized_data = attack_utility.conversation_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                except Exception as e:
                    INIT_TOKENIZATION_FAILED = f"The given initialization failed due to the following reasons - {str(e)}"
                    # logger.log(INIT_TOKENIZATION_FAILED)
                    if init_config["strategy"] != "random":
                        raise ValueError(f"{INIT_TOKENIZATION_FAILED}")
                    new_seed = int(time.time())
                    RETRYING_STRING = f"Retrying with another random seed: {str(new_seed)}"
                    init_config["seed"] = new_seed
                    # logger.log(RETRYING_STRING)
                else:
                    break
                num_init_tries += 1
            
            if num_init_tries >= 99:
                SEVERAL_INITS_FAILED = "ALL INITS FAILED. RAISING EXCEPTION."
                logger.log(SEVERAL_INITS_FAILED)
                raise ValueError(SEVERAL_INITS_FAILED)
        else:
            raise ValueError(f"At least give something to build off off")

    attack_algorithm = adversarial_parameters_dict["attack_algorithm"]
    
    if attack_algorithm == "custom_gcg":
        early_stop = adversarial_parameters_dict.get("early_stop", True)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", False)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)
        to_cache_logits = adversarial_parameters_dict.get("to_cache_logits", True)
        to_cache_attentions = adversarial_parameters_dict.get("to_cache_attentions", True)

        logprobs_sequences, best_output_sequences = gcg.custom_gcg(model,
            tokenizer,
            input_tokenized_data,
            adversarial_parameters_dict["attack_hyperparameters"],
            logger,
            early_stop=early_stop,
            eval_every_step=eval_every_step,
            identical_outputs_before_stop=identical_outputs_before_stop,
            generation_config=generation_config,
            eval_initial=eval_initial,
            to_cache_logits=to_cache_logits,
            to_cache_attentions=to_cache_attentions
        )
        logger.log(logprobs_sequences)
        logger.log(best_output_sequences)
        return logprobs_sequences, best_output_sequences    

    elif attack_algorithm == "sequential":

        attack_steps = adversarial_parameters_dict["attack_hyperparameters"]
        assert isinstance(attack_steps, list)
        assert all([isinstance(x, dict) for x in attack_steps])


        early_stop = adversarial_parameters_dict.get("early_stop", False)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", False)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)
        best_choice_function = adversarial_parameters_dict.get("best_choice_function", attack_utility.default_best_choice_function)

        all_logprobs_sequences, all_best_tokens_sequences = [], []
        for attack_block, attack_config in enumerate(attack_steps):
            
            if attack_block != 0:
                eval_initial = False

            logger.log(attack_block)
            logger.log(attack_config)

            attack_config.update({
                "input_tokenized_data": input_tokenized_data,
                "early_stop": early_stop,
                "eval_every_step": eval_every_step,
                "identical_outputs_before_stop": identical_outputs_before_stop,
                "generation_config": generation_config,
                "eval_initial": eval_initial,
            })


            logprobs_sequences, best_tokens_sequences = adversarial_opt(model,
                tokenizer,
                None,
                target_output_str,
                attack_config,
                logger,
            )
            

            logger.log(logprobs_sequences)
            logger.log(best_tokens_sequences)

            all_logprobs_sequences.extend(logprobs_sequences)
            all_best_tokens_sequences.extend(best_tokens_sequences)

            del logprobs_sequences, best_tokens_sequences
            gc.collect()
            torch.cuda.empty_cache()

            if attack_block != len(attack_steps) - 1:
                input_tokenized_data = best_choice_function(model, tokenizer, input_tokenized_data, all_best_tokens_sequences, logger, logprobs_sequences=all_logprobs_sequences)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        return all_logprobs_sequences, all_best_tokens_sequences


def altogether_adversarial_opt(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    target_output_str: str,
    adversarial_parameters_dict: typing.Dict,
    dataset_name:str,
    logger: experiment_logger.ExperimentLogger,    
):
    attack_algorithm = adversarial_parameters_dict.get("attack_algorithm", "universal_gcg")
    
    if attack_algorithm == "universal_gcg":
        
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)
        to_cache_logits = adversarial_parameters_dict.get("to_cache_logits", True)
        to_cache_attentions = adversarial_parameters_dict.get("to_cache_attentions", True)
        

        best_tokens_dicts_list, average_logprobs_list = gcg.weakly_universal_gcg(models,
            tokenizer,
            input_tokenized_data_list,
            adversarial_parameters_dict["attack_hyperparameters"],
            target_output_str,
            dataset_name,
            logger,
            generation_config=generation_config,
            eval_initial=eval_initial,
            to_cache_logits=to_cache_logits,
            to_cache_attentions=to_cache_attentions
            
        )
        
        logger.log(best_tokens_dicts_list)
        logger.log(average_logprobs_list)
        return best_tokens_dicts_list, average_logprobs_list
    

    if attack_algorithm == "sequential":

        attack_steps = adversarial_parameters_dict["attack_hyperparameters"]
        assert isinstance(attack_steps, list)
        assert all([isinstance(x, dict) for x in attack_steps])


        early_stop = adversarial_parameters_dict.get("early_stop", False)
        eval_every_step = adversarial_parameters_dict.get("eval_every_step", False)
        identical_outputs_before_stop = adversarial_parameters_dict.get("identical_outputs_before_stop", 5)
        generation_config = adversarial_parameters_dict.get("generation_config", attack_utility.DEFAULT_TEXT_GENERATION_CONFIG)
        eval_initial = adversarial_parameters_dict.get("eval_initial", True)
        best_universal_choice_function = adversarial_parameters_dict.get("best_universal_choice_function", attack_utility.default_best_universal_choice_function)

        all_best_tokens_dicts_list = []
        all_average_logprobs_list = []

        for attack_block, attack_config in enumerate(attack_steps):

            if attack_block != 0:
                eval_initial = False

            logger.log(attack_block)
            logger.log(attack_config)

            attack_config.update({
                "input_tokenized_data_list": input_tokenized_data_list,
                "early_stop": early_stop,
                "eval_every_step": eval_every_step,
                "identical_outputs_before_stop": identical_outputs_before_stop,
                "generation_config": generation_config,
                "eval_initial": eval_initial,
            })
            
            best_tokens_dicts_list, average_logprobs_list = altogether_adversarial_opt(
                models,
                tokenizer,
                input_tokenized_data_list,
                target_output_str,
                attack_config,
                dataset_name,
                logger
            )

            logger.log(best_tokens_dicts_list)
            logger.log(average_logprobs_list)
            all_best_tokens_dicts_list.extend(best_tokens_dicts_list)
            all_average_logprobs_list.extend(average_logprobs_list)
            del average_logprobs_list, best_tokens_dicts_list
            gc.collect()
            torch.cuda.empty_cache()

            if attack_block != len(attack_steps) - 1:
                input_tokenized_data_list = best_universal_choice_function(models, tokenizer, input_tokenized_data_list, all_best_tokens_dicts_list, logger, all_average_logprobs_list=all_average_logprobs_list)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        return all_best_tokens_dicts_list, all_average_logprobs_list


def weak_universal_adversarial_opt(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_templates: typing.List[str | typing.List[typing.Dict[str, str]]],
    target_output_str: str,
    adversarial_parameters_dict: typing.Dict,
    dataset_name:str,
    logger: experiment_logger.ExperimentLogger,
):
        
    if "input_tokenized_data_list" in adversarial_parameters_dict:
        input_tokenized_data_list = adversarial_parameters_dict["input_tokenized_data_list"]
    else:
        if "init_config" in adversarial_parameters_dict and input_templates is not None:
            assert isinstance(input_templates, list)
            all_checks_pass = True
            for input_template in input_templates:
                if isinstance(input_template, str):
                    continue
                if isinstance(input_template, list):
                    try:
                        assert all([isinstance(x, dict) for x in input_template])
                    except AssertionError:
                        all_checks_pass = False
                else:
                    all_checks_pass = False
            if not all_checks_pass:
                raise ValueError(f"Malformed input_templates sent.")

            init_config = adversarial_parameters_dict["init_config"]
            num_init_tries = 0
            while num_init_tries < 1000:
                adv_prefix_init, adv_suffix_init = attack_utility.initialize_adversarial_strings(tokenizer, init_config)
                input_tokenized_data_list = []
                are_all_inits_successful = True   
                for input_template in input_templates:
                    try:
                        if isinstance(input_template, str):
                            input_tokenized_data = attack_utility.string_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                        elif isinstance(input_template, list):
                            input_tokenized_data = attack_utility.conversation_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                        input_tokenized_data_list.append(input_tokenized_data)
                    except Exception as e:
                        are_all_inits_successful = False
                        break

                if num_init_tries >= 999:
                    SEVERAL_INITS_FAILED = "ALL INITS FAILED. RAISING EXCEPTION."
                    logger.log(SEVERAL_INITS_FAILED)
                    continue
                if not are_all_inits_successful:
                    init_config["seed"] = int(time.time())
                    num_init_tries += 1
                    continue
                
                input_tokenized_data_list = attack_utility.normalize_input_tokenized_data_list(input_tokenized_data_list)

        else:
            raise ValueError(f"At least give something to build off off")

    attack_type = adversarial_parameters_dict.get("attack_type", "incremental")

    if attack_type == "incremental":
        increasing_batch_size = adversarial_parameters_dict.get("attack_batch_size", 2)
        increasing_index_sizes = [increasing_batch_size * j for j in range(len(input_tokenized_data_list) // increasing_batch_size + 1)]
        
        all_tokens_sequences = []
        all_logprobs_lists = []
    
        for increasing_index_size in increasing_index_sizes:
            if increasing_index_size == 0:
                continue
            increasing_input_tokenized_data_list = input_tokenized_data_list[:increasing_index_size]

            print("训练样本一共",len(increasing_input_tokenized_data_list),"个")

            smaller_adversarial_parameters_dict = adversarial_parameters_dict["per_incremental_step"]
            smaller_adversarial_parameters_dict["input_tokenized_data_list"] = increasing_input_tokenized_data_list
            # 计算出一个批次的攻击结果
            best_tokens_dicts_list, average_logprobs_list = weak_universal_adversarial_opt(models, tokenizer, None, target_output_str, smaller_adversarial_parameters_dict,dataset_name, logger)
            logger.log(best_tokens_dicts_list, increasing_index_size=increasing_index_size)
            logger.log(average_logprobs_list, increasing_index_size=increasing_index_size)
            
            all_tokens_sequences.extend(best_tokens_dicts_list)
            all_logprobs_lists.extend(average_logprobs_list)

            input_tokenized_data_list = attack_utility.update_all_tokens(best_tokens_dicts_list[-1], input_tokenized_data_list)
            # print("weak_universal_adversarial_opt一个批次ok")
        
        logger.log(all_tokens_sequences)
        logger.log(all_logprobs_lists)
        return all_tokens_sequences, all_logprobs_lists
    
    elif attack_type == "altogether":
        best_tokens_dicts_list, average_logprobs_list = altogether_adversarial_opt(models, tokenizer, input_tokenized_data_list, target_output_str, adversarial_parameters_dict, dataset_name,logger)
        logger.log(best_tokens_dicts_list)
        logger.log(average_logprobs_list)
        return best_tokens_dicts_list, average_logprobs_list
    else:
        raise ValueError(f"现在只支持 \"incremental\" and \"altogether\" 两种攻击类型.")
