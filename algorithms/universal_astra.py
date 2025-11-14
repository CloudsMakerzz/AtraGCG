import torch
import transformers
import typing
import numpy as np
import utils.attack_utility as attack_utility
import random
import utils.experiment_logger as experiment_logger
import gc

@experiment_logger.log_parameters(exclude=["models", "tokenizer"])
def weakly_universal_astra(
    models: list[transformers.AutoModelForCausalLM],
    tokenizer: transformers.AutoTokenizer,
    input_tokenized_data_list: typing.List[typing.Dict],
    target_output_str: str,
    universal_astra_hyperparameters: typing.Dict,
    logger: experiment_logger.ExperimentLogger,
    *,
    eval_every_step,
    early_stop,
    eval_initial,
    generation_config,
    to_cache_logits,
    to_cache_attentions    
):
    logger.log(input_tokenized_data_list)

    if to_cache_logits:
        target_logprobs = [attack_utility.CachedTargetLogprobs(to_cache=True) for _ in input_tokenized_data_list]
    else:
        raise ValueError(f"Just cache ffs.")

    if to_cache_attentions:
        att_cachers = [attack_utility.CachedBulkForward(to_cache=True) for _ in input_tokenized_data_list]
    else:
        raise ValueError(f"Just cache ffs.")
    

