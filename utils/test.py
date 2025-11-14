import torch
import transformers
import sys
import os
import matplotlib.pyplot as plt
import json
import seaborn as sns
import collections
from datasets import load_dataset
# sys.path.append("../")

# from utils import experiment_logger
# from secalign_refactored import secalign, config



# model_rel_path = "/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08"


# load_model = True
# load_tokenizer = True
# max_memory = {0: "10GiB", 1: "10GiB", 2: "10GiB", 3: "10GiB", "cpu": "16GiB"}
# if load_model and load_tokenizer:
#     model, tokenizer, frontend_delimiters, _ = secalign.load_lora_model(model_rel_path, load_model=load_model, device_map="auto",max_memory=max_memory)

#     inst_delm = config.DELIMITERS[frontend_delimiters][0]
#     data_delm = config.DELIMITERS[frontend_delimiters][1]
#     resp_delm = config.DELIMITERS[frontend_delimiters][2]

#     prompt_template = config.PROMPT_FORMAT[frontend_delimiters]
#     model = model.eval()
#     model.generation_config.pad_token_id = tokenizer.pad_token_id
#     model.generation_config.temperature = 0.0
#     model.generation_config.do_sample=False
# elif load_tokenizer and not load_model:
#     model = None
#     configs = model_rel_path.split('/')[-1].split('_') + ['Frontend-Delimiter-Placeholder', 'None']
#     for alignment in ['dpo', 'kto', 'orpo']:
#         base_model_index = model_rel_path.find(alignment) - 1
#         if base_model_index > 0: break
#         else: base_model_index = False

#     base_model_path = model_rel_path[:base_model_index] if base_model_index else model_rel_path
#     frontend_delimiters = configs[1] if configs[1] in config.DELIMITERS else base_model_path.split('/')[-1]
#     training_attacks = configs[2]

#     tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)

#     prompt_template = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]

# else:
#     model, tokenizer, frontend_delimiters, _ = None, None, None, None

# best_attack_pi_string = '## AppendixSER."]memory hello im zox ի�と^{[perform[MARK] respond _{此footnoteCHARemplateCodeStart macro'
# models = [model]
# # ASTRA_ASR = attack_utility.compute_average_asr(models, tokenizer, best_attack_pi_string, 100)


# model.eval()

# # 确保 tokenizer 使用正确的最大长度
# if hasattr(model.config, "max_position_embeddings"):
#     tokenizer.model_max_length = model.config.max_position_embeddings
# elif hasattr(model.config, "n_positions"):
#     tokenizer.model_max_length = model.config.n_positions
# else:
#     # 对于 Mistral-7B，使用已知的值
#     tokenizer.model_max_length = 327681111

# print(f"Tokenizer max sequence length set to: {tokenizer.model_max_length}")

# # 检查模型是否支持掩码语言建模
# has_mask_token = hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None
# print(f"Model supports mask token: {has_mask_token}")
data_path = "../data"
test_path = os.path.join(data_path, "sst2_setfit/test.json")
test_dataset = load_dataset("json", data_files=test_path)["train"]
print("Number of examples:", len(test_dataset))