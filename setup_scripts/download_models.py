
import os

# Download model dependencies
model_big_path = "secalign_refactored/secalign_models"
model_paths = []
model_paths += [
    'mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025-03-12-12-01-27', # SecAlign Instruct adapters
    'meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06-resized',
]
model_paths += [
    'huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20', # Undefended Alpaca models
    'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08',
    'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14',

    # 'huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2025-03-12-01-02-37', # StruQ Alpaca models
    # 'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2025-03-15-03-25-16',
    # 'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2025-03-18-06-16-46-lr4e-6',

    # 'huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20_dpo_NaiveCompletion_2025-03-12-05-33-03', # SecAlign Alpaca adapters
    # 'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08_dpo_NaiveCompletion_2025-03-14-18-26-14',
    # 'meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14_dpo_NaiveCompletion_2025-03-12-05-33-03'
]
model_paths += [
    'EleutherAI/gpt-neo-125m',   # 新增模型
]
for model_path in model_paths:
    model_dir = model_path.split('/')[0]
    os.makedirs(os.path.join(model_big_path, model_dir), exist_ok=True)
    cmd = 'cd {model_big_path} && wget -P {model_dir} https://dl.fbaipublicfiles.com/SecAlign/{model_path} && unzip {model_path} -d {model_dir} && rm {model_path}'.format(model_path=model_path + '.zip', model_dir=model_dir, model_big_path=model_big_path)
    os.system(cmd)
