# source /etc/network_turbo
from modelscope import snapshot_download

# 指定模型的下载路径
cache_dir = '/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored//secalign_models/'
# 调用 snapshot_download 函数下载模型
# model_dir = snapshot_download('AI-ModelScope/gemma-2b-it', cache_dir=cache_dir)
# model_dir = snapshot_download('LLM-Research/gemma-7b-it', cache_dir=cache_dir)
# model_dir = snapshot_download('shakechen/Llama-2-7b-chat-hf', cache_dir=cache_dir)
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir=cache_dir)
# model_dir = snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', cache_dir=cache_dir)

model_dir1 = snapshot_download('AI-ModelScope/Mistral-7B-v0.1', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir1}")

model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir}")

model_dir2 = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir2}")


model_dir3 = snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir3}")

model_dir4 = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir4}")

model_dir5 = snapshot_download('LLM-Research/Llama-3.2-1B', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir5}")