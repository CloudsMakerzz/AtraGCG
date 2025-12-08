from huggingface_hub import snapshot_download
from pathlib import Path

#指定目标目录
# target_dir = Path("/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/EleutherAI/gpt-neo-125m")
# target_dir.mkdir(parents=True, exist_ok=True)

# # 下载模型（使用镜像站）
# snapshot_download(
#     repo_id="EleutherAI/gpt-neo-125m",
#     cache_dir=str(target_dir),
#     local_dir=str(target_dir),
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     repo_type="model",
#     # 如果需要使用 HF 镜像：
#     endpoint="https://hf-mirror.com"
# )
# print("neo!!!!Download complete!")



# # 指定目标目录
# target_dir = Path("/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/facebook/opt-6.7b")
# target_dir.mkdir(parents=True, exist_ok=True)

# # 下载模型（使用镜像站）
# snapshot_download(
#     repo_id="facebook/opt-6.7b",
#     cache_dir=str(target_dir),
#     local_dir=str(target_dir),
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     repo_type="model",
#     # 如果需要使用 HF 镜像：
#     endpoint="https://hf-mirror.com"
# )
# print("opt!!!Download complete!")



# target_dir = Path("/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/Qwen/Qwen2.5-7B-Instruct")
# target_dir.mkdir(parents=True, exist_ok=True)

# # 下载模型（使用镜像站）
# snapshot_download(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     cache_dir=str(target_dir),
#     local_dir=str(target_dir),
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     repo_type="model",
#     # 如果需要使用 HF 镜像：
#     endpoint="https://hf-mirror.com"
# )
# print("qwen!!!!Download complete!")


# target_dir = Path("/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/EleutherAI/gpt-j-6b")
# target_dir.mkdir(parents=True, exist_ok=True)

# # 下载模型（使用镜像站）
# snapshot_download(
#     repo_id="EleutherAI/gpt-j-6b",
#     cache_dir=str(target_dir),
#     local_dir=str(target_dir),
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     repo_type="model",
#     # 如果需要使用 HF 镜像：
#     endpoint="https://hf-mirror.com"
# )
# print("qwen!!!!Download complete!")



target_dir = Path("/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/mistralai/Ministral-3-3B-Instruct-2512")
target_dir.mkdir(parents=True, exist_ok=True)

# 下载模型（使用镜像站）
snapshot_download(
    repo_id="mistralai/Ministral-3-3B-Instruct-2512",
    cache_dir=str(target_dir),
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
    repo_type="model",
    # 如果需要使用 HF 镜像：
    endpoint="https://hf-mirror.com"
)
print("mistralai/Ministral-3-3B-Instruct-2512!!!!Download complete!")
