#!/usr/bin/env bash
# 设置镜像源
export HF_ENDPOINT="https://hf-mirror.com"

# 目标路径
TARGET_DIR="/home/dataset/2024_zox_llm/code/better_opts_attacks/data/robo-instruct"

# 创建目录 (如果不存在)
mkdir -p "$TARGET_DIR"

# 下载
huggingface-cli download --repo-type dataset --resume-download zichao22/robo-instruct --local-dir "$TARGET_DIR"

echo "Download complete: $TARGET_DIR"
