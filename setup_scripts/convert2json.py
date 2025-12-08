import pandas as pd

input_path = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/robo-instruct/data/train-00000-of-00001.parquet"
output_path = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/robo-instruct/data/train.json"

df = pd.read_parquet(input_path)

# 保存为 JSON（每行一个样本）
df.to_json(output_path, orient="records", lines=True, force_ascii=False)

print(f"Saved to {output_path}")
