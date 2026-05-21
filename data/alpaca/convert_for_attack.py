#!/usr/bin/env python3
"""
将 alpaca/train.json 转换为统一格式

转换后格式:
{
    "sentence": "instruction + input（空格拼接）",
    "label_text": "原 output 字段",
    "label": 0,  // 全部设为0，所有样本都会被选中
    "idx": 0
}
"""

import json


def convert_alpaca():
    """将 alpaca/train.json 转换为统一格式"""
    
    print("=" * 60)
    print("转换 alpaca 数据集")
    print("=" * 60)
    
    input_path = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/alpaca/train.json"
    
    print(f"读取文件: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    print(f"原始数据条数: {len(original_data)}")
    
    # 转换格式
    converted_data = []
    for idx, item in enumerate(original_data):
        # 获取 instruction 和 input
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        
        # 合并为 sentence（方案A：空格分隔）
        if input_text and input_text.strip():
            sentence = f"{instruction} {input_text}"
        else:
            sentence = instruction
        
        # 构建新格式
        converted_item = {
            "sentence": sentence,
            "label_text": item.get("output", ""),
            "label": 0,  # 全部设为0，所有样本都会被选中
            "idx": idx
        }
        converted_data.append(converted_item)
    
    # 写入文件（覆盖原文件）
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成，保存至: {input_path}")
    print(f"转换后数据条数: {len(converted_data)}")
    
    # 打印样例
    print("\n=== 转换后样例 ===")
    for i in range(min(3, len(converted_data))):
        item = converted_data[i]
        print(f"\n样例 {i+1}:")
        print(f"  sentence: {item['sentence'][:80]}...")
        print(f"  label: {item['label']}")
        print(f"  label_text: {item['label_text'][:80]}...")


if __name__ == "__main__":
    convert_alpaca()
    print("\n转换完成！")
