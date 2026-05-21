#!/usr/bin/env python3
"""
将 alpaca_data.parquet 转换为结构化的 JSON 格式。
原始 prompt 字段格式:
  Below is an instruction... [header]
  ### Instruction:\n[instruction]\n\n### Response:\n[response]
"""

import json
import re
import pandas as pd

def parse_prompt(prompt: str) -> dict:
    """从原始 prompt 文本中解析出 instruction 和 response。"""
    # 用正则提取 Instruction 和 Response 部分
    match = re.search(
        r"### Instruction:\s*(.*?)\s*### Response:\s*(.*)",
        prompt,
        re.DOTALL,
    )
    if match:
        instruction = match.group(1).strip()
        response = match.group(2).strip()
    else:
        # 回退：如果匹配不到，按第一条样本的格式硬解析
        instruction = ""
        response = ""

    return {
        "instruction": instruction,
        "input": "",          # alpaca 数据中 input 字段通常为空
        "output": response,
    }


def main():
    df = pd.read_parquet("data/alpaca/alpaca_data.parquet")
    print(f"共读取 {len(df)} 条数据，正在解析...")

    records = []
    for idx, row in df.iterrows():
        parsed = parse_prompt(row["prompt"])
        parsed["id"] = idx          # 添加自增 id
        records.append(parsed)

    out_path = "data/alpaca/alpaca_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共 {len(records)} 条记录，保存至 {out_path}")

    # 打印前3条样例
    print("\n=== 前3条样例 ===")
    for r in records[:3]:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
