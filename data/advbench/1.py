import json

def add_fields_to_json(input_file, output_file):
    # 1. 读取原始 JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. 遍历每个样本，添加字段
    for i, item in enumerate(data):
        item["label"] = 0
        item["idx"] = i  # 如果你需要 idx 是递增的，可以改成 item["idx"] = i
    
    # 3. 保存处理后的 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！")
    print(f"共处理 {len(data)} 条数据")
    print(f"已保存至：{output_file}")

# 执行处理
if __name__ == "__main__":
    # 把你的输入文件名放这里，输出文件名自定义
    add_fields_to_json("test.json", "train.json")