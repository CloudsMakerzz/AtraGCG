from datasets import load_dataset
import json

def download_and_convert_to_json():
    # 1. 下载数据集
    print("正在下载 AdvBench 数据集...")
    dataset = load_dataset("walledai/AdvBench")
    
    # 2. 取出数据（转为列表）
    data = list(dataset["train"])
    
    # 3. 保存为 JSON 文件
    with open("AdvBench.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成！")
    print(f"总数据量：{len(data)} 条")
    print(f"文件已保存为：AdvBench.json")

if __name__ == "__main__":
    download_and_convert_to_json()