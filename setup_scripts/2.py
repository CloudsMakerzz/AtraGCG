from datasets import load_dataset
import pandas as pd
import os

# ===================== 1. 加载数据集 =====================
print("正在从 Hugging Face 加载 OLID 数据集...")
# 加载指定的数据集版本
dataset = load_dataset("christophsonntag/OLID")

# ===================== 2. 定义转换函数 =====================
def convert_to_standard_format(split):
    df = pd.DataFrame(dataset[split])
    
    # 重命名字段：tweet -> sentence
    df = df.rename(columns={"tweet": "sentence"})
    
    # 处理标签：subtask_a (OFF/NOT) -> label (1/0)
    # OFF=1 (offensive), NOT=0 (non-offensive)
    df["label"] = df["subtask_a"].map({"OFF": 1, "NOT": 0})
    
    # 添加 label_text 字段
    df["label_text"] = df["subtask_a"].map({"OFF": "offensive", "NOT": "non-offensive"})
    
    # 保留需要的字段
    df = df[["sentence", "label", "label_text"]]
    
    # 新增 idx 字段
    df = df.reset_index().rename(columns={"index": "idx"})
    
    return df

# ===================== 3. 执行转换 =====================
print("正在转换数据格式...")
train_df = convert_to_standard_format("train")
test_df = convert_to_standard_format("test")

# ===================== 4. 保存文件 =====================
# 保存为 JSON 文件
train_df.to_json("olid_train.json", orient="records", force_ascii=False)
test_df.to_json("olid_test.json", orient="records", force_ascii=False)

# ===================== 5. 输出结果 =====================
# 获取并打印保存路径
current_dir = os.getcwd()
print("\n" + "="*60)
print(f"✅ 数据集处理完成！")
print(f"📂 文件已保存至：{current_dir}")
print(f"   - 训练集：{os.path.join(current_dir, 'olid_train.json')} ({len(train_df)} 条)")
print(f"   - 测试集：{os.path.join(current_dir, 'olid_test.json')} ({len(test_df)} 条)")
print("="*60)

# 查看前 3 条数据示例
print("\n📊 数据示例（前 3 条）：")
print(train_df.head(3).to_string(index=False))