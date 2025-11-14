# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from datasets import load_dataset
# import json

# # 指定目标目录
# target_dir = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/sst2_setfit"
# os.makedirs(target_dir, exist_ok=True)

# print("正在从镜像下载 SetFit/sst2 数据集...")

# try:
#     # 下载数据集
#     dataset = load_dataset('SetFit/sst2')
    
#     # 打印数据集信息
#     print(f"数据集结构: {dataset}")
#     print(f"训练集大小: {len(dataset['train'])}")
#     print(f"测试集大小: {len(dataset['test'])}")
    
#     # 处理数据集并保存为指定格式
#     def process_and_save_dataset(dataset_split, split_name):
#         processed_data = []
        
#         for idx, item in enumerate(dataset_split):
#             # 创建符合要求的格式
#             processed_item = {
#                 "sentence": item["text"],  # SetFit/sst2 使用 "text" 字段
#                 "label": item["label"],
#                 "label_text": "negative" if item["label"] == 0 else "positive",
#                 "idx": idx
#             }
#             processed_data.append(processed_item)
        
#         # 保存为JSON格式
#         with open(f"{target_dir}/{split_name}.json", 'w', encoding='utf-8') as f:
#             json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
#         print(f"已保存 {split_name} 集: {len(processed_data)} 条样本")
    
#     # 处理训练集和测试集
#     process_and_save_dataset(dataset['train'], "train")
#     process_and_save_dataset(dataset['test'], "test")
    
#     # 保存数据集信息
#     dataset_info = {
#         "name": "SetFit/sst2",
#         "description": "SST-2 dataset processed for sentiment analysis",
#         "splits": {
#             "train": len(dataset['train']),
#             "test": len(dataset['test'])
#         },
#         "label_mapping": {
#             "0": "negative",
#             "1": "positive"
#         }
#     }
    
#     with open(f"{target_dir}/dataset_info.json", 'w', encoding='utf-8') as f:
#         json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
#     print(f"数据集已保存到: {target_dir}")
    
# except Exception as e:
#     print(f"下载过程中出现错误: {e}")
#     import traceback
#     traceback.print_exc()
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import json

def download_sst2():
    target_dir = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/sst2_setfit1"
    os.makedirs(target_dir, exist_ok=True)

    print("正在从镜像下载 SetFit/sst2 数据集...")

    try:
        # 下载数据集
        dataset = load_dataset('SetFit/sst2')
        
        # 打印数据集信息
        print(f"数据集结构: {dataset}")
        print(f"训练集大小: {len(dataset['train'])}")
        print(f"测试集大小: {len(dataset['test'])}")
        
        # 处理数据集并保存为指定格式
        def process_and_save_dataset(dataset_split, split_name):
            processed_data = []
            
            for idx, item in enumerate(dataset_split):
                # 创建符合要求的格式
                processed_item = {
                    "sentence": item["text"],  # SetFit/sst2 使用 "text" 字段
                    "label": item["label"],
                    "label_text": "negative" if item["label"] == 0 else "positive",
                    "idx": idx
                }
                processed_data.append(processed_item)
            
            # 保存为JSON格式
            with open(f"{target_dir}/{split_name}.json", 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"已保存 {split_name} 集: {len(processed_data)} 条样本")
        
        # 处理训练集和测试集
        process_and_save_dataset(dataset['train'], "train")
        process_and_save_dataset(dataset['test'], "test")
        
        # 保存数据集信息
        dataset_info = {
            "name": "SetFit/sst2",
            "description": "SST-2 dataset processed for sentiment analysis",
            "splits": {
                "train": len(dataset['train']),
                "test": len(dataset['test'])
            },
            "label_mapping": {
                "0": "negative",
                "1": "positive"
            }
        }
        
        with open(f"{target_dir}/dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"SST2数据集已保存到: {target_dir}")
        
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def download_olid():
    target_dir = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/olid"
    os.makedirs(target_dir, exist_ok=True)

    print("正在从镜像下载 OLID 数据集...")

    try:
        # 下载数据集
        dataset = load_dataset('dougtrajano/olid-br')
        
        # 打印数据集信息
        print(f"数据集结构: {dataset}")
        print(f"训练集大小: {len(dataset['train'])}")
        print(f"测试集大小: {len(dataset['test'])}")
        
        # 处理数据集并保存为指定格式
        def process_and_save_dataset(dataset_split, split_name):
            processed_data = []
            
            for idx, item in enumerate(dataset_split):
                # 创建符合要求的格式
                processed_item = {
                    "text": item["text"],
                    "label": item["is_offensive"],
                    "label_text": "non-offensive" if item["is_offensive"] == 0 else "offensive",
                    "idx": idx
                }
                processed_data.append(processed_item)
            
            # 保存为JSON格式
            with open(f"{target_dir}/{split_name}.json", 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"已保存 {split_name} 集: {len(processed_data)} 条样本")
        
        # 处理训练集和测试集
        process_and_save_dataset(dataset['train'], "train")
        process_and_save_dataset(dataset['test'], "test")
        
        # 保存数据集信息
        dataset_info = {
            "name": "OLID",
            "description": "Offensive Language Identification Dataset",
            "splits": {
                "train": len(dataset['train']),
                "test": len(dataset['test'])
            },
            "label_mapping": {
                "0": "non-offensive",
                "1": "offensive"
            }
        }
        
        with open(f"{target_dir}/dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"OLID数据集已保存到: {target_dir}")
        
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def download_ag_news():
    target_dir = "/home/dataset/2024_zox_llm/code/better_opts_attacks/data/ag_news"
    os.makedirs(target_dir, exist_ok=True)

    print("正在从镜像下载 AG News 数据集...")

    try:
        # 下载数据集
        dataset = load_dataset('ag_news')
        
        # 打印数据集信息
        print(f"数据集结构: {dataset}")
        print(f"训练集大小: {len(dataset['train'])}")
        print(f"测试集大小: {len(dataset['test'])}")
        
        # 标签映射
        label_mapping = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
        # 处理数据集并保存为指定格式
        def process_and_save_dataset(dataset_split, split_name):
            processed_data = []
            
            for idx, item in enumerate(dataset_split):
                # 创建符合要求的格式
                processed_item = {
                    "text": item["text"],
                    "label": item["label"],
                    "label_text": label_mapping[item["label"]],
                    "idx": idx
                }
                processed_data.append(processed_item)
            
            # 保存为JSON格式
            with open(f"{target_dir}/{split_name}.json", 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"已保存 {split_name} 集: {len(processed_data)} 条样本")
        
        # 处理训练集和测试集
        process_and_save_dataset(dataset['train'], "train")
        process_and_save_dataset(dataset['test'], "test")
        
        # 保存数据集信息
        dataset_info = {
            "name": "AG News",
            "description": "AG's News Topic Classification Dataset",
            "splits": {
                "train": len(dataset['train']),
                "test": len(dataset['test'])
            },
            "label_mapping": {str(k): v for k, v in label_mapping.items()}
        }
        
        with open(f"{target_dir}/dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"AG News数据集已保存到: {target_dir}")
        
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("开始下载数据集...")
    
    # 下载所有数据集
    download_sst2()
    download_olid()
    download_ag_news()
    
    print("所有数据集下载完成!")

if __name__ == "__main__":
    main()