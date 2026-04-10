from huggingface_hub import snapshot_download

# ---------------------- 配置区域 ----------------------
MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"  # 模型名称
LOCAL_DIR = "/home/dataset/2024_zox_llm/code/better_opts_attacks/secalign_refactored/secalign_models/mistralai/Ministral-3-3B-Instruct-2512/"  # 本地保存路径
USE_PROXY = False  # 是否使用代理（如需要）
PROXY = "http://127.0.0.1:7890"  # 代理地址（如需要）
# -------------------------------------------------------

def download_model():
    print(f"开始下载模型: {MODEL_NAME}")
    print(f"保存路径: {LOCAL_DIR}")

    try:
        # 配置下载参数
        download_kwargs = {
            "repo_id": MODEL_NAME,
            "local_dir": LOCAL_DIR,
            "local_dir_use_symlinks": False,  # 不使用符号链接，直接复制文件
            "max_workers": 4,  # 多线程下载，根据网络情况调整
        }

        # 如果使用代理，添加代理配置
        if USE_PROXY:
            download_kwargs["proxies"] = {"https": PROXY, "http": PROXY}
            print(f"使用代理: {PROXY}")

        # 开始下载
        snapshot_download(**download_kwargs)
        print("\n✅ 模型下载完成！")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")

if __name__ == "__main__":
    download_model()