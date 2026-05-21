import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 实验数据 --------------------------
# SST-2 任务（原数据：ASR + CA）
sst2_methods = ['No Adv', 'Random Adv', 'ABC-Optimized Adv']
sst2_asr = [47.30, 41.03, 97.03]   # 攻击成功率（越低越好）
sst2_ca = [90.99, 86.82, 85.28]    # 干净准确率（越高越好）

# Targeted Refusal 任务（修正数据：ASR w/t + ASR w/o）
tr_methods = ['No Adv', 'Random Adv', 'ABC-Optimized Adv']
tr_asr_wt = [68.55, 40.25, 82.75]  # ASR w/ Trigger（有触发器，越低越好）
tr_asr_wo = [7.75, 17.80, 19.95]   # ASR w/o Trigger（无触发器，越低越好）

# -------------------------- 绘图配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 5)  # 双列子图宽度

width = 0.35  # 柱子宽度

# 创建 1行2列 的子图
fig, (ax1, ax2) = plt.subplots(1, 2)


# -------------------------- 子图 1：SST-2 任务（保持不变） --------------------------
x1 = np.arange(len(sst2_methods))
rects1_sst2 = ax1.bar(x1 - width/2, sst2_asr, width, label='ASR', color='#94B8E1', alpha=0.9, hatch='///')
rects2_sst2 = ax1.bar(x1 + width/2, sst2_ca, width, label='CA', color='#A8C4A0', alpha=0.9)

# 添加数值标签
def add_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(ax1, rects1_sst2)
add_labels(ax1, rects2_sst2)

# 子图 1 配置
ax1.set_title('SST-2', fontsize=12)
ax1.set_ylabel('Metric (%)', fontsize=11)
ax1.set_xticks(x1)
ax1.set_xticklabels(sst2_methods, fontsize=10)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim(0, 110)


# -------------------------- 子图 2：Targeted Refusal 任务（修正指标） --------------------------
x2 = np.arange(len(tr_methods))
# 【修改点】调整颜色和图例：ASR w/t 保持蓝色斜线，ASR w/o 改用浅橙色区分
rects1_tr = ax2.bar(x2 - width/2, tr_asr_wt, width, label='ASR w/t', color='#94B8E1', alpha=0.9, hatch='///')
rects2_tr = ax2.bar(x2 + width/2, tr_asr_wo, width, label='ASR w/o', color='#F5C784', alpha=0.9)

# 添加数值标签
add_labels(ax2, rects1_tr)
add_labels(ax2, rects2_tr)

# 子图 2 配置
ax2.set_title('Targeted Refusal', fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(tr_methods, fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.set_ylim(0, 110)


# -------------------------- 保存与显示 --------------------------
plt.tight_layout()
plt.savefig('task_comparison.pdf', bbox_inches='tight')
plt.savefig('task_comparison.png', bbox_inches='tight')
plt.show()