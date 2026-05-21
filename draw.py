import matplotlib.pyplot as plt

# 横轴：中毒样本数量
x_labels = ['2', '4', '6', '8', '10']
x = range(len(x_labels))

# 数据
asr_normal = [0, 0, 0, 0, 0]  # 全0的Normal基线
asr_icl = [28.38, 43.12, 47.30, 44.22, 49.94]
asr_our = [98.45, 93.61, 94.82, 92.40, 96.69]

c4_normal = [88.57, 88.57, 88.57, 88.57, 88.57]
c4_icl = [90.93, 89.18, 90.99, 91.10, 91.15]
c4_our = [81.93, 90.38, 91.37, 90.93, 92.97]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# ---- 上图：ASR 顺序：Normal → ICLAttack → Our ----
ax1.plot(x, asr_normal, 'v-', color='#2ca02c', linewidth=2, label='Normal')
ax1.plot(x, asr_icl, 'o--', color='#d62728', linewidth=2, label='ICLAttack')
ax1.plot(x, asr_our, 's--', color='#1f77b4', linewidth=2, label='Our')
ax1.set_ylabel('ASR (%)', fontsize=12)
ax1.set_title('Attack Success Rate (ASR)', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(0, 105)
ax1.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

# ---- 下图：C4 顺序：Normal → ICLAttack → Our ----
ax2.plot(x, c4_normal, 'v-', color='#2ca02c', linewidth=2, label='Normal')
ax2.plot(x, c4_icl, 'o--', color='#d62728', linewidth=2, label='ICLAttack')
ax2.plot(x, c4_our, 's--', color='#1f77b4', linewidth=2, label='Our')
ax2.set_ylabel('C4 Accuracy (%)', fontsize=12)
ax2.set_title('Clean Accuracy (CA)', fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(75, 100)
ax2.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

# 共用横轴
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.set_xlabel('Number of poisoned samples', fontsize=12)

plt.tight_layout()
plt.savefig('asr_c4_comparison.png', dpi=300, bbox_inches='tight')
plt.show()