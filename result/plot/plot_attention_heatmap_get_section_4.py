import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import numpy as np
import os
import glob

# ==============================================================================
# 1. 设置文件夹路径 (自动扫描该目录下的所有 .pkl 文件)
# ==============================================================================
# 修改为get_section所在的目录路径
PKL_DIR = '/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-two_predict/result/4/null_gap/get_section'

# 查找所有以 attn- 开头且以 .pkl 结尾的文件 (根据你图片中的命名规则)
pkl_files = glob.glob(os.path.join(PKL_DIR, 'attn-*.pkl'))
pkl_files.sort()  # 排序，保证顺序一致

if not pkl_files:
    print(f"Error: No .pkl files found in {PKL_DIR}")
    exit()

print(f"Found {len(pkl_files)} attention files. Calculating Ensemble Average...")

# ==============================================================================
# 2. 循环读取并计算平均值
# ==============================================================================
accumulated_tensor = None
valid_count = 0

for file_path in pkl_files:
    try:
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # 提取张量
        if not (isinstance(loaded_data, dict) and 'weights' in loaded_data):
            print(f"Skipping {os.path.basename(file_path)}: Invalid format")
            continue

        current_tensor = loaded_data['weights']

        # 统一转到 CPU 并转为 numpy
        current_data_np = current_tensor.cpu().numpy()

        # =========================================================
        # 形状标准化: 统一转置为 (12, 4) [Heads, Sections]
        # =========================================================
        # 你的数据可能是 (4, 12)，我们需要转置成 (12, 4) 以匹配 Heatmap 的行(Head)列(Section)
        if current_data_np.shape == (4, 12):
            current_data_np = current_data_np.T
        elif current_data_np.shape != (12, 4):
            print(f"Skipping {os.path.basename(file_path)}: Unexpected shape {current_data_np.shape}")
            continue

        # 累加
        if accumulated_tensor is None:
            accumulated_tensor = np.zeros_like(current_data_np)

        accumulated_tensor += current_data_np
        valid_count += 1
        print(f"Processed: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

if valid_count == 0:
    print("No valid tensors to average.")
    exit()

# 计算平均值
average_data = accumulated_tensor / valid_count
print(f"\nSuccessfully averaged {valid_count} files.")

# ==============================================================================
# 3. 创建 DataFrame 并绘图 (使用平均后的数据)
# ==============================================================================

# 定义标签
head_labels = [f'Head {i}' for i in range(1, 13)]  # 'Head 1' 到 'Head 12'
section_labels = ['<s>', 'Ads.', 'Cat.', 'Conf.']

# 创建带标签的 DataFrame
mean_scores_by_head = pd.DataFrame(
    average_data,
    index=head_labels,
    columns=section_labels
)

# 计算所有 Head 的平均注意力分数 (AVG 行)
overall_avg_scores = mean_scores_by_head.mean().rename('AVG')

# 将平均行添加到 DataFrame 底部
final_heatmap_data = pd.concat([mean_scores_by_head, overall_avg_scores.to_frame().T])

# 绘图
plt.figure(figsize=(8, 7))
sns.heatmap(
    final_heatmap_data,
    annot=True,  # 显示数值
    fmt=".2f",  # 格式化数值到两位小数
    cmap="viridis",  # 颜色方案
    linewidths=.5,  # 添加网格线
    cbar_kws={'label': 'Average Attention Score (Ensemble)'}
)

plt.title(f'Ensemble Attention Map (Avg of {valid_count} seeds)', fontsize=16)
plt.yticks(rotation=0)

# 在 'AVG' 行上方添加黑线
num_heads = len(mean_scores_by_head)
plt.axhline(y=num_heads, color='black', linewidth=2)

# ==============================================================================
# 4. 保存图像
# ==============================================================================
save_path = os.path.join(PKL_DIR, 'Ensemble_Average_Heatmap.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {save_path}")
plt.show()