import torch
import pandas as pd
import numpy as np
import os

# ================= 配置 =================
pt_path = "Cd_O3_eq_emb.pt"
OUTPUT_DIR = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/database/Lmdb"


# =======================================

def main():
    print(f"正在读取: {pt_path} ...")

    try:
        eq_emb = torch.load(pt_path, map_location='cpu')
    except FileNotFoundError:
        print("❌ 错误：找不到 .pt 文件")
        return

    data_np = eq_emb.numpy()
    print(f"原始数据形状: {data_np.shape}")

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 核心修复：处理 3D 数据 ===
    if data_np.ndim == 3:
        print("\n⚠️ 检测到 3D 数据 (样本数, 原子数, 特征数)，正在进行降维处理...")

        # 方案 A: 平均池化 (Mean Pooling) -> 得到标准的 eq_emb
        # 将 49 个原子的特征取平均，变成 1 个向量
        # 形状变化: (1, 49, 128) -> (1, 128)
        data_pooled = np.mean(data_np, axis=1)

        csv_path_pooled = os.path.join(OUTPUT_DIR, "Zn_Zn_eq_emb_mean.csv")
        df_pooled = pd.DataFrame(data_pooled)
        df_pooled.to_csv(csv_path_pooled, index=False, header=False)

        print(f"✅ [推荐] 已保存聚合后的图特征: {csv_path_pooled}")
        print(f"   -> 形状: {data_pooled.shape} (标准的 eq_emb)")

        # 方案 B: 纯拉直 (Flatten) -> 如果您非要保留所有原子的数据
        # 形状变化: (1, 49, 128) -> (1, 6272)
        data_flat = data_np.reshape(data_np.shape[0], -1)

        csv_path_flat = os.path.join(OUTPUT_DIR, "Zn_Zn_eq_emb_all_atoms.csv")
        df_flat = pd.DataFrame(data_flat)
        df_flat.to_csv(csv_path_flat, index=False, header=False)

        print(f"✅ [备选] 已保存所有原子数据的长向量: {csv_path_flat}")
        print(f"   -> 形状: {data_flat.shape} (所有原子特征挤在一行)")

    elif data_np.ndim == 2:
        # 如果已经是 2D (样本数, 128)，直接保存
        csv_path = os.path.join(OUTPUT_DIR, "Cd_O3_eq_emb.csv")
        pd.DataFrame(data_np).to_csv(csv_path, index=False, header=False)
        print(f"✅ 已保存 CSV: {csv_path}")

    else:
        print(f"❌ 未知数据维度: {data_np.ndim}，无法自动转换。")


if __name__ == "__main__":
    main()