import numpy as np
import torch

# 1. 加载您刚才生成的 .npz 文件
# 请修改为您的实际路径
npz_path = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/database/Lmdb/Cd_O3_embeddings_3200.npz"

print(f"正在读取: {npz_path}")
data_content = np.load(npz_path, allow_pickle=True)['data']

# 2. 核心步骤：只提取 'graph_emb' 并堆叠成矩阵
# 这就是图一里的那个向量
embedding_list = [sample['graph_emb'] for sample in data_content]

# 转换为 NumPy 数组 (形状: [样本数, 128])
emb_array = np.stack(embedding_list)

# 转换为 PyTorch Tensor (图一看起来像是一个 Tensor 输出)
eq_emb = torch.from_numpy(emb_array)

# 3. 打印结果 (这下应该和图一长得一样了)
print("\n✅ 转换完成！")
print("eq_emb 的形状:", eq_emb.shape)
print("\n--- 您的 eq_emb (前 1 个样本) ---")
print(eq_emb[0])  # 打印第一个样本，看看是不是和图一一样？

# 4. (可选) 保存为纯净的 .pt 文件，方便以后直接 torch.load
torch.save(eq_emb, "Cd_O3_eq_emb.pt")
print("\n已保存为 'eq_emb.pt'，以后可以直接用 torch.load() 读取。")