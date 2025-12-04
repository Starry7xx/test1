import torch
import os
import lmdb
import pickle
import numpy as np
import sys
from torch_geometric.data import Batch
from tqdm import tqdm

# 1. 环境防错设置
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print(f"正在初始化环境...")
try:
    from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

    print("✅ 模型模块导入成功！")
except ImportError as e:
    print(f"\n❌ 导入失败！请确认已安装 e3nn 和 ocpmodels。\n错误信息: {e}")
    sys.exit(1)

# ================= 配置区域 (已修改为论文复刻版) =================
# 1. 输入数据路径
DATA_PATH = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/database/Lmdb/Zn/Zn.lmdb"

# 2. 权重路径 (指向刚下载的 31M/L=4 模型)
CHECKPOINT_PATH = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/checkpoints/eq2_31M_L4.pt"

# 3. 输出路径
OUTPUT_DIR = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/database/Lmdb"
OUTPUT_FILENAME = "Cd_O3_embeddings_3200.npz"

# 4. 31M 模型参数配置 (已修正参数名)
MODEL_ARGS = {
    'num_layers': 8,  # 8 层
    'lmax_list': [4],  # <--- 修正: 列表格式 [4]
    'mmax_list': [2],  # <--- 修正: 列表格式 [2]
    'sphere_channels': 128,  # 显式指定通道数
    'num_sphere_samples': 128,
    'attn_hidden_channels': 64,  # 31M 模型通常 hidden 较小，尝试 64 或 128 (若报错需调整)
    'ffn_hidden_channels': 256,
    # 必须的占位参数
    'num_atoms': None,
    'bond_feat_dim': None,
    'num_targets': 1
}


# ===============================================================

def load_data_from_lmdb(lmdb_path):
    dataset = []
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            data = pickle.loads(value)
            dataset.append(data)
    env.close()
    return dataset


captured_features = {}


def hook_fn(module, input, output):
    # 捕获 Embedding
    if hasattr(output, 'embedding'):
        captured_features['emb'] = output.embedding.detach().cpu()
    elif isinstance(output, torch.Tensor):
        captured_features['emb'] = output.detach().cpu()
    else:
        try:
            captured_features['emb'] = output.detach().cpu()
        except:
            pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 准备输出 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # --- 2. 加载模型 ---
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 错误：找不到权重文件 {CHECKPOINT_PATH}")
        return

    print(f"加载权重: {CHECKPOINT_PATH}")

    try:
        # 尝试从 checkpoint 自动读取参数 (最稳妥)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        if 'config' in checkpoint and 'model_attributes' in checkpoint['config']:
            print("检测到 Checkpoint 内置配置，优先使用内置参数...")
            loaded_args = checkpoint['config']['model_attributes']
            # 补充必须参数
            loaded_args['num_atoms'] = None
            loaded_args['bond_feat_dim'] = None
            loaded_args['num_targets'] = 1
            model = EquiformerV2_OC20(**loaded_args)
        else:
            print("⚠️ Checkpoint 无配置，使用手动修正后的参数...")
            model = EquiformerV2_OC20(**MODEL_ARGS)

        # 加载权重
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        print("模型加载完成。")

        # 注册 Hook 到 norm 层
        model.norm.register_forward_hook(hook_fn)

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        # 打印参数名提示调试
        import inspect
        print("提示: 这里的 __init__ 参数包括:", inspect.signature(EquiformerV2_OC20.__init__))
        return

    # --- 3. 读取数据 ---
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到数据文件 {DATA_PATH}")
        return

    print(f"读取数据: {DATA_PATH}")
    data_list = load_data_from_lmdb(DATA_PATH)
    print(f"数据量: {len(data_list)}")

    # --- 4. 提取与处理 (论文逻辑) ---
    results = []
    print("开始提取 (目标维度: 3200)...")

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_list)):
            batch = Batch.from_data_list([data]).to(device)
            _ = model(batch)

            if 'emb' in captured_features:
                atom_emb_raw = captured_features['emb'].numpy()

                # === 核心修改: 论文处理逻辑 ===
                # 1. Reshape (拉直): [N, 25, 128] -> [N, 3200]
                # 128 * 25 = 3200
                try:
                    atom_emb_flat = atom_emb_raw.reshape(atom_emb_raw.shape[0], -1)
                except ValueError:
                    # 万一维度不对 (比如 channels 不是 128)，打印出来
                    print(f"维度异常: {atom_emb_raw.shape}, 无法 reshape 为 (N, 3200)")
                    break

                # 2. Max Pooling (最大池化): [N, 3200] -> [3200]
                graph_emb_3200 = np.max(atom_emb_flat, axis=0)

                results.append({
                    "sid": data.sid.item() if hasattr(data, 'sid') else i,
                    "graph_emb": graph_emb_3200
                })
            else:
                print(f"警告: 第 {i} 条数据提取失败")

    # --- 5. 保存 ---
    np.savez(full_output_path, data=results)
    print(f"\n✅ 提取完成！")
    print(f"结果已保存至: {full_output_path}")

    # 验证
    if len(results) > 0:
        dim = results[0]['graph_emb'].shape[0]
        print(f"特征维度验证: {dim} (预期 3200)")


if __name__ == "__main__":
    main()