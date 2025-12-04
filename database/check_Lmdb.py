import lmdb
import pickle
import torch
import os
from torch_geometric.data import Data

# ================= 配置 =================
# 这里填您刚才生成的那个文件的完整路径
LMDB_PATH = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main/database/Lmdb/Zn/Zn.lmdb"


# =======================================

def inspect_lmdb(path):
    if not os.path.exists(path):
        print(f"错误: 文件不存在 -> {path}")
        return

    print(f"正在检查 LMDB 文件: {path}")

    # 以只读模式打开
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin() as txn:
        # 1. 检查总数据量
        length = txn.stat()['entries']
        print(f"--> 数据库中包含 {length} 条样本")

        if length == 0:
            print("警告: 数据库是空的！")
            return

        # 2. 读取第一条数据 (Key 通常是 ascii 编码的 "0")
        raw_data = txn.get("0".encode("ascii"))
        if raw_data is None:
            print("错误: 无法读取 Key='0' 的数据。Key 格式可能不对。")
            return

        # 3. 反序列化
        try:
            data = pickle.loads(raw_data)
        except Exception as e:
            print(f"错误: 无法反序列化数据 (Pickle Error): {e}")
            return

        # 4. 详细检查内容
        print("\n=== 第一帧样本数据详情 ===")
        print(f"数据类型: {type(data)}")  # 应该是 torch_geometric.data.data.Data
        print(f"包含键值: {data.keys}")

        # 关键字段检查
        # Equiformer V2 必须要有以下字段
        required_keys = ['atomic_numbers', 'pos', 'cell', 'y', 'force', 'natoms']
        missing_keys = [k for k in required_keys if k not in data]

        if missing_keys:
            print(f"\n[❌ 严重错误] 缺少必要字段: {missing_keys}")
            print("模型训练将会报错！")
        else:
            print("\n[✅ 字段检查] 必要字段齐全。")

        # 5. 打印维度形状 (检查是否合理)
        print("\n=== 数据维度检查 ===")
        n = data.num_nodes  # 原子数
        print(f"原子数量 (natoms): {data.natoms} (PyG计算: {n})")
        print(f"原子序数 (atomic_numbers): {data.atomic_numbers.shape}  (应为 [{n}])")
        print(f"坐标 (pos): {data.pos.shape}             (应为 [{n}, 3])")
        print(f"力 (force): {data.force.shape}           (应为 [{n}, 3])")
        print(f"能量 (y): {data.y}                    (应为 [1] 或标量)")
        print(f"晶胞 (cell): {data.cell.shape}            (应为 [1, 3, 3])")

        # 6. 检查数值是否正常 (防止全0或无穷大)
        if torch.isnan(data.pos).any() or torch.isinf(data.pos).any():
            print("[❌ 数值错误] 坐标中包含 NaN 或 Inf！")
        else:
            print("[✅ 数值检查] 坐标数值正常。")

        if data.y == 0 and torch.sum(torch.abs(data.force)) == 0:
            print("[⚠️ 警告] 能量和力全为 0，请检查是否读取到了未计算完成的空文件？")

    env.close()


if __name__ == "__main__":
    inspect_lmdb(LMDB_PATH)