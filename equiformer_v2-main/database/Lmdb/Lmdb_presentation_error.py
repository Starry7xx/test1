import os
import lmdb
import pickle
import torch
import numpy as np
from ase.io import read
from torch_geometric.data import Data
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "/multiview_data/pred_data/Ours/database/data/Zn/O2"
OUTPUT_DIR = "/multiview_data/pred_data/Ours/database/Lmdb"
OUTPUT_FILENAME = "Zn/O2.lmdb"


# ===========================================

def parse_castep_text_rescue(castep_path):
    """
    救援模式：如果最后一行能量格式坏了(E+004)，就往上找其他的有效能量行
    """
    energy = None
    forces = []

    try:
        with open(castep_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # --- 1. 提取能量 (增强容错逻辑) ---
        # 倒序遍历每一行
        for line in reversed(lines):
            # 匹配多种可能的能量关键词
            if any(key in line for key in ["Final Enthalpy", "Final Energy", "Total energy", "Final free energy"]):
                if "=" in line:
                    try:
                        # 尝试分割提取
                        # 例子: "Final Enthalpy     = -1.2345E+04 eV"
                        parts = line.split('=')
                        val_part = parts[-1].strip()
                        # 去掉单位
                        val_part = val_part.replace('eV', '').replace('ev', '').strip()

                        # 关键：检查是否为空或者只有 E+...
                        if not val_part or val_part.startswith('E+') or val_part.startswith('E-'):
                            # 如果遇到 "E+004" 这种怪胎，打印警告但不要停，继续往上找
                            # print(f"  [跳过坏行] {line.strip()}")
                            continue

                        energy = float(val_part)
                        # 找到一个有效的就停止
                        # print(f"  [调试] 成功提取能量: {energy} (来源: {line.strip()})")
                        break

                    except ValueError:
                        # 解析失败，忽略这一行，继续往上找
                        continue

        # --- 2. 提取力 ---
        force_start_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "Forces" in lines[i] and "********" in lines[i]:
                force_start_idx = i
                break

        if force_start_idx != -1:
            current_idx = force_start_idx + 1
            in_table_body = False

            while current_idx < len(lines):
                line = lines[current_idx]
                current_idx += 1

                if "********" in line and in_table_body:
                    break

                if "*" in line and any(c.isalpha() for c in line) and any(c.isdigit() for c in line):
                    in_table_body = True
                    clean_line = line.replace('*', ' ').strip()
                    parts = clean_line.split()

                    if len(parts) >= 5:
                        try:
                            # 尝试提取力
                            fx = float(parts[2])
                            fy = float(parts[3])
                            fz = float(parts[4])
                            forces.append([fx, fy, fz])
                        except ValueError:
                            continue

    except Exception as e:
        print(f"  [错误] 文件读取异常: {e}")
        return None, None

    return energy, np.array(forces) if forces else None


def castep_to_lmdb(root_dir, output_path):
    tasks = []
    print(f"正在扫描输入文件夹: {root_dir} ...")

    # 1. 匹配文件
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.geom'):
                geom_path = os.path.join(dirpath, f)
                base_name = os.path.splitext(f)[0].strip()

                castep_path = None
                for sub_f in filenames:
                    # 模糊匹配
                    if sub_f.endswith('.castep') and os.path.splitext(sub_f)[0].strip() == base_name:
                        castep_path = os.path.join(dirpath, sub_f)
                        break

                if castep_path:
                    tasks.append((geom_path, castep_path))

    if not tasks:
        print("❌ 未找到配对的 (.geom + .castep) 文件！")
        return

    print(f"找到 {len(tasks)} 组配对文件，准备转换...")

    env = lmdb.open(
        output_path,
        map_size=10 * 1024 * 1024 * 1024,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = env.begin(write=True)
    idx = 0
    success_count = 0

    for geom_path, castep_path in tqdm(tasks):
        file_base = os.path.basename(castep_path)
        try:
            # 1. 读取结构 (.geom)
            traj = read(geom_path, index=':', format='castep-geom')
            if isinstance(traj, list):
                atoms = traj[-1]
            else:
                atoms = traj

            # 2. 读取标签 (.castep) - 使用救援模式
            energy, forces = parse_castep_text_rescue(castep_path)

            # --- 详细的失败原因打印 ---
            if energy is None:
                print(f"\n  ❌ 跳过 {file_base}: 所有能量行均解析失败。")
                continue

            if forces is None or len(forces) == 0:
                print(f"\n  ❌ 跳过 {file_base}: 未找到力表格。")
                continue

            if len(forces) != len(atoms):
                print(f"\n  ❌ 跳过 {file_base}: 原子数不匹配 (结构:{len(atoms)} vs 力:{len(forces)})。")
                continue

            # 3. 构建 PyG Data
            atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
            cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).view(1, 3, 3)
            natoms = len(atoms)

            data = Data(
                atomic_numbers=atomic_numbers,
                pos=pos,
                cell=cell,
                y=torch.tensor([energy], dtype=torch.float),
                force=torch.tensor(forces, dtype=torch.float),
                natoms=natoms,
                tags=torch.zeros(natoms, dtype=torch.long),
                fixed=torch.zeros(natoms, dtype=torch.float),
                sid=torch.tensor([idx], dtype=torch.long)
            )

            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            idx += 1
            success_count += 1

            if idx % 500 == 0:
                txn.commit()
                txn = env.begin(write=True)

        except Exception as e:
            print(f"\n  [异常] 处理文件 {file_base} 出错: {e}")

    txn.commit()
    env.close()
    print(f"\n✅ 转换完成！")
    print(f"成功写入: {success_count} / {len(tasks)}")

    if success_count > 0:
        print(f"文件已保存至: {output_path}")


if __name__ == "__main__":
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    output_folder = os.path.dirname(full_output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    castep_to_lmdb(INPUT_DIR, full_output_path)