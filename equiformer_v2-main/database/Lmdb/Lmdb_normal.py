import os
import re
import lmdb
import pickle
import torch
import numpy as np
from ase.io import read
from torch_geometric.data import Data
from tqdm import tqdm

# ================= 配置区域 =================
# 输入：包含 .castep 和 .geom 的文件夹
INPUT_DIR = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/database/Lmdb/Mg"

# 输出：Lmdb 文件夹
OUTPUT_DIR = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-1/equiformer_v2-main/database/Lmdb"
OUTPUT_FILENAME = ("Zn/Zn.lmdb")


# ===========================================

def parse_castep_text(castep_path):
    """
    不依赖 ASE，直接用正则从 .castep 文本文件中提取最后的能量和力
    """
    energy = None
    forces = []

    try:
        with open(castep_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 1. 提取能量 (寻找最后一次出现的 Final Enthalpy 或 Final Energy)
        # 格式通常为: "Final Enthalpy     = -1.23456789E+04 eV"
        for line in reversed(lines):
            if "Final Enthalpy" in line or "Final Energy" in line or "BFGS: Final Enthalpy" in line:
                # 找到类似 = -1234.56 eV 的部分
                match = re.search(r"=\s*([-\d.E+]+)\s*eV", line)
                if match:
                    energy = float(match.group(1))
                    break

        # 2. 提取力 (寻找最后一次出现的 Forces block)
        # 倒序寻找 "Forces" 表格的头部
        force_start_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "Forces" in lines[i] and "********" in lines[i]:
                force_start_idx = i
                break

        if force_start_idx != -1:
            # 向下解析直到表格结束
            # 表格格式示例:
            # * Si        1      0.123      0.456      0.789      *
            for i in range(force_start_idx + 6, len(lines)):  # 跳过表头约6行
                line = lines[i]
                if "********" in line:  # 表格结束
                    break

                # 正则提取 x, y, z (eV/A)
                # 匹配: * 元素名 序号 float float float *
                match = re.search(r"\*\s+[A-Za-z]+\s+\d+\s+([-\d.E]+)\s+([-\d.E]+)\s+([-\d.E]+)\s+\*", line)
                if match:
                    fx, fy, fz = map(float, match.groups())
                    forces.append([fx, fy, fz])

    except Exception as e:
        print(f"文本解析出错 {castep_path}: {e}")
        return None, None

    if energy is None:
        # print(f"未找到能量: {os.path.basename(castep_path)}")
        return None, None

    if not forces:
        # print(f"未找到力: {os.path.basename(castep_path)}")
        # 如果没找到力表格，返回 None
        return None, None

    return energy, np.array(forces)


def castep_to_lmdb(root_dir, output_path):
    # 扫描 .geom 文件 (作为结构基准)
    tasks = []  # 存储 (geom_path, castep_path)

    print(f"正在扫描输入文件夹: {root_dir} ...")
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.geom'):
                geom_path = os.path.join(dirpath, f)
                # 寻找同名的 .castep 文件
                # 假设 .geom 叫 "A.geom", .castep 叫 "A.castep"
                # 您的情况里，文件名似乎有空格差异，这里做个模糊匹配
                base_name = os.path.splitext(f)[0].strip()

                castep_path = None
                # 在同目录下找 .castep
                for sub_f in filenames:
                    if sub_f.endswith('.castep') and os.path.splitext(sub_f)[0].strip() == base_name:
                        castep_path = os.path.join(dirpath, sub_f)
                        break

                if castep_path:
                    tasks.append((geom_path, castep_path))
                else:
                    print(f"警告: 找到 .geom 但没找到对应的 .castep: {f}")

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
        try:
            # 1. 用 ASE 读取结构 (从 .geom) - 这个很稳
            # 读取最后一步
            traj = read(geom_path, index=':', format='castep-geom')
            if isinstance(traj, list):
                atoms = traj[-1]
            else:
                atoms = traj

            # 2. 用 Python 正则读取标签 (从 .castep) - 绕过 Bug
            energy, forces = parse_castep_text(castep_path)

            if energy is None or forces is None:
                # print(f"跳过 {os.path.basename(castep_path)}: 解析能量或力失败")
                continue

            # 3. 校验原子数是否匹配
            if len(forces) != len(atoms):
                print(f"跳过 {os.path.basename(castep_path)}: 原子数不匹配 (结构:{len(atoms)}, 力:{len(forces)})")
                continue

            # 4. 构建 PyG Data
            atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
            cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).view(1, 3, 3)
            natoms = len(atoms)

            data = Data(
                atomic_numbers=atomic_numbers,
                pos=pos,
                cell=cell,
                y=torch.tensor([energy], dtype=torch.float),
                force=torch.tensor(forces, dtype=torch.float),  # 使用解析出的力
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
            print(f"处理失败: {e}")
            pass

    txn.commit()
    env.close()
    print(f"\n✅ 转换完成！")
    print(f"成功写入: {success_count} / {len(tasks)}")
    if success_count > 0:
        print(f"文件已保存至: {output_path}")
    else:
        print("仍未写入数据，请检查 .castep 文件中是否包含 'Final Enthalpy' 和 'Forces' 表格。")


if __name__ == "__main__":
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    output_folder = os.path.dirname(full_output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    castep_to_lmdb(INPUT_DIR, full_output_path)