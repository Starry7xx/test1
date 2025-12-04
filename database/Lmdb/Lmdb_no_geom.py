import os
import lmdb
import pickle
import torch
import numpy as np
import re
from ase import Atoms
from ase.data import atomic_numbers
from torch_geometric.data import Data
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "/multiview_data/pred_data/Ours/database/data/Rh/Rh"
OUTPUT_DIR = "/multiview_data/pred_data/Ours/database/Lmdb"
OUTPUT_FILENAME = "Rh/Rh.lmdb"


# ===========================================

def manual_parse_structure(castep_path):
    """
    [核心功能] 手动解析 .castep 文件中的晶胞和原子坐标
    不依赖 ASE，专治各种读取疑难杂症
    """
    try:
        with open(castep_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 1. 寻找最后的晶胞信息 (Unit Cell)
        cell = None
        # 倒序寻找 "Real Lattice(A)"
        for i in range(len(lines) - 1, -1, -1):
            if "Real Lattice(A)" in lines[i]:
                # CASTEP 格式通常是:
                # Real Lattice(A)
                #   ax ay az
                #   bx by bz
                #   cx cy cz
                try:
                    row1 = lines[i + 1].strip().split()[:3]
                    row2 = lines[i + 2].strip().split()[:3]
                    row3 = lines[i + 3].strip().split()[:3]
                    cell = np.array([
                        [float(row1[0]), float(row1[1]), float(row1[2])],
                        [float(row2[0]), float(row2[1]), float(row2[2])],
                        [float(row3[0]), float(row3[1]), float(row3[2])]
                    ])
                    break
                except:
                    continue

        if cell is None:
            # print(f"  [调试] 未找到晶胞信息 (Real Lattice)")
            return None

        # 2. 寻找最后的原子坐标 (Fractional coordinates)
        symbols = []
        positions_frac = []

        start_idx = -1
        # 倒序找 "Fractional coordinates of atoms"
        for i in range(len(lines) - 1, -1, -1):
            if "Fractional coordinates of atoms" in lines[i]:
                start_idx = i
                break

        if start_idx != -1:
            # 向下解析表格
            # 格式示例:
            # x  O             1         0.50000    0.50000    0.00000         x
            for i in range(start_idx + 3, len(lines)):
                line = lines[i]
                # 遇到分隔线停止
                if "xxxxxxx" in line or "-------" in line:
                    if len(symbols) > 0:
                        break  # 如果已经读到了数据，就结束
                    else:
                        continue

                # 简单清洗: 去掉 x 和 *
                clean = line.replace('x', ' ').replace('*', ' ').strip()
                parts = clean.split()

                # 至少要有: Element, AtomNo, u, v, w (5列)
                if len(parts) >= 5:
                    try:
                        elem = parts[0]
                        # 简单的元素符号检查 (排除表头)
                        if elem in atomic_numbers:
                            u = float(parts[2])
                            v = float(parts[3])
                            w = float(parts[4])
                            symbols.append(elem)
                            positions_frac.append([u, v, w])
                    except:
                        continue

        if not symbols:
            # print(f"  [调试] 未找到原子坐标")
            return None

        # 3. 转换为 ASE Atoms 对象 (方便后续处理)
        # 坐标转换: Cartesian = Fractional * Cell
        positions_frac = np.array(positions_frac)
        # 注意: numpy dot 矩阵乘法
        positions_cart = np.dot(positions_frac, cell)

        atoms = Atoms(symbols=symbols, positions=positions_cart, cell=cell, pbc=True)
        return atoms

    except Exception as e:
        print(f"  [结构解析错误] {e}")
        return None


def manual_parse_labels(castep_path):
    """
    手动提取能量和力 (复用之前的救援逻辑)
    """
    energy = None
    forces = []

    try:
        with open(castep_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 1. 提取能量
        for line in reversed(lines):
            if any(k in line for k in ["Final Enthalpy", "Final Energy", "Total energy"]):
                if "=" in line:
                    try:
                        val = line.split('=')[-1].strip().split()[0]  # 取 = 号后第一个词
                        # 处理 E+004
                        if 'E' in val and len(val.split('E')[0]) == 0: continue
                        energy = float(val)
                        break
                    except:
                        continue

        # 2. 提取力
        force_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if "Forces" in lines[i] and "*****" in lines[i]:
                force_idx = i
                break

        if force_idx != -1:
            for i in range(force_idx + 6, len(lines)):
                line = lines[i]
                if "*****" in line: break
                parts = line.replace('*', ' ').strip().split()
                if len(parts) >= 5:
                    try:
                        # CASTEP Force table usually: El, No, Fx, Fy, Fz
                        forces.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    except:
                        continue

    except:
        pass

    return energy, np.array(forces) if forces else None


def castep_to_lmdb(root_dir, output_path):
    print(f"正在扫描输入文件夹: {root_dir} ...")

    castep_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.castep') and not any(x in f for x in ['BandStr', 'DOS']):
                castep_files.append(os.path.join(dirpath, f))

    if not castep_files:
        print("❌ 未找到 .castep 文件")
        return

    print(f"发现 {len(castep_files)} 个任务文件，准备开始硬核转换...")

    env = lmdb.open(output_path, map_size=10 * 1024 ** 3, subdir=False, readonly=False, meminit=False, map_async=True)
    txn = env.begin(write=True)
    idx = 0
    success = 0

    for castep_path in tqdm(castep_files):
        base_name = os.path.basename(castep_path)

        try:
            # --- 第一步：手动搞定结构 ---
            atoms = manual_parse_structure(castep_path)
            if atoms is None:
                print(f"\n  ❌ 跳过 {base_name}: 结构解析失败 (文本中未找到晶胞或坐标)。")
                continue

            # --- 第二步：手动搞定标签 ---
            energy, forces = manual_parse_labels(castep_path)

            if energy is None or forces is None:
                print(f"\n  ❌ 跳过 {base_name}: 标签缺失 (能量或力未找到)。")
                continue

            if len(forces) != len(atoms):
                print(f"\n  ❌ 跳过 {base_name}: 原子数不匹配 (结构:{len(atoms)} vs 力:{len(forces)})")
                continue

            # --- 第三步：打包写入 ---
            data = Data(
                atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
                pos=torch.tensor(atoms.get_positions(), dtype=torch.float),
                cell=torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).view(1, 3, 3),
                y=torch.tensor([energy], dtype=torch.float),
                force=torch.tensor(forces, dtype=torch.float),
                natoms=len(atoms),
                tags=torch.zeros(len(atoms), dtype=torch.long),
                fixed=torch.zeros(len(atoms), dtype=torch.float),
                sid=torch.tensor([idx], dtype=torch.long)
            )

            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            idx += 1
            success += 1

            if idx % 500 == 0:
                txn.commit()
                txn = env.begin(write=True)

        except Exception as e:
            print(f"\n  [异常] {base_name}: {e}")

    txn.commit()
    env.close()
    print(f"\n✅ 转换完成！成功写入: {success} / {len(castep_files)}")
    if success > 0:
        print(f"文件保存至: {output_path}")


if __name__ == "__main__":
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    output_folder = os.path.dirname(full_output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    castep_to_lmdb(INPUT_DIR, full_output_path)