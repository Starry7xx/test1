import pickle
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 1. 设置预测结果 .pkl 文件的路径 (确保这是双任务预测生成的 pkl)
PRED_PKL_PATH = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-two_predict/result/1/predict/1_regress_train_1201_2013-251201_201856-strc.pkl"

# 2. (可选) 设置原始测试数据 .pkl 文件的路径
DATA_PKL_PATH = "/media/mhx/f230303b-a03a-48de-baa3-a423f80e2a89/wwx/ML/multi-view-main-two_predict/multiview_data/Ours/DATABASE_all_val.pkl"

# 3. 想要查看前多少个最好的候选者
TOP_K = 10

# 4. 吸附能 (Adsorption Energy) 筛选标准 (单位: eV)
TARGET_ENERGY = -0.67  # 理想目标值
RANGE_MIN_E = -0.80  # 能量范围下限
RANGE_MAX_E = -0.50  # 能量范围上限

# 5. d带中心 (d-band Center) 筛选标准 (单位: eV)
# 要求: -1.5 到 -2.5 之间 (注意负数大小: -2.5 是下限, -1.5 是上限)
RANGE_MIN_D = -2.50
RANGE_MAX_D = -1.50


# ===========================================

def find_best_catalysts():
    print(f"正在加载预测文件: {PRED_PKL_PATH} ...")
    try:
        with open(PRED_PKL_PATH, "rb") as f:
            predictions = pickle.load(f)
    except Exception as e:
        print(f"错误: 无法加载预测文件。{e}")
        return

    # --- 修改的核心逻辑开始 ---
    filtered_candidates = []

    # 遍历所有预测结果
    for cid, val in predictions.items():
        # 1. 提取数值 (兼容标量和向量)
        if isinstance(val, (list, np.ndarray)) and len(val) >= 2:
            # 双任务模式
            energy = float(val[0])
            dband = float(val[1])
        else:
            # 单任务兼容模式 (如果没有d带数据，设为 None)
            energy = float(val) if np.isscalar(val) else float(val[0])
            dband = None

        # 2. 筛选逻辑
        # 条件A: 吸附能在范围内
        is_energy_ok = RANGE_MIN_E <= energy <= RANGE_MAX_E

        # 条件B: d带中心在范围内 (如果只有吸附能数据，则跳过此判断或视为不满足，这里设定为必须有d带数据)
        is_dband_ok = False
        if dband is not None:
            is_dband_ok = RANGE_MIN_D <= dband <= RANGE_MAX_D

        # 同时满足两个条件
        if is_energy_ok and is_dband_ok:
            filtered_candidates.append((cid, energy, dband))

    print(f"原始数据共 {len(predictions)} 条。")
    print(f"筛选标准: Energy [{RANGE_MIN_E}, {RANGE_MAX_E}], d-band [{RANGE_MIN_D}, {RANGE_MAX_D}]")
    print(f"筛选后剩余 {len(filtered_candidates)} 条符合条件的候选者。")

    if not filtered_candidates:
        print("未找到同时符合这两个条件的催化剂。")
        # 可选：如果没找到，可以尝试放宽 d-band 条件，或者只按 Energy 筛选打印提示
        return

    # 3. 排序：仍然按照与 TARGET_ENERGY 的“距离”排序 (主要目标是吸附能最理想)
    # 如果你也想考虑 d-band 的排序，可以修改 key
    sorted_preds = sorted(filtered_candidates, key=lambda item: abs(item[1] - TARGET_ENERGY))
    # --- 修改的核心逻辑结束 ---

    # 取前 K 个
    top_candidates = sorted_preds[:TOP_K]

    # 尝试加载原始数据以获取详细信息
    df_data = None
    if DATA_PKL_PATH and os.path.exists(DATA_PKL_PATH):
        try:
            print(f"正在加载原始数据文件: {DATA_PKL_PATH} ...")
            df_data = pd.read_pickle(DATA_PKL_PATH)
            if "id" in df_data.columns:
                df_data = df_data.set_index("id")
        except Exception as e:
            print(f"提示: 无法加载原始数据文件 ({e})，将仅显示预测值。")

    # 准备输出内容
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"筛选 Top {TOP_K} 候选催化剂 (双指标筛选)")
    output_lines.append(f"Energy 目标: {TARGET_ENERGY} eV (范围 {RANGE_MIN_E} ~ {RANGE_MAX_E})")
    output_lines.append(f"d-band 范围: {RANGE_MIN_D} ~ {RANGE_MAX_D} eV")
    output_lines.append("=" * 80)

    for rank, (cid, energy, dband) in enumerate(top_candidates, 1):
        diff = energy - TARGET_ENERGY

        output_lines.append(f"排名 {rank}:")
        output_lines.append(f" ID: {cid}")
        output_lines.append(f" 预测吸附能 (Energy): {energy:.5f} eV (距目标: {diff:+.5f})")
        output_lines.append(f" 预测d带中心 (d-band): {dband:.5f} eV")

        if df_data is not None and cid in df_data.index:
            # 显示文本描述
            if "text" in df_data.columns:
                output_lines.append(f" 描述: {df_data.loc[cid, 'text']}")

            # 如果有真实值，显示对比
            if "target" in df_data.columns:
                true_val = df_data.loc[cid, 'target']
                # 判断真实值是标量还是向量
                if isinstance(true_val, (list, np.ndarray)) and len(true_val) >= 2:
                    t_en, t_db = true_val[0], true_val[1]
                    output_lines.append(f" 真实值 (DFT): Energy={t_en:.4f}, d-band={t_db:.4f}")
                else:
                    output_lines.append(f" 真实值 (DFT): {true_val}")

        output_lines.append("-" * 80)

    # 打印到控制台
    print("\n" + "\n".join(output_lines))

    # 保存到txt文件
    predict_dir = os.path.dirname(PRED_PKL_PATH)
    base_name = os.path.basename(PRED_PKL_PATH).replace('.pkl', '')
    txt_path = os.path.join(predict_dir, f"{base_name}_screened_dual_target.txt")

    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(output_lines) + "\n")
    print(f"\n结果已保存到: {txt_path}")


if __name__ == "__main__":
    find_best_catalysts()