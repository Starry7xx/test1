import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def analyze_results(pred_path, true_data_path, save_dir):
    """
    分析预测结果，自动适应单任务(Energy)或多任务(Energy + d-band)。
    如果维度不匹配，尝试降级处理。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 1. 读取数据
    print(f"Loading predictions from: {pred_path}")
    pred_dict = pd.read_pickle(pred_path)

    print(f"Loading ground truth from: {true_data_path}")
    df_true = pd.read_pickle(true_data_path)

    if 'id' not in df_true.columns or 'target' not in df_true.columns:
        raise ValueError(f"Ground truth file must contain 'id' and 'target' columns. Found: {df_true.columns}")

    true_dict = dict(zip(df_true['id'], df_true['target']))

    # 2. 对齐数据
    matched_ids = []
    y_true_raw = []
    y_pred_raw = []

    print("Aligning data...")
    for sid, pred_val in pred_dict.items():
        if sid in true_dict:
            y_true_raw.append(true_dict[sid])
            y_pred_raw.append(pred_val)
            matched_ids.append(sid)

    if len(y_true_raw) == 0:
        print("Error: No matching IDs found.")
        return

    print(f"Matched {len(y_true_raw)} samples.")

    # 3. 维度检查与预处理
    sample_true = y_true_raw[0]
    sample_pred = y_pred_raw[0]

    # 判断是否为标量或长度为1的数组
    is_true_scalar = np.isscalar(sample_true) or (isinstance(sample_true, (list, np.ndarray)) and len(sample_true) == 1)
    is_pred_scalar = np.isscalar(sample_pred) or (isinstance(sample_pred, (list, np.ndarray)) and len(sample_pred) == 1)

    y_true_final = []
    y_pred_final = []
    task_names = []

    # --- 场景判定逻辑 ---

    # 场景 A: 真值是标量 (只有 Energy)
    if is_true_scalar:
        print("Detected Single-Task Ground Truth (Energy only).")
        y_true_final = np.array([float(x) if np.isscalar(x) else x[0] for x in y_true_raw]).reshape(-1, 1)

        if is_pred_scalar:
            # 预测也是标量 -> 完美匹配
            y_pred_final = np.array([float(x) if np.isscalar(x) else x[0] for x in y_pred_raw]).reshape(-1, 1)
        else:
            # 预测是向量 -> 取第0个
            print(
                "Warning: Prediction is Multi-task but Ground Truth is Single-task. Using 1st dimension of prediction.")
            y_pred_final = np.array([x[0] for x in y_pred_raw]).reshape(-1, 1)

        task_names = ["Adsorption Energy"]

    # 场景 B: 真值是向量 (Energy, d-band)
    else:
        print("Detected Multi-Task Ground Truth (Energy + d-band).")

        if is_pred_scalar:
            # === 这里的逻辑解决了你遇到的错误 ===
            print("Warning: Ground Truth is Multi-task (Energy, d-band) but Prediction is Scalar.")
            print("-> Assuming Prediction is Adsorption Energy only. Comparing with GT column 0.")
            print("-> d-band Center will be SKIPPED.")

            # 取真值的第0列 (Energy)
            y_true_final = np.array([x[0] for x in y_true_raw]).reshape(-1, 1)
            # 预测值保持原样 (并reshape)
            y_pred_final = np.array([float(x) if np.isscalar(x) else x[0] for x in y_pred_raw]).reshape(-1, 1)

            task_names = ["Adsorption Energy (Partial)"]

        else:
            # 都是多任务 -> 正常比较
            y_true_final = np.array(y_true_raw)  # (N, 2)
            y_pred_final = np.array(y_pred_raw)  # (N, 2)
            task_names = ["Adsorption Energy", "d-band Center"]

    num_tasks = y_true_final.shape[1]
    units = ["eV", "eV"]

    # 4. 绘图
    fig, axes = plt.subplots(1, num_tasks, figsize=(7 * num_tasks, 6))
    if num_tasks == 1:
        axes = [axes]  # 确保它是可迭代的列表

    metrics_log = []
    metrics_log.append(f"Sample Count: {len(y_true_final)}\n")

    for i in range(num_tasks):
        curr_true = y_true_final[:, i]
        curr_pred = y_pred_final[:, i]
        curr_name = task_names[i]
        curr_unit = units[i] if i < len(units) else "eV"
        ax = axes[i]

        mae = mean_absolute_error(curr_true, curr_pred)
        rmse = np.sqrt(mean_squared_error(curr_true, curr_pred))
        r2 = r2_score(curr_true, curr_pred)

        log_str = f"--- {curr_name} ---\nR2: {r2:.4f} | MAE: {mae:.4f} {curr_unit} | RMSE: {rmse:.4f} {curr_unit}"
        print(log_str)
        metrics_log.append(log_str + "\n")

        ax.scatter(curr_true, curr_pred, alpha=0.6, s=20, edgecolors='none', color='blue')

        min_val = min(curr_true.min(), curr_pred.min())
        max_val = max(curr_true.max(), curr_pred.max())
        # 防止 max == min 导致报错
        if max_val == min_val:
            buffer = 1.0
        else:
            buffer = (max_val - min_val) * 0.05

        lims = [min_val - buffer, max_val + buffer]

        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        ax.set_xlabel(f'True {curr_name} [{curr_unit}]', fontsize=12)
        ax.set_ylabel(f'Predicted {curr_name} [{curr_unit}]', fontsize=12)
        ax.set_title(f'{curr_name}\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}', fontsize=14)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'parity_plot.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.writelines(metrics_log)
    print(f"Metrics saved to: {metrics_path}")

    # 保存对齐数据
    save_dict = {'ids': matched_ids}
    for i in range(num_tasks):
        name_clean = task_names[i].replace(" ", "_").replace("(", "").replace(")", "").lower()
        save_dict[f'{name_clean}_true'] = y_true_final[:, i]
        save_dict[f'{name_clean}_pred'] = y_pred_final[:, i]

    npz_path = os.path.join(save_dir, 'pred_aligned.npz')
    np.savez(npz_path, **save_dict)
    print(f"Aligned data saved to: {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze regression results.")
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--true_data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()

    analyze_results(args.pred_path, args.true_data_path, args.save_dir)