import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.models import RegressionModel, RegressionModel2
import numpy as np
import pandas as pd
import os, yaml, pickle, glob
from transformers import RobertaTokenizerFast
import tqdm
import argparse
# --- [修复] 添加 datetime 导入 ---
from datetime import datetime


# -------------------------------


class InferenceScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def inverse_transform(self, normalized_preds):
        if self.mean is None: return normalized_preds
        return normalized_preds * self.std + self.mean


def predict_fn(data_loader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="   Inferring", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            if outputs.shape[-1] == 1: outputs = outputs.squeeze(-1)
            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions)


def run_ensemble_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug=False):
    print("=============================================================")
    print(f"Ensemble Prediction scanning: {pt_ckpt_dir_path}")
    print("=============================================================")

    # 1. 扫描所有模型权重
    seed_ckpts = glob.glob(os.path.join(pt_ckpt_dir_path, "best_model_seed_*.pt"))
    if len(seed_ckpts) == 0:
        # 回退：如果不是集成训练的，找普通权重
        fallback = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")
        # 再次回退：找默认名 checkpoint.pt
        fallback_default = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")

        if os.path.exists(fallback):
            seed_ckpts = [fallback]
        elif os.path.exists(fallback_default):
            seed_ckpts = [fallback_default]
        else:
            raise FileNotFoundError("No 'best_model_seed_*.pt', 'checkpoint.pt', or 'checkpoint.pt' found!")

    print(f"🔎 Found {len(seed_ckpts)} models: {[os.path.basename(x) for x in seed_ckpts]}")

    # 2. 准备数据
    df_test = pd.read_pickle(data_path)
    if debug: df_test = df_test.sample(10)

    device = "cuda" if torch.cuda.is_available() and not debug else "cpu"
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # 检查 target 列，如果没有则创建占位符
    if "target" in df_test.columns:
        targets_placeholder = df_test["target"].values
        # 简单的维度检查，确保是 numpy array
        if isinstance(targets_placeholder[0], (list, np.ndarray)):
            targets_placeholder = np.stack(targets_placeholder)
    else:
        targets_placeholder = np.zeros((len(df_test), 2))

    test_ds = RegressionDataset(texts=df_test["text"].values, targets=targets_placeholder,
                                tokenizer=tokenizer, seq_len=tokenizer.model_max_length)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # 3. 加载 Config
    model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml")
    if not os.path.exists(model_config_path):
        # 尝试在上级目录寻找 (兼容不同的目录结构)
        model_config_path = os.path.join(os.path.dirname(pt_ckpt_dir_path), "clip.yml")
        if not os.path.exists(model_config_path):
            model_config_path = "model/clip.yml"  # 最后尝试默认路径

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    # 4. 循环预测并累加
    accumulated_preds = None
    scaler = None  # 只需读取一次 scaler (所有种子 scaler 是一样的)

    for i, ckpt_path in enumerate(seed_ckpts):
        print(f"🤖 Model {i + 1}/{len(seed_ckpts)}: {os.path.basename(ckpt_path)}")

        # 加载权重
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)

        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        # 初始化 Scaler (第一次时，且checkpoint里有统计量)
        if scaler is None:
            if 'stats' in checkpoint:  # 兼容新版 regress_run_optimized.py
                scaler = InferenceScaler(mean=checkpoint['stats']['mean'], std=checkpoint['stats']['std'])
                print("   Loaded scaler stats from checkpoint['stats']")
            elif 'scaler_state_dict' in checkpoint:  # 兼容旧版命名
                scaler = InferenceScaler()
                scaler.mean = checkpoint['scaler_state_dict']['mean']
                scaler.std = checkpoint['scaler_state_dict']['std']
                print("   Loaded scaler stats from checkpoint['scaler_state_dict']")
            # 尝试从目录下的 pkl 文件加载
            elif os.path.exists(os.path.join(pt_ckpt_dir_path, 'scaler_stats.pkl')):
                with open(os.path.join(pt_ckpt_dir_path, 'scaler_stats.pkl'), 'rb') as f:
                    stats = pickle.load(f)
                scaler = InferenceScaler(mean=stats['mean'], std=stats['std'])
                print("   Loaded scaler stats from scaler_stats.pkl file")

        # 初始化模型
        if any("regresshead" in k for k in state_dict.keys()):
            model = RegressionModel2(model_config).to(device)
        else:
            model = RegressionModel(model_config).to(device)

        model.load_state_dict(state_dict, strict=True)

        # 预测
        preds = predict_fn(test_loader, model, device)

        if accumulated_preds is None:
            accumulated_preds = preds
        else:
            accumulated_preds += preds

    # 5. 取平均
    avg_preds = accumulated_preds / len(seed_ckpts)

    # 6. 反标准化
    if scaler:
        print("✅ Applying inverse transform (Denormalization)")
        final_preds = scaler.inverse_transform(avg_preds)
    else:
        print("⚠️ No scaler found, using raw outputs (Assuming model output is already in eV)")
        final_preds = avg_preds

    # 7. 保存与评估
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_file = os.path.join(save_path, f"ENSEMBLE-{tag}.pkl")

    # 保存字典
    with open(save_file, "wb") as f:
        pickle.dump(dict(zip(df_test["id"].values, final_preds)), f)
    print(f"💾 Predictions saved to: {save_file}")

    if "target" in df_test.columns:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        try:
            # 确保 targets 是 numpy 数组且形状匹配
            targets_val = df_test["target"].values
            if isinstance(targets_val[0], (list, np.ndarray)):
                targets = np.stack(targets_val)
            else:
                targets = targets_val.reshape(-1, 1)

            if final_preds.shape == targets.shape:
                print("\n📊 Ensemble Evaluation:")
                tasks = ["Adsorption Energy", "d-band Center"]
                for i in range(targets.shape[1]):
                    if i < len(tasks):
                        task_name = tasks[i]
                    else:
                        task_name = f"Task {i + 1}"

                    r2 = r2_score(targets[:, i], final_preds[:, i])
                    mae = mean_absolute_error(targets[:, i], final_preds[:, i])
                    rmse = np.sqrt(mean_squared_error(targets[:, i], final_preds[:, i]))
                    print(f"   {task_name}: R2 = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")
            else:
                print(f"\n⚠️ Shape mismatch for evaluation: Preds {final_preds.shape} vs Targets {targets.shape}")
        except Exception as e:
            print(f"\n⚠️ Evaluation skipped due to error: {e}")
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pt_ckpt_dir_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.tag is None: args.tag = datetime.now().strftime("%m%d_%H%M")
    run_ensemble_prediction(args.data_path, args.pt_ckpt_dir_path, args.save_path, args.tag, args.debug)