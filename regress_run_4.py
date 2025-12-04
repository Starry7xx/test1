import numpy as np
import pandas as pd
import torch, transformers, os
from torch.utils.data import DataLoader
from dataset import RegressionDataset
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml, os, shutil
from model.models import RegressionModel, RegressionModel2
from transformers import RobertaTokenizerFast
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import random


# ==========================================
# 0. 固定随机种子
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 1. 辅助类：标准化工具
# ==========================================
class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, targets):
        self.mean = np.mean(targets, axis=0)
        self.std = np.std(targets, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, targets):
        if self.mean is None: raise ValueError("Scaler not fitted")
        return (targets - self.mean) / self.std

    def inverse_transform(self, normalized_targets):
        if self.mean is None: return normalized_targets
        return normalized_targets * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# ==========================================
# 2. 训练与验证函数
# ==========================================
def run_epoch(data_loader, model, optimizer, device, loss_fn, scaler, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = []
    all_preds = []
    all_targets = []

    pbar = tqdm(data_loader, desc="Train" if is_train else "Val", leave=False, disable=not is_train)

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch["target"]

            outputs = model(batch)

            # --- 加权 Loss (保持 4:1) ---
            if outputs.dim() > 1 and outputs.shape[1] == 2:
                loss = 4.0 * loss_fn(outputs[:, 0], targets[:, 0]) + 1.0 * loss_fn(outputs[:, 1], targets[:, 1])
            else:
                loss = loss_fn(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss.append(loss.item())
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    real_preds = scaler.inverse_transform(all_preds)
    real_targets = scaler.inverse_transform(all_targets)

    metrics = {'loss': np.mean(epoch_loss)}
    if real_targets.shape[1] == 2:
        metrics['mae_en'] = mean_absolute_error(real_targets[:, 0], real_preds[:, 0])
        metrics['r2_en'] = r2_score(real_targets[:, 0], real_preds[:, 0])

    return metrics, real_targets, real_preds


# ==========================================
# 3. 单次训练流程 (封装)
# ==========================================
def train_one_seed(seed, config, df_train, df_val, scaler, run_dir):
    print(f"\n🌱 Starting Seed {seed}...")
    seed_everything(seed)

    DEVICE = config["device"]
    if config.get("debug", False): DEVICE = "cpu"

    # Dataset
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    train_ds = RegressionDataset(texts=df_train["text"].values,
                                 targets=np.stack(df_train["target_norm"].values),
                                 tokenizer=tokenizer, seq_len=tokenizer.model_max_length)
    val_ds = RegressionDataset(texts=df_val["text"].values,
                               targets=np.stack(df_val["target_norm"].values),
                               tokenizer=tokenizer, seq_len=tokenizer.model_max_length)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    # Model
    with open(config["model_config"], "r") as f:
        m_config = yaml.safe_load(f)
    if config.get("head") == "pooler":
        model = RegressionModel2(m_config).to(DEVICE)
    else:
        model = RegressionModel(m_config).to(DEVICE)

    # Load Pretrained
    if config.get("pt_ckpt_path"):
        try:
            ckpt = torch.load(config['pt_ckpt_path'], map_location=DEVICE, weights_only=False)
        except TypeError:
            ckpt = torch.load(config['pt_ckpt_path'], map_location=DEVICE)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)

    # 解冻策略
    for param in model.parameters(): param.requires_grad = False
    for name, param in model.named_parameters():
        if "regresshead" in name or "regressor" in name or "projection" in name or "dense" in name:
            param.requires_grad = True
    for name, param in model.text_encoder.named_parameters():
        if "layer.11" in name or "layer.10" in name or "pooler" in name:
            param.requires_grad = True

    # 差分 LR
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "text_encoder" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "text_encoder" not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    loss_fn = torch.nn.MSELoss()

    # Loop
    best_loss = 999
    early_stop = 0

    for epoch in range(1, config['num_epochs'] + 1):
        tr_m, _, _ = run_epoch(train_loader, model, optimizer, DEVICE, loss_fn, scaler, True)
        val_m, _, _ = run_epoch(val_loader, model, optimizer, DEVICE, loss_fn, scaler, False)

        scheduler.step(val_m['loss'])

        if val_m['loss'] < best_loss:
            best_loss = val_m['loss']
            early_stop = 0
            # 保存当前种子下的最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss
            }, os.path.join(run_dir, f'best_model_seed_{seed}.pt'))
        else:
            early_stop += 1

        if early_stop >= config['early_stop_threshold']:
            print(f"   Seed {seed} stopped at epoch {epoch}")
            break

    print(f"   Seed {seed} finished. Best Loss: {best_loss:.4f}")


# ==========================================
# 4. 主入口
# ==========================================
def run_ensemble(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # 强制参数
    config['lr'] = 1e-3
    config['num_epochs'] = 120
    config['batch_size'] = 16
    config['early_stop_threshold'] = 20

    RUN_NAME = config["run_name"] + "_ENSEMBLE_" + datetime.now().strftime("%m%d_%H%M")
    CKPT_SAVE_DIR = os.path.join(config["ckpt_save_path"], RUN_NAME)
    if not os.path.exists(CKPT_SAVE_DIR): os.makedirs(CKPT_SAVE_DIR)

    # 保存配置
    shutil.copyfile(config_file, os.path.join(CKPT_SAVE_DIR, os.path.basename(config_file)))
    shutil.copyfile(config["model_config"], os.path.join(CKPT_SAVE_DIR, os.path.basename(config["model_config"])))

    if not config.get("debug", False): wandb.init(project="clip-regress", name=RUN_NAME)

    # 准备数据 (所有种子共享同一套数据和Scaler)
    df_train = pd.read_pickle(config["train_path"])
    df_val = pd.read_pickle(config["val_path"])

    scaler = TargetScaler()
    train_targets = np.vstack(df_train["target"].values)
    val_targets = np.vstack(df_val["target"].values)
    scaler.fit(train_targets)

    df_train["target_norm"] = list(scaler.transform(train_targets))
    df_val["target_norm"] = list(scaler.transform(val_targets))

    # === 训练 5 个种子 ===
    SEEDS = [42, 43, 44, 45, 46]
    print(f"🚀 Launching Ensemble Training (Seeds: {SEEDS})")

    for seed in SEEDS:
        train_one_seed(seed, config, df_train, df_val, scaler, CKPT_SAVE_DIR)

    print("\n✅ All seeds finished.")
    if not config.get("debug", False): wandb.finish()


if __name__ == "__main__":
    run_ensemble("regress_train.yml")