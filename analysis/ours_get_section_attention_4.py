import sys
import os
import glob  # 用于查找文件

# =========================================================
# 核心修复 1: 将项目根目录添加到 Python 搜索路径
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # analysis/
project_root = os.path.dirname(current_dir)  # multi-view-main/
if project_root not in sys.path:
    sys.path.append(project_root)
# =========================================================

import torch
from torch.utils.data import DataLoader
from dataset import RegressionDataset
from model.modules import TextEncoder_attn
import numpy as np
import pandas as pd
import yaml, pickle
from transformers import RobertaTokenizerFast
import tqdm


def attention_per_part(data_loader, model, tokenizer, device):
    """
    计算并聚合模型不同注意力头的注意力权重。
    """
    model.eval()
    num_heads = 12
    weights = torch.zeros((4, num_heads))

    num_fails = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            attentions = model(batch).squeeze(-1)

            for i in range(attentions.size(0)):
                ids = batch['input_ids'][i]
                tokens = tokenizer.convert_ids_to_tokens(ids)

                for head in range(num_heads):
                    head_attn = attentions[i, head, 0, :]

                    partsAtt = []
                    tokens_copy = tokens[1:]
                    num_end_tokens = tokens_copy.count('</s>')

                    # 核心检查点：期望找到 3 个 </s>
                    num_of_sections = 3
                    if num_end_tokens != num_of_sections:
                        num_fails += 1
                        continue

                    partsAtt.append(head_attn[0])  # <s>

                    for _ in range(num_of_sections):
                        if '</s>' in tokens_copy:
                            index = tokens_copy.index('</s>') + 1
                            partsAtt.append(head_attn[:index])
                            tokens_copy = tokens_copy[index:]
                            head_attn = head_attn[index:]
                        else:
                            break

                    for part in range(len(partsAtt)):
                        if part < 4:
                            weights[part, head] += partsAtt[part].sum().item()

        n = len(data_loader.dataset)
        # 只有在有成功样本的情况下才计算平均值
        valid_samples = (n * num_heads - num_fails) / num_heads
        if valid_samples > 0:
            weights = weights / valid_samples
        else:
            print("Warning: No valid samples found!")

        print("Weights Matrix:")
        print(weights)
        print('Number of failures (structure mismatch): ', num_fails)
        results = {'weights': weights, 'num_fails': num_fails}

    return results


def run_prediction(data_path, pt_ckpt_dir_path, save_path, tag, debug=False, specific_model_name=None):
    """
    修改说明：增加了 specific_model_name 参数，用于指定具体的 .pt 文件名
    """

    # 默认配置逻辑
    if 'roberta-base' not in pt_ckpt_dir_path:
        # 如果指定了具体文件名，则使用具体文件名，否则默认为 checkpoint.pt
        if specific_model_name:
            ckpt_name = specific_model_name.replace('.pt', '')  # 用于日志打印
            pt_ckpt_path = os.path.join(pt_ckpt_dir_path, specific_model_name)
        else:
            ckpt_name = pt_ckpt_dir_path.rstrip('/').split('/')[-1]
            pt_ckpt_path = os.path.join(pt_ckpt_dir_path, "checkpoint.pt")

        model_config_path = os.path.join(pt_ckpt_dir_path, "clip.yml")

        print(f"Loading config from: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

    elif 'roberta-base' in pt_ckpt_dir_path:
        ckpt_name = 'roberta-base'
        config_path = os.path.join(project_root, "model/clip.yml")
        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f)
        model_config['Path']['pretrain_ckpt'] = "roberta-base"
        print('loading encoder from roberta-base')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        device = "cpu"

    print("=============================================================")
    print(f"Attention from {ckpt_name} (File: {specific_model_name if specific_model_name else 'checkpoint.pt'})")
    print("=============================================================")

    # ========================= 数据加载与修复 =========================
    print(f"Loading data from: {data_path}")
    df_test = pd.read_pickle(data_path)

    # 【修复】：将 <s> 替换为 </s> 以匹配代码逻辑
    if "text" in df_test.columns:
        # print("Fixing separators: replacing <s> with </s>...")
        df_test["text"] = df_test["text"].astype(str).str.replace("<s>", "</s>")
    # ================================================================

    if debug:
        df_test = df_test.sample(10)

    local_roberta_path = os.path.join(project_root, 'local_roberta')
    if os.path.exists(local_roberta_path):
        print(f"Loading tokenizer from local path: {local_roberta_path}")
        tokenizer = RobertaTokenizerFast.from_pretrained(local_roberta_path)
    else:
        print("Loading tokenizer from HuggingFace (roberta-base)...")
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    test_dataset = RegressionDataset(texts=df_test["text"].values,
                                     targets=df_test["target"].values,
                                     tokenizer=tokenizer,
                                     seq_len=tokenizer.model_max_length)
    test_data_loader = DataLoader(test_dataset, batch_size=1,
                                  shuffle=False, num_workers=1)

    model = TextEncoder_attn(model_config).to(device)

    print(f'Loading pretrained checkpoint from: {pt_ckpt_path}')
    if ckpt_name != 'roberta-base':
        prefix = 'text_encoder.'

        if 'ssl' not in ckpt_name:
            # =========================================================
            # 【修复点】：显式设置 weights_only=False 以兼容旧版/Numpy对象
            # =========================================================
            checkpoint = torch.load(pt_ckpt_path, map_location=device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        elif 'ssl' in ckpt_name:
            # 这里也加上 weights_only=False 以防万一
            state_dict = torch.load(pt_ckpt_path, map_location=device, weights_only=False)

        # 权重键值过滤与匹配
        new_state_dict = {key[len(prefix):]: value for key, value in state_dict.items() \
                          if key.startswith(prefix) and not key.startswith(prefix + 'chg_embedding')}

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load status: {msg}")

    print("Calculating attention weights...")
    predictions = attention_per_part(test_data_loader, model, tokenizer, device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存文件名增加具体模型名称，防止覆盖
    save_name_suffix = specific_model_name.replace('.pt', '') if specific_model_name else ckpt_name
    final_save_name = f"attn-{save_name_suffix}-{tag}-strc.pkl"
    final_save_path = os.path.join(save_path, final_save_name)

    print(f"Saving results to: {final_save_path}")
    with open(final_save_path, "wb") as f:
        pickle.dump(predictions, f)

    print("-" * 60)


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Script to run predictions.")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--pt_ckpt_dir_path", type=str, required=True,
                        help="Path to the pretrained checkpoint directory.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the predictions.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--tag", type=str, default=datetime.now().strftime("%y%m%d_%H%M%S"), help="Tag for the run.")

    args = parser.parse_args()

    # =========================================================
    # 自动扫描目录下的所有 best_model_seed_*.pt 文件
    # =========================================================

    # 获取目录下所有文件
    if os.path.exists(args.pt_ckpt_dir_path) and 'roberta-base' not in args.pt_ckpt_dir_path:
        all_files = os.listdir(args.pt_ckpt_dir_path)
        # 筛选出符合 best_model_seed_*.pt 模式的文件
        model_files = [f for f in all_files if f.startswith("best_model_seed") and f.endswith(".pt")]

        # 按文件名排序，确保顺序执行（比如 seed_42, 43, 44...）
        model_files.sort()

        if len(model_files) > 0:
            print(f"Detected {len(model_files)} ensemble models. Starting batch processing...")
            for model_file in model_files:
                run_prediction(args.data_path, args.pt_ckpt_dir_path, args.save_path, args.tag, args.debug,
                               specific_model_name=model_file)
        else:
            # 如果没找到特定 seed 文件，回退到默认行为（找 checkpoint.pt）
            print("No 'best_model_seed_*.pt' files found. Looking for default checkpoint.pt...")
            run_prediction(args.data_path, args.pt_ckpt_dir_path, args.save_path, args.tag, args.debug)

    else:
        # 如果是 roberta-base 或者是无效路径，直接运行一次
        run_prediction(args.data_path, args.pt_ckpt_dir_path, args.save_path, args.tag, args.debug)