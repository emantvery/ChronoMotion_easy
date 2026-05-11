#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量模型评估工具
"""
import os, csv, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import MultiModalVectorNet
from enhanced_train import TrajectoryDataset

OBS_LEN = 20
PRED_LEN = 30
FEATURE_DIM = 8
OUT_DIM = PRED_LEN * 2
K_MODES = 3
DATA_DIR = './mydata'


def load_model(model_path, device='cpu'):
    model = MultiModalVectorNet(in_channels=FEATURE_DIM, out_channels=OUT_DIM,
                                k_modes=K_MODES, hidden_dim=128, obs_len=OBS_LEN)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model


def compute_ade(pred, target):
    pred_coords = pred.reshape(-1, PRED_LEN, 2)
    target_coords = target.reshape(-1, PRED_LEN, 2)
    diff = pred_coords - target_coords
    dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
    return dist.mean(dim=1)


def compute_fde(pred, target):
    pred_final = pred[:, -2:].reshape(-1, 2)
    target_final = target[:, -2:].reshape(-1, 2)
    diff = pred_final - target_final
    dist = torch.sqrt(torch.sum(diff ** 2, dim=1))
    return dist


def read_csv_trajectory(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[2] == 'AGENT':
                data.append({'TIMESTAMP': float(row[0]), 'X': float(row[3]), 'Y': float(row[4])})
    data.sort(key=lambda x: x['TIMESTAMP'])
    return [[d['X'], d['Y']] for d in data]


def compute_features(obs_traj):
    features = []
    for i in range(len(obs_traj)):
        if i == 0:
            vel_x, vel_y = 0, 0
        else:
            vel_x = obs_traj[i, 0] - obs_traj[i-1, 0]
            vel_y = obs_traj[i, 1] - obs_traj[i-1, 1]
        features.append([obs_traj[i, 0], obs_traj[i, 1], vel_x, vel_y, i / OBS_LEN, 1, 0, 0])
    return np.array(features, dtype=np.float32)


def evaluate_model(model, data_dir, split='test', device='cpu'):
    dataset = TrajectoryDataset(data_dir, split, obs_len=OBS_LEN, pred_len=PRED_LEN, feature_dim=FEATURE_DIM)
    all_ade, all_fde, all_mse, all_best_mode, all_probs = [], [], [], [], []
    results = []
    with torch.no_grad():
        for i in range(len(dataset)):
            features, targets, filename = dataset[i]
            input_tensor = features.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            pred_trajs = outputs['trajectories'][0].cpu()
            mode_probs = outputs['mode_probs'][0].cpu().numpy()
            best_ade = float('inf')
            best_mode, best_pred = 0, None
            for k in range(K_MODES):
                ade = compute_ade(pred_trajs[k:k+1], targets.unsqueeze(0))
                if ade < best_ade:
                    best_ade = ade.item()
                    best_mode = k
                    best_pred = pred_trajs[k].numpy()
            fde = compute_fde(torch.from_numpy(best_pred).unsqueeze(0), targets.unsqueeze(0)).item()
            mse = torch.mean((torch.from_numpy(best_pred).unsqueeze(0) - targets.unsqueeze(0)) ** 2).item()
            all_ade.append(best_ade); all_fde.append(fde); all_mse.append(mse)
            all_best_mode.append(best_mode); all_probs.append(mode_probs[best_mode])
            results.append({'filename': filename, 'best_mode': best_mode + 1,
                            'mode_prob': float(mode_probs[best_mode]), 'ade': best_ade, 'fde': fde, 'mse': mse})
    return {'ade': all_ade, 'fde': all_fde, 'mse': all_mse, 'best_mode': all_best_mode,
            'probabilities': all_probs, 'details': results}


def generate_report(eval_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ade_mean = np.mean(eval_results['ade']); ade_std = np.std(eval_results['ade'])
    fde_mean = np.mean(eval_results['fde']); fde_std = np.std(eval_results['fde'])
    mse_mean = np.mean(eval_results['mse']); mse_std = np.std(eval_results['mse'])
    mode_counts = {}
    for mode in eval_results['best_mode']:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    report = f"""
{'='*60}
                轨迹预测模型评估报告
{'='*60}
【评估配置】 数据集: {DATA_DIR}  观测: {OBS_LEN}  预测: {PRED_LEN}  模态: {K_MODES}
【总体性能】  ADE: {ade_mean:.2f} +/- {ade_std:.2f} m  |  FDE: {fde_mean:.2f} +/- {fde_std:.2f} m  |  MSE: {mse_mean:.4f} +/- {mse_std:.4f}
【模式分布】  模式1: {mode_counts.get(0, 0)}  模式2: {mode_counts.get(1, 0)}  模式3: {mode_counts.get(2, 0)}
【概率】      平均: {np.mean(eval_results['probabilities']):.4f}  Std: {np.std(eval_results['probabilities']):.4f}
"""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f: f.write(report)
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': {'ade_mean': float(ade_mean), 'ade_std': float(ade_std), 'fde_mean': float(fde_mean),
                               'fde_std': float(fde_std), 'mse_mean': float(mse_mean), 'mse_std': float(mse_std),
                               'mode_distribution': mode_counts, 'avg_probability': float(np.mean(eval_results['probabilities']))},
                   'details': eval_results['details']}, f, indent=2, ensure_ascii=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(eval_results['ade'], bins=10, color='blue', alpha=0.7)
    axes[0, 0].axvline(ade_mean, color='red', linestyle='--', label=f'Mean: {ade_mean:.2f}')
    axes[0, 0].set_xlabel('ADE (m)'); axes[0, 0].set_ylabel('Count'); axes[0, 0].set_title('ADE Distribution'); axes[0, 0].legend()
    axes[0, 1].hist(eval_results['fde'], bins=10, color='green', alpha=0.7)
    axes[0, 1].axvline(fde_mean, color='red', linestyle='--', label=f'Mean: {fde_mean:.2f}')
    axes[0, 1].set_xlabel('FDE (m)'); axes[0, 1].set_ylabel('Count'); axes[0, 1].set_title('FDE Distribution'); axes[0, 1].legend()
    axes[1, 0].pie(mode_counts.values(), labels=[f'Mode {k+1}' for k in mode_counts.keys()],
                   autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4', '#9B59B6'])
    axes[1, 0].set_title('Mode Distribution')
    axes[1, 1].bar(range(len(eval_results['probabilities'])), eval_results['probabilities'], color='purple', alpha=0.7)
    axes[1, 1].axhline(np.mean(eval_results['probabilities']), color='red', linestyle='--',
                       label=f'Mean: {np.mean(eval_results["probabilities"]):.3f}')
    axes[1, 1].set_xlabel('Sample'); axes[1, 1].set_ylabel('Probability'); axes[1, 1].set_title('Best Mode Probability'); axes[1, 1].legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight'); plt.close()
    return report, report_path, json_path, fig_path


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation Tool')
    parser.add_argument('--model', type=str, default='best_multimodal_model.pth')
    parser.add_argument('--data', type=str, default='./mydata')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default='./evaluation_results')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    global DATA_DIR; DATA_DIR = args.data

    print("=" * 60)
    print("Trajectory Prediction Model Evaluation")
    print("=" * 60)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    print(f"Model loaded, device: {device}")
    print(f"Evaluating on {args.split} set...")
    eval_results = evaluate_model(model, args.data, args.split, device)
    print("\nGenerating report...")
    report, report_path, json_path, fig_path = generate_report(eval_results, args.output)
    print(report)
    print(f"Results saved: {report_path}, {json_path}, {fig_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
