#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量化自动驾驶轨迹预测系统 - 增强版
1. 多模态轨迹预测 - 输出K条可能的轨迹及其概率
2. 不确定性量化 - 预测置信区间
3. 周围车辆特征 - 利用CSV中OTHERS数据
"""

import os, csv, json, argparse, time
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import MLPEncoder, AttentionPooling, MultiModalVectorNet, LinearPredictor, LSTMPredictor, MultiModalLoss

DEFAULT_CONFIG = 'config.json'


def load_config(config_path=DEFAULT_CONFIG):
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg


def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append({
                'TIMESTAMP': float(row[0]),
                'TRACK_ID': row[1],
                'OBJECT_TYPE': row[2],
                'X': float(row[3]),
                'Y': float(row[4]),
                'CITY_NAME': row[5]
            })
    return data


def get_agent_trajectory(data):
    agent_data = [d for d in data if d['OBJECT_TYPE'] == 'AGENT']
    agent_data.sort(key=lambda x: x['TIMESTAMP'])
    return [[d['X'], d['Y']] for d in agent_data]


def get_others_trajectories(data, agent_timestamps):
    others = {}
    for d in data:
        if d['OBJECT_TYPE'] in ('AGENT', 'AV'):
            continue
        tid = d['TRACK_ID']
        if tid not in others:
            others[tid] = []
        others[tid].append((d['TIMESTAMP'], d['X'], d['Y']))
    result = []
    for tid, pts in others.items():
        pts.sort(key=lambda x: x[0])
        traj = [[p[1], p[2]] for p in pts]
        if len(traj) >= 2:
            result.append(traj)
    return result


def compute_features_for_agent(obs_norm, obs_len):
    features = []
    for i in range(len(obs_norm)):
        if i == 0:
            vel_x, vel_y = 0, 0
        else:
            vel_x = obs_norm[i, 0] - obs_norm[i-1, 0]
            vel_y = obs_norm[i, 1] - obs_norm[i-1, 1]
        features.append([obs_norm[i, 0], obs_norm[i, 1], vel_x, vel_y, i / obs_len, 1, 0, 0])
    return np.array(features, dtype=np.float32)


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, split='train', obs_len=20, pred_len=30, feature_dim=8):
        self.data_dir = os.path.join(data_dir, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = read_csv_file(file_path)
        agent_traj = get_agent_trajectory(data)

        obs_traj = agent_traj[:self.obs_len]
        pred_traj = agent_traj[self.obs_len:self.obs_len + self.pred_len]

        obs_arr = np.array(obs_traj, dtype=np.float32)
        pred_arr = np.array(pred_traj, dtype=np.float32)

        origin = obs_arr[0].copy()
        obs_norm = obs_arr - origin
        pred_norm = pred_arr - origin

        features = compute_features_for_agent(obs_norm, self.obs_len)

        others_trajs = get_others_trajectories(data, [d['TIMESTAMP'] for d in data if d['OBJECT_TYPE'] == 'AGENT'])
        if others_trajs:
            agent_last_obs = obs_arr[-1]
            nearest_dist = float('inf')
            nearest_rel = np.array([0.0, 0.0])
            for ot in others_trajs:
                ot_arr = np.array(ot, dtype=np.float32)
                dist = np.linalg.norm(ot_arr[-1] - agent_last_obs)
                if dist < nearest_dist and dist > 0:
                    nearest_dist = dist
                    nearest_rel = (ot_arr[-1] - origin).astype(np.float32)
            nearest_dist = min(nearest_dist, 999.0)
            features[:, 5] = nearest_dist / 100.0
            features[:, 6] = nearest_rel[0] / 100.0
            features[:, 7] = nearest_rel[1] / 100.0

        target = pred_norm.flatten().astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(target), self.files[idx]


def collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    return features, targets, filenames


def compute_ade(pred, target, pred_len=30):
    pred_coords = pred.reshape(-1, pred_len, 2)
    target_coords = target.reshape(-1, pred_len, 2)
    diff = pred_coords - target_coords
    dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
    return dist.mean(dim=1)


def compute_fde(pred, target):
    pred_final = pred[:, -2:].reshape(-1, 2)
    target_final = target[:, -2:].reshape(-1, 2)
    diff = pred_final - target_final
    dist = torch.sqrt(torch.sum(diff ** 2, dim=1))
    return dist.mean()


def train_epoch(model, loader, optimizer, criterion, device, k_modes=3, pred_len=30):
    model.train()
    total_loss = 0
    metrics = {'ade': 0, 'fde': 0, 'mse': 0}
    count = 0

    for features, targets, _ in tqdm(loader, desc='Training'):
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss, loss_info = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            traj_preds = outputs['trajectories']
            best_ade = float('inf')
            best_traj = traj_preds[:, 0, :]
            for k in range(k_modes):
                ade_vals = compute_ade(traj_preds[:, k, :], targets, pred_len)
                mean_ade = ade_vals.mean()
                if mean_ade < best_ade:
                    best_ade = mean_ade
                    best_traj = traj_preds[:, k, :]
            metrics['ade'] += best_ade.item()
            metrics['fde'] += compute_fde(best_traj, targets).item()
            metrics['mse'] += F.mse_loss(best_traj, targets).item()
            count += 1

    return total_loss / count, {k: v / count for k, v in metrics.items()}


def evaluate(model, loader, criterion, device, k_modes=3, pred_len=30):
    model.eval()
    total_loss = 0
    metrics = {'ade': 0, 'fde': 0, 'mse': 0}
    count = 0

    with torch.no_grad():
        for features, targets, _ in loader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss, loss_info = criterion(outputs, targets)

            total_loss += loss.item()

            traj_preds = outputs['trajectories']
            best_ade = float('inf')
            best_traj = traj_preds[:, 0, :]
            for k in range(k_modes):
                ade_vals = compute_ade(traj_preds[:, k, :], targets, pred_len)
                mean_ade = ade_vals.mean()
                if mean_ade < best_ade:
                    best_ade = mean_ade
                    best_traj = traj_preds[:, k, :]
            metrics['ade'] += best_ade.item()
            metrics['fde'] += compute_fde(best_traj, targets).item()
            metrics['mse'] += F.mse_loss(best_traj, targets).item()
            count += 1

    return total_loss / count, {k: v / count for k, v in metrics.items()}


class TrajectoryVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None

    def visualize_prediction(self, obs_traj, gt_traj, pred_trajs, mode_probs,
                             uncertainties, save_path=None, show=True, pred_len=30):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))

        self.ax.clear()

        obs_traj = np.array(obs_traj)
        gt_traj = np.array(gt_traj)
        origin = obs_traj[0]
        obs_traj = obs_traj - origin
        gt_traj = gt_traj - origin

        if len(pred_trajs.shape) == 2 and pred_trajs.shape[1] == pred_len * 2:
            pred_trajs = pred_trajs.reshape(-1, pred_len, 2)
        pred_trajs = np.array(pred_trajs) - origin

        self.ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'b-o', linewidth=2, markersize=4, label='Observed')
        self.ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-s', linewidth=2, markersize=4, label='Ground Truth')

        colors = ['#FF6B6B', '#4ECDC4', '#9B59B6']
        sorted_indices = np.argsort(mode_probs)[::-1]

        for i, idx in enumerate(sorted_indices):
            prob = mode_probs[idx]
            pred = pred_trajs[idx]
            self.ax.plot(pred[:, 0], pred[:, 1], '--', color=colors[i % len(colors)],
                         linewidth=2, label=f'Mode {i+1} (p={prob:.3f})')

            for t in range(0, pred_len, 5):
                x, y = pred[t]
                u = uncertainties[t * 2] if t * 2 < len(uncertainties) else 1.0
                v = uncertainties[t * 2 + 1] if t * 2 + 1 < len(uncertainties) else 1.0
                ellipse = plt.matplotlib.patches.Ellipse(
                    (x, y), width=2 * u, height=2 * v,
                    color=colors[i % len(colors)], alpha=0.3
                )
                self.ax.add_patch(ellipse)

        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_title('Multi-Modal Trajectory Prediction', fontsize=14)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')

        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Trajectory Prediction')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='Config file path')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['lr'] = args.lr

    DATA_DIR = cfg['data_dir']
    OBS_LEN = cfg['obs_len']
    PRED_LEN = cfg['pred_len']
    FEATURE_DIM = cfg['feature_dim']
    HIDDEN_DIM = cfg['hidden_dim']
    K_MODES = cfg['k_modes']
    OUT_DIM = cfg['out_dim']
    BATCH_SIZE = cfg['batch_size']
    EPOCHS = cfg['epochs']
    LR = cfg['lr']
    NUM_WORKERS = cfg.get('num_workers', 2)
    DEVICE = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    print("=" * 60)
    print("Multi-Modal Trajectory Prediction System")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Config: obs_len={OBS_LEN}, pred_len={PRED_LEN}, feature_dim={FEATURE_DIM}")
    print(f"        hidden_dim={HIDDEN_DIM}, k_modes={K_MODES}, batch_size={BATCH_SIZE}")

    train_dataset = TrajectoryDataset(DATA_DIR, split='train', obs_len=OBS_LEN, pred_len=PRED_LEN, feature_dim=FEATURE_DIM)
    val_dataset = TrajectoryDataset(DATA_DIR, split='val', obs_len=OBS_LEN, pred_len=PRED_LEN, feature_dim=FEATURE_DIM)

    print(f"\nDataset: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                            num_workers=NUM_WORKERS)

    model = MultiModalVectorNet(in_channels=FEATURE_DIM, out_channels=OUT_DIM,
                                k_modes=K_MODES, hidden_dim=HIDDEN_DIM, obs_len=OBS_LEN).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    linear_model = LinearPredictor(FEATURE_DIM, OUT_DIM, obs_len=OBS_LEN).to(DEVICE)
    lstm_model = LSTMPredictor(FEATURE_DIM, HIDDEN_DIM // 2, OUT_DIM).to(DEVICE)

    criterion = MultiModalLoss(k_modes=K_MODES)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg.get('weight_decay', 0.01))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining ({EPOCHS} epochs)...")
    print("-" * 60)

    best_val_loss = float('inf')
    best_ade = float('inf')

    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                                k_modes=K_MODES, pred_len=PRED_LEN)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, DEVICE,
                                         k_modes=K_MODES, pred_len=PRED_LEN)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"ADE: {val_metrics['ade']:.4f} | FDE: {val_metrics['fde']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ade = val_metrics['ade']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'ade': best_ade,
                'config': cfg
            }, 'best_multimodal_model.pth')
            print(f"  [NEW BEST] Val Loss: {val_loss:.4f}, ADE: {best_ade:.4f}")

    print("-" * 60)
    print("Training complete!")

    checkpoint = torch.load('best_multimodal_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = TrajectoryDataset(DATA_DIR, split='test', obs_len=OBS_LEN, pred_len=PRED_LEN, feature_dim=FEATURE_DIM)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    vectornet_metrics = {'ade': [], 'fde': [], 'mse': []}
    model.eval()
    with torch.no_grad():
        for features, targets, _ in test_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(features)
            traj_preds = outputs['trajectories']
            best_ade = float('inf')
            best_traj = traj_preds[:, 0, :]
            for k in range(K_MODES):
                ade_vals = compute_ade(traj_preds[:, k, :], targets, PRED_LEN)
                if ade_vals.mean() < best_ade:
                    best_ade = ade_vals.mean()
                    best_traj = traj_preds[:, k, :]
            vectornet_metrics['ade'].append(best_ade.item())
            vectornet_metrics['fde'].append(compute_fde(best_traj, targets).item())
            vectornet_metrics['mse'].append(F.mse_loss(best_traj, targets).item())

    print(f"\nTest Results:")
    print(f"  ADE: {np.mean(vectornet_metrics['ade']):.4f} m")
    print(f"  FDE: {np.mean(vectornet_metrics['fde']):.4f} m")
    print(f"  MSE: {np.mean(vectornet_metrics['mse']):.6f}")
    print(f"  Params: {total_params:,}")

    linear_criterion = nn.MSELoss()
    linear_optimizer = optim.Adam(linear_model.parameters(), lr=LR)
    print(f"\nBaseline - Linear Model (training 50 epochs)...")
    for epoch in range(50):
        linear_model.train()
        for features, targets, _ in train_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            linear_optimizer.zero_grad()
            outputs = linear_model(features)
            loss = linear_criterion(outputs, targets)
            loss.backward()
            linear_optimizer.step()

    linear_metrics = {'ade': [], 'fde': [], 'mse': []}
    linear_model.eval()
    with torch.no_grad():
        for features, targets, _ in test_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = linear_model(features)
            linear_metrics['ade'].append(compute_ade(outputs, targets, PRED_LEN).mean().item())
            linear_metrics['fde'].append(compute_fde(outputs, targets).item())
            linear_metrics['mse'].append(F.mse_loss(outputs, targets).item())

    print(f"  Linear ADE: {np.mean(linear_metrics['ade']):.4f} m")
    print(f"  Linear FDE: {np.mean(linear_metrics['fde']):.4f} m")

    ade_improvement = (np.mean(linear_metrics['ade']) - np.mean(vectornet_metrics['ade'])) / max(np.mean(linear_metrics['ade']), 1e-8) * 100
    print(f"\nImprovement over linear: {ade_improvement:.1f}%")

    torch.save(model.state_dict(), 'final_multimodal_model.pth')
    print(f"\nModels saved: best_multimodal_model.pth, final_multimodal_model.pth")

    print(f"\nGenerating demo...")
    test_dataset_demo = TrajectoryDataset(DATA_DIR, split='test', obs_len=OBS_LEN, pred_len=PRED_LEN, feature_dim=FEATURE_DIM)
    vis = TrajectoryVisualizer()
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(test_dataset_demo))):
            features, targets, filename = test_dataset_demo[i]
            features = features.unsqueeze(0).to(DEVICE)
            outputs = model(features)
            pred_trajs = outputs['trajectories'][0].cpu().numpy()
            mode_probs = outputs['mode_probs'][0].cpu().numpy()
            uncertainties = outputs['uncertainty'][0].cpu().numpy()
            obs_traj = features[0, :, :2].cpu().numpy()
            gt_traj = targets.numpy().reshape(-1, 2)
            save_path = f'demo_pred_{i+1}.png'
            vis.visualize_prediction(obs_traj, gt_traj, pred_trajs, mode_probs, uncertainties, save_path, show=False, pred_len=PRED_LEN)
            print(f"  {filename}: best_mode={np.argmax(mode_probs)+1}, prob={mode_probs.max():.3f}")
    vis.close()

    return model, vectornet_metrics


if __name__ == '__main__':
    main()
