#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 VectorNet 轨迹预测训练脚本
无需外部依赖，直接使用 Python 标准库读取 CSV
"""

import os
import sys
import csv
import math

# 尝试导入 torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")
    print(f"Error: {e}")
    print("\nPlease install PyTorch first:")
    print("pip install torch torchvision")

try:
    import numpy as np
    EXTRA_AVAILABLE = True
except ImportError:
    EXTRA_AVAILABLE = False
    print("Warning: numpy not available")

# 配置参数
DATA_DIR = './mydata'
OBS_LEN = 20
PRED_LEN = 30
FEATURE_DIM = 8
HIDDEN_DIM = 64
OUT_DIM = 60  # 30 timesteps * 2 (x, y)
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001

def read_csv_file(file_path):
    """使用 Python 标准库读取 CSV 文件"""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            timestamp = float(row[0])
            track_id = row[1]
            obj_type = row[2]
            x = float(row[3])
            y = float(row[4])
            city_name = row[5]
            data.append({
                'TIMESTAMP': timestamp,
                'TRACK_ID': track_id,
                'OBJECT_TYPE': obj_type,
                'X': x,
                'Y': y,
                'CITY_NAME': city_name
            })
    return data

def get_agent_trajectory(data):
    """从数据中提取 AGENT 的轨迹"""
    agent_data = [d for d in data if d['OBJECT_TYPE'] == 'AGENT']
    # 按时间戳排序
    agent_data.sort(key=lambda x: x['TIMESTAMP'])
    traj = [[d['X'], d['Y']] for d in agent_data]
    return np.array(traj, dtype=np.float32)

def test_data_loading():
    """测试数据加载"""
    print("\n=== Testing Data Loading ===")
    train_dir = os.path.join(DATA_DIR, 'train')
    files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
    
    if not files:
        print(f"No CSV files found in {train_dir}")
        return False
    
    print(f"Found {len(files)} files in train directory")
    
    # 读取第一个文件测试
    file_path = os.path.join(train_dir, files[0])
    print(f"\nTesting file: {files[0]}")
    data = read_csv_file(file_path)
    print(f"Total rows: {len(data)}")
    
    agent_data = [d for d in data if d['OBJECT_TYPE'] == 'AGENT']
    print(f"AGENT rows: {len(agent_data)}")
    
    if len(agent_data) >= OBS_LEN + PRED_LEN:
        print(f"OK: Data has enough timesteps ({len(agent_data)} >= {OBS_LEN + PRED_LEN})")
        return True
    else:
        print(f"ERROR: Data has insufficient timesteps ({len(agent_data)} < {OBS_LEN + PRED_LEN})")
        return False

def main():
    if not TORCH_AVAILABLE:
        print("\n=== Please Install Dependencies ===")
        print("1. Install PyTorch:")
        print("   pip install torch torchvision")
        print("\n2. Install numpy (optional for visualization):")
        print("   pip install numpy matplotlib tqdm")
        print("\n3. After installation, run:")
        print("   python simple_train.py")
        return
    
    if not EXTRA_AVAILABLE:
        print("Warning: numpy not installed, some functionality may be limited")
        return
    
    # 测试数据加载
    if not test_data_loading():
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    # 定义模型（需要在 torch 可用时定义）
    class SubGraph(nn.Module):
        def __init__(self, in_channels, hidden_dim, num_layers=3):
            super(SubGraph, self).__init__()
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(nn.Linear(in_channels, hidden_dim))
                else:
                    self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        
        def forward(self, x):
            for conv in self.convs:
                x = F.relu(conv(x))
            return x
    
    class HGNN(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(HGNN, self).__init__()
            self.subgraph = SubGraph(in_channels, HIDDEN_DIM)
            self.fc1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.fc2 = nn.Linear(HIDDEN_DIM, out_channels)
        
        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(-1, FEATURE_DIM)
            sub_out = self.subgraph(x)
            sub_out = sub_out.view(batch_size, OBS_LEN, HIDDEN_DIM)
            global_out = torch.mean(sub_out, dim=1)
            x = F.relu(self.fc1(global_out))
            pred = self.fc2(x)
            return pred
    
    class TrajectoryDataset(Dataset):
        def __init__(self, data_dir, split='train'):
            self.data_dir = os.path.join(data_dir, split)
            self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            file_path = os.path.join(self.data_dir, self.files[idx])
            data = read_csv_file(file_path)
            agent_traj = get_agent_trajectory(data)
            
            obs_traj = agent_traj[:OBS_LEN]
            pred_traj = agent_traj[OBS_LEN:OBS_LEN+PRED_LEN]
            
            obs_traj_norm = obs_traj - obs_traj[0]
            pred_traj_norm = pred_traj - obs_traj[0]
            
            features = []
            for i in range(len(obs_traj_norm)):
                if i == 0:
                    vel_x, vel_y = 0, 0
                else:
                    vel_x = obs_traj_norm[i, 0] - obs_traj_norm[i-1, 0]
                    vel_y = obs_traj_norm[i, 1] - obs_traj_norm[i-1, 1]
                features.append([obs_traj_norm[i, 0], obs_traj_norm[i, 1], vel_x, vel_y, i/OBS_LEN, 1, 0, 0])
            
            features = np.array(features, dtype=np.float32)
            target = pred_traj_norm.flatten().astype(np.float32)
            
            return torch.from_numpy(features), torch.from_numpy(target)
    
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return features, targets
    
    def train(model, train_loader, optimizer, device):
        model.train()
        total_loss = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def evaluate(model, val_loader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item() * features.size(0)
        
        return total_loss / len(val_loader.dataset)
    
    # 创建数据集和数据加载器
    train_dataset = TrajectoryDataset(DATA_DIR, split='train')
    val_dataset = TrajectoryDataset(DATA_DIR, split='val')
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    batch_size_train = min(BATCH_SIZE, len(train_dataset))
    batch_size_val = min(BATCH_SIZE, len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, collate_fn=collate_fn)
    
    # 创建模型
    model = HGNN(FEATURE_DIM, OUT_DIM).to(device)
    print(f"\nModel created successfully")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 训练循环
    print("\n=== Starting Training ===")
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss:   {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  Best model saved!')
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("\nModel saved as: best_model.pth")

if __name__ == '__main__':
    main()