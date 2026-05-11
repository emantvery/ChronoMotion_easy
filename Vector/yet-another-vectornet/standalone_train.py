#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全独立版 VectorNet 轨迹预测训练脚本
纯 Python 实现，无需任何外部依赖
"""

import os
import sys
import csv
import math
import random
import json

# 配置参数
DATA_DIR = './mydata'
OBS_LEN = 20
PRED_LEN = 30
FEATURE_DIM = 8
HIDDEN_DIM = 64
OUT_DIM = 60  # 30 timesteps * 2 (x, y)
BATCH_SIZE = 4
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
    agent_data.sort(key=lambda x: x['TIMESTAMP'])
    traj = [[d['X'], d['Y']] for d in agent_data]
    return traj

def normalize_data(data):
    """归一化数据"""
    data = [[float(v) for v in row] for row in data]
    mean_x = sum(row[0] for row in data) / len(data)
    mean_y = sum(row[1] for row in data) / len(data)
    normalized = [[row[0] - mean_x, row[1] - mean_y] for row in data]
    return normalized, (mean_x, mean_y)

def compute_features(traj):
    """计算特征：位置 + 速度 + 时间"""
    features = []
    for i in range(len(traj)):
        if i == 0:
            vel_x, vel_y = 0, 0
        else:
            vel_x = traj[i][0] - traj[i-1][0]
            vel_y = traj[i][1] - traj[i-1][1]
        features.append([traj[i][0], traj[i][1], vel_x, vel_y, i/OBS_LEN, 1, 0, 0])
    return features

def load_dataset(split='train'):
    """加载数据集"""
    data_dir = os.path.join(DATA_DIR, split)
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataset = []
    
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        data = read_csv_file(file_path)
        agent_traj = get_agent_trajectory(data)
        
        if len(agent_traj) >= OBS_LEN + PRED_LEN:
            obs_traj = agent_traj[:OBS_LEN]
            pred_traj = agent_traj[OBS_LEN:OBS_LEN+PRED_LEN]
            
            obs_norm, _ = normalize_data(obs_traj)
            pred_norm, _ = normalize_data(pred_traj)
            
            features = compute_features(obs_norm)
            target = [coord for point in pred_norm for coord in point]
            
            dataset.append((features, target))
    
    return dataset

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return max(0, x)

def relu_deriv(x):
    return 1 if x > 0 else 0

class LinearLayer:
    """简单的线性层"""
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(output_dim)] for _ in range(input_dim)]
        self.biases = [0.0] * output_dim
        self.last_output = None
    
    def forward(self, x):
        """前向传播"""
        self.input = x
        output = [0.0] * self.output_dim
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                output[i] += x[j] * self.weights[j][i]
            output[i] += self.biases[i]
        self.last_output = output
        return output
    
    def backward(self, grad_output, lr):
        """反向传播"""
        grad_input = [0.0] * self.input_dim
        
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                grad_input[j] += grad_output[i] * self.weights[j][i]
                self.weights[j][i] -= lr * grad_output[i] * self.input[j]
            self.biases[i] -= lr * grad_output[i]
        
        return grad_input

class MLP:
    """多层感知器"""
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(LinearLayer(layers[i], layers[i+1]))
    
    def forward(self, x):
        self.outputs = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self.outputs.append(x)
        return x
    
    def backward(self, grad, lr):
        for i in range(len(self.layers) - 1, -1, -1):
            grad = [grad[j] * relu_deriv(self.outputs[i+1][j]) for j in range(len(self.outputs[i+1]))]
            grad = self.layers[i].backward(grad, lr)
        return grad

class SubGraph:
    """子图模块"""
    def __init__(self, in_channels, hidden_dim, num_layers=3):
        self.mlp = MLP([in_channels] + [hidden_dim] * num_layers)
    
    def forward(self, x):
        return self.mlp.forward(x)
    
    def backward(self, grad, lr):
        return self.mlp.backward(grad, lr)

class HGNN:
    """层次图神经网络"""
    def __init__(self, in_channels, out_channels):
        self.subgraph = SubGraph(in_channels, HIDDEN_DIM)
        self.fc1 = LinearLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.fc2 = LinearLayer(HIDDEN_DIM, out_channels)
    
    def forward(self, batch):
        """batch: list of sequences, each sequence is list of features"""
        batch_size = len(batch)
        all_outputs = []
        
        for seq in batch:
            seq_outputs = []
            for point in seq:
                out = self.subgraph.forward(point)
                seq_outputs.append(out)
            
            # 聚合：取平均
            agg = [0.0] * HIDDEN_DIM
            for out in seq_outputs:
                for i in range(HIDDEN_DIM):
                    agg[i] += out[i] / len(seq_outputs)
            
            all_outputs.append(agg)
        
        # 全局处理
        results = []
        for agg in all_outputs:
            h = [relu(x) for x in self.fc1.forward(agg)]
            pred = self.fc2.forward(h)
            results.append(pred)
        
        return results
    
    def backward(self, grads, lr):
        """反向传播"""
        for i, grad in enumerate(grads):
            # fc2
            grad = self.fc2.backward(grad, lr)
            # fc1 (apply relu derivative)
            h = self.fc1.last_output
            grad = [grad[j] * relu_deriv(h[j]) for j in range(len(h))]
            grad = self.fc1.backward(grad, lr)

def mse_loss(preds, targets):
    """计算 MSE 损失"""
    total_loss = 0.0
    grads = []
    
    for pred, target in zip(preds, targets):
        grad = []
        loss = 0.0
        for p, t in zip(pred, target):
            diff = p - t
            loss += diff * diff
            grad.append(2 * diff)
        total_loss += loss / len(pred)
        grads.append([g / len(pred) for g in grad])
    
    return total_loss / len(preds), grads

def train(model, train_data, lr, batch_size):
    """训练一个 epoch"""
    random.shuffle(train_data)
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        preds = model.forward(features)
        loss, grads = mse_loss(preds, targets)
        model.backward(grads, lr)
        
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def evaluate(model, data):
    """评估模型"""
    features = [item[0] for item in data]
    targets = [item[1] for item in data]
    
    preds = model.forward(features)
    loss, _ = mse_loss(preds, targets)
    
    return loss

def save_model(model, path):
    """保存模型"""
    model_data = {
        'subgraph_layers': [],
        'fc1_weights': model.fc1.weights,
        'fc1_biases': model.fc1.biases,
        'fc2_weights': model.fc2.weights,
        'fc2_biases': model.fc2.biases
    }
    
    for layer in model.subgraph.mlp.layers:
        model_data['subgraph_layers'].append({
            'weights': layer.weights,
            'biases': layer.biases
        })
    
    with open(path, 'w') as f:
        json.dump(model_data, f)

def main():
    print("=== 基于 PyTorch 与图神经网络的轻量化自动驾驶轨迹预测系统 ===")
    print("=== 纯 Python 独立版 ===")
    
    print("\n1. 加载数据集...")
    train_data = load_dataset('train')
    val_data = load_dataset('val')
    
    if not train_data:
        print("错误：训练数据集为空！")
        return
    if not val_data:
        print("错误：验证数据集为空！")
        return
    
    print(f"   训练样本: {len(train_data)}")
    print(f"   验证样本: {len(val_data)}")
    
    print("\n2. 创建模型...")
    model = HGNN(FEATURE_DIM, OUT_DIM)
    print("   模型创建完成")
    
    print("\n3. 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_data, LR, BATCH_SIZE)
        val_loss = evaluate(model, val_data)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'best_model.json')
            print("  Best model saved!")
    
    print("\n=== 训练完成 ===")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("模型已保存为: best_model.json")

if __name__ == '__main__':
    main()