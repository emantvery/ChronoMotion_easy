#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行轨迹预测工具
"""
import os, csv, argparse, json
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MultiModalVectorNet

OBS_LEN = 20
PRED_LEN = 30
FEATURE_DIM = 8
OUT_DIM = PRED_LEN * 2
K_MODES = 3


def load_model(model_path, device='cpu'):
    model = MultiModalVectorNet(in_channels=FEATURE_DIM, out_channels=OUT_DIM,
                                k_modes=K_MODES, hidden_dim=128, obs_len=OBS_LEN)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


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


def predict(model, obs_traj, device='cpu'):
    obs_norm = obs_traj - obs_traj[0]
    features = compute_features(obs_norm)
    input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    pred_trajs = outputs['trajectories'][0].cpu().numpy()
    mode_probs = outputs['mode_probs'][0].cpu().numpy()
    uncertainty = outputs['uncertainty'][0].cpu().numpy()
    return {'trajectories': pred_trajs, 'probabilities': mode_probs, 'uncertainty': uncertainty}


def visualize_prediction(obs_traj, result, output_path, gt_traj=None):
    plt.figure(figsize=(10, 8))
    obs_norm = obs_traj - obs_traj[0]
    origin = obs_traj[0]
    plt.plot(obs_norm[:, 0], obs_norm[:, 1], 'b-o', linewidth=2, markersize=4, label='Observed')
    if gt_traj is not None:
        gt_norm = gt_traj - origin
        plt.plot(gt_norm[:, 0], gt_norm[:, 1], 'g-s', linewidth=2, markersize=4, label='Ground Truth')
    colors = ['#FF6B6B', '#4ECDC4', '#9B59B6']
    sorted_indices = np.argsort(result['probabilities'])[::-1]
    for i, idx in enumerate(sorted_indices):
        prob = result['probabilities'][idx]
        pred = result['trajectories'][idx].reshape(-1, 2)
        plt.plot(pred[:, 0], pred[:, 1], '--', color=colors[i], linewidth=2, label=f'Mode {i+1} (p={prob:.3f})')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.title('Trajectory Prediction')
    plt.legend(); plt.grid(True, alpha=0.3); plt.axis('equal')
    plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()


def batch_predict(model, input_dir, output_dir, device='cpu', max_files=None):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if max_files is not None and max_files > 0:
        csv_files = csv_files[:max_files]
        print(f"Limiting to first {len(csv_files)} files")
    results = []
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        try:
            file_path = os.path.join(input_dir, csv_file)
            traj = read_csv_trajectory(file_path)
            if len(traj) < OBS_LEN:
                print(f"  Skipped: insufficient data"); continue
            obs_traj = np.array(traj[:OBS_LEN])
            gt_traj = np.array(traj[OBS_LEN:OBS_LEN + PRED_LEN]) if len(traj) >= OBS_LEN + PRED_LEN else None
            result = predict(model, obs_traj, device)
            output_path = os.path.join(output_dir, os.path.splitext(csv_file)[0] + '_pred.png')
            visualize_prediction(obs_traj, result, output_path, gt_traj)
            results.append({'file': csv_file, 'best_mode': int(np.argmax(result['probabilities'])) + 1,
                            'best_prob': float(np.max(result['probabilities'])),
                            'avg_uncertainty': float(result['uncertainty'].mean())})
            print(f"  Done: best mode={results[-1]['best_mode']}, prob={results[-1]['best_prob']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
    results_path = os.path.join(output_dir, 'predictions.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Trajectory Prediction Tool')
    parser.add_argument('--model', type=str, default='best_multimodal_model.pth')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_files', type=int, default=None, help='Max files to process in batch mode')
    args = parser.parse_args()

    print("=" * 50)
    print("Trajectory Prediction Tool")
    print("=" * 50)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Loading model: {args.model}")
    model = load_model(args.model, device)
    print(f"Model loaded, device: {device}")

    if os.path.isfile(args.input):
        print(f"\nProcessing: {args.input}")
        traj = read_csv_trajectory(args.input)
        if len(traj) < OBS_LEN:
            print(f"Error: insufficient data ({len(traj)} points)"); return
        obs_traj = np.array(traj[:OBS_LEN])
        gt_traj = np.array(traj[OBS_LEN:OBS_LEN + PRED_LEN]) if len(traj) >= OBS_LEN + PRED_LEN else None
        result = predict(model, obs_traj, device)
        print(f"\nBest mode: {np.argmax(result['probabilities']) + 1}")
        print(f"Probabilities: {result['probabilities']}")
        if args.visualize:
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(args.output, 'prediction.png')
            visualize_prediction(obs_traj, result, output_path, gt_traj)
            print(f"Visualization saved: {output_path}")
        os.makedirs(args.output, exist_ok=True)
        json_path = os.path.join(args.output, 'prediction.json')
        with open(json_path, 'w') as f:
            json.dump({'trajectories': result['trajectories'].tolist(),
                       'probabilities': result['probabilities'].tolist(),
                       'uncertainty': result['uncertainty'].tolist()}, f, indent=2)
        print(f"Results saved: {json_path}")
    elif os.path.isdir(args.input):
        print(f"\nBatch: {args.input}")
        batch_predict(model, args.input, args.output, device, args.max_files)
    else:
        print(f"Error: input not found: {args.input}")

    print("\nDone!")


if __name__ == '__main__':
    main()
