import os, sys, json
import torch
import numpy as np
from model import MultiModalVectorNet


def check():
    print("=" * 50)
    print("Smoke Test")
    print("=" * 50)

    cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    print(f"\n[1/4] Config loaded: feature_dim={cfg['feature_dim']}, hidden_dim={cfg['hidden_dim']}, k_modes={cfg['k_modes']}")

    model = MultiModalVectorNet(
        in_channels=cfg['feature_dim'],
        out_channels=cfg['out_dim'],
        k_modes=cfg['k_modes'],
        hidden_dim=cfg['hidden_dim'],
        obs_len=cfg['obs_len']
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[2/4] Model created: {total_params:,} params")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dummy_input = torch.randn(2, cfg['obs_len'], cfg['feature_dim']).to(device)
    print(f"[3/4] Input shape: {dummy_input.shape}")

    with torch.no_grad():
        out = model(dummy_input)
    print(f"[4/4] Output: trajectories={out['trajectories'].shape}, mode_probs={out['mode_probs'].shape}, uncertainty={out['uncertainty'].shape}")

    assert out['trajectories'].shape == (2, cfg['k_modes'], cfg['out_dim']), f"Bad trajectories shape: {out['trajectories'].shape}"
    assert out['mode_probs'].shape == (2, cfg['k_modes']), f"Bad mode_probs shape: {out['mode_probs'].shape}"

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == '__main__':
    check()
