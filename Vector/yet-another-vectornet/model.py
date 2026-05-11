import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3):
        super(MLPEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(nn.Linear(in_channels, hidden_dim))
            else:
                self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x)))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        out = torch.sum(x * attn_weights, dim=1)
        return out


class MultiModalVectorNet(nn.Module):
    def __init__(self, in_channels, out_channels, k_modes=3, hidden_dim=128, obs_len=20):
        super(MultiModalVectorNet, self).__init__()
        self.k_modes = k_modes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.obs_len = obs_len

        self.encoder = MLPEncoder(in_channels, hidden_dim, num_layers=3)
        self.attn_pool = AttentionPooling(hidden_dim)

        self.traj_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k_modes)
        )

        self.traj_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_channels)
            ) for _ in range(k_modes)
        ])

        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(-1, self.in_channels)
        features = self.encoder(x)
        features = features.view(batch_size, self.obs_len, self.hidden_dim)

        pooled = self.attn_pool(features)
        encoded = self.traj_encoder(pooled)

        mode_logits = self.mode_selector(encoded)
        mode_probs = F.softmax(mode_logits, dim=1)

        traj_preds = []
        for predictor in self.traj_predictors:
            traj_pred = predictor(encoded)
            traj_preds.append(traj_pred)
        traj_preds = torch.stack(traj_preds, dim=1)

        uncertainty = F.softplus(self.uncertainty_predictor(encoded))

        return {
            'trajectories': traj_preds,
            'mode_probs': mode_probs,
            'uncertainty': uncertainty,
            'mode_logits': mode_logits
        }


class LinearPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, obs_len=20):
        super(LinearPredictor, self).__init__()
        self.linear = nn.Linear(in_channels * obs_len, out_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.linear(x)


class LSTMPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class MultiModalLoss(nn.Module):
    def __init__(self, k_modes=3, alpha=0.7, beta=0.3):
        super(MultiModalLoss, self).__init__()
        self.k_modes = k_modes
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        traj_preds = outputs['trajectories']
        mode_logits = outputs['mode_logits']

        all_traj_loss = []
        for k in range(self.k_modes):
            traj_loss = F.mse_loss(traj_preds[:, k, :], targets, reduction='none')
            traj_loss = traj_loss.mean(dim=1)
            all_traj_loss.append(traj_loss)
        all_traj_loss = torch.stack(all_traj_loss, dim=1)

        best_traj_loss, best_mode = all_traj_loss.min(dim=1)

        mode_labels = best_mode.detach()
        mode_loss = F.cross_entropy(mode_logits, mode_labels)

        traj_loss = best_traj_loss.mean()
        total_loss = self.alpha * traj_loss + self.beta * mode_loss

        return total_loss, {
            'traj_loss': traj_loss.item(),
            'mode_loss': mode_loss.item(),
            'best_mode': best_mode.float().mean().item()
        }
