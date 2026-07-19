from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import math

from haco.policies import BasePolicy


WORLD_SCALE = 2000.0
USV_ACTION_SCALE = 6.0
AUV_ACTION_SCALE = 2.0
USV_FEATURE_DIM = 21
AUV_FEATURE_DIM = 17
GLOBAL_FEATURE_DIM = USV_FEATURE_DIM + AUV_FEATURE_DIM + 1


def _norm_pos(pos: np.ndarray) -> np.ndarray:
    return (np.asarray(pos, dtype=np.float32) / WORLD_SCALE) * 2.0 - 1.0


def _rel(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return (np.asarray(target, dtype=np.float32) - np.asarray(source, dtype=np.float32)) / WORLD_SCALE


def _nearest(source: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float, int]:
    if positions.size == 0:
        return np.zeros(2, dtype=np.float32), 1.0, -1
    rel = positions - source[None, :]
    dist = np.linalg.norm(rel, axis=1)
    idx = int(np.argmin(dist))
    return (rel[idx] / WORLD_SCALE).astype(np.float32), float(dist[idx] / WORLD_SCALE), idx


def mask_acoustic_features(
    usv_feat: np.ndarray, auv_feat: np.ndarray, global_feat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace acoustic-link observations with neutral constants for ablations."""
    usv_feat = usv_feat.copy()
    auv_feat = auv_feat.copy()
    global_feat = global_feat.copy()

    # USV feature slots: min packet, mean packet, outage fraction.
    usv_feat[8] = 1.0
    usv_feat[9] = 1.0
    usv_feat[10] = 0.0

    # AUV feature slot: own packet probability.
    auv_feat[:, 6] = 1.0

    global_feat[:USV_FEATURE_DIM] = usv_feat
    global_feat[USV_FEATURE_DIM : USV_FEATURE_DIM + AUV_FEATURE_DIM] = auv_feat.mean(axis=0)
    return usv_feat.astype(np.float32), auv_feat.astype(np.float32), global_feat.astype(np.float32)


def build_feature_arrays(
    obs: Dict[str, np.ndarray], use_acoustic: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usv_pos = obs["usv_pos"].astype(np.float32)
    auv_pos = obs["auv_pos"].astype(np.float32)
    tasks = obs["tasks"].astype(np.float32)
    done = obs["task_done"].astype(bool)
    packets = obs["packet_probs"].astype(np.float32)
    obstacles = obs["obstacles"].astype(np.float32)
    vessel_pos = obs["vessel_pos"].astype(np.float32)
    vessel_vel = obs["vessel_vel"].astype(np.float32)

    active = ~done
    if not np.any(active):
        active = np.ones(len(auv_pos), dtype=bool)
    auv_center = auv_pos.mean(axis=0)
    task_center = tasks[active].mean(axis=0)
    low_idx = int(np.argmin(packets))
    vessel_rel, vessel_dist, vessel_idx = _nearest(usv_pos, vessel_pos)
    vessel_speed = (
        np.clip(vessel_vel[vessel_idx] / 5.0, -1.0, 1.0).astype(np.float32)
        if vessel_idx >= 0
        else np.zeros(2, dtype=np.float32)
    )
    obs_rel, obs_dist, obs_idx = _nearest(usv_pos, obstacles[:, :2])
    obs_rad = float(obstacles[obs_idx, 2] / 100.0) if obs_idx >= 0 else 0.0

    usv_feat = np.concatenate(
        [
            _norm_pos(usv_pos),
            _rel(auv_center, usv_pos),
            _rel(task_center, usv_pos),
            _rel(auv_pos[low_idx], usv_pos),
            np.array(
                [
                    packets.min(),
                    packets.mean(),
                    np.mean(packets < 0.78),
                    done.mean(),
                ],
                dtype=np.float32,
            ),
            vessel_rel,
            vessel_speed,
            np.array([vessel_dist], dtype=np.float32),
            obs_rel,
            np.array([obs_dist, obs_rad], dtype=np.float32),
        ]
    ).astype(np.float32)

    auv_feats = []
    for i in range(len(auv_pos)):
        o_rel, o_dist, o_idx = _nearest(auv_pos[i], obstacles[:, :2])
        o_rad = float(obstacles[o_idx, 2] / 100.0) if o_idx >= 0 else 0.0
        v_rel, v_dist, _ = _nearest(auv_pos[i], vessel_pos)
        auv_feats.append(
            np.concatenate(
                [
                    _norm_pos(auv_pos[i]),
                    _rel(tasks[i], auv_pos[i]),
                    _rel(usv_pos, auv_pos[i]),
                    np.array([packets[i], float(done[i])], dtype=np.float32),
                    o_rel,
                    np.array([o_dist, o_rad], dtype=np.float32),
                    v_rel,
                    np.array([v_dist], dtype=np.float32),
                    _rel(auv_center, auv_pos[i]),
                ]
            ).astype(np.float32)
        )
    auv_feat = np.vstack(auv_feats).astype(np.float32)
    global_feat = np.concatenate(
        [usv_feat, auv_feat.mean(axis=0), np.array([len(auv_feat) / 8.0], dtype=np.float32)]
    ).astype(np.float32)
    if not use_acoustic:
        return mask_acoustic_features(usv_feat, auv_feat, global_feat)
    return usv_feat, auv_feat, global_feat


def features_to_tensors(
    obs: Dict[str, np.ndarray], device: torch.device | str, use_acoustic: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    usv, auv, glob = build_feature_arrays(obs, use_acoustic=use_acoustic)
    return (
        torch.as_tensor(usv, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(auv, dtype=torch.float32, device=device),
        torch.as_tensor(glob, dtype=torch.float32, device=device).unsqueeze(0),
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HACoActorCritic(nn.Module):
    def __init__(self, hidden: int = 128, use_attention: bool = True):
        super().__init__()
        self.hidden = hidden
        self.use_attention = use_attention
        self.usv_encoder = MLP(USV_FEATURE_DIM, hidden, hidden)
        self.auv_encoder = MLP(AUV_FEATURE_DIM, hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.usv_actor = MLP(hidden * 2, 2, hidden)
        self.auv_actor = MLP(hidden * 2, 2, hidden)
        self.critic = MLP(GLOBAL_FEATURE_DIM, 1, hidden)
        self.usv_log_std = nn.Parameter(torch.full((2,), -0.6))
        self.auv_log_std = nn.Parameter(torch.full((2,), -0.7))

    def means(self, usv_feat: torch.Tensor, auv_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        usv_h = self.usv_encoder(usv_feat)
        auv_h = self.auv_encoder(auv_feat)
        if self.use_attention:
            score = self.query(usv_h) @ self.key(auv_h).T / math.sqrt(float(self.hidden))
            weight = torch.softmax(score, dim=-1)
            auv_context = weight @ auv_h
        else:
            # Standard mean aggregation gives a comparably sized MAPPO baseline
            # without the learned graph-attention mechanism.
            auv_context = auv_h.mean(dim=0, keepdim=True)
        usv_mean = torch.tanh(self.usv_actor(torch.cat([usv_h, auv_context], dim=-1))) * USV_ACTION_SCALE
        usv_context = usv_h.expand(auv_h.shape[0], -1)
        auv_mean = torch.tanh(self.auv_actor(torch.cat([auv_h, usv_context], dim=-1))) * AUV_ACTION_SCALE
        return usv_mean, auv_mean

    def value(self, global_feat: torch.Tensor) -> torch.Tensor:
        return self.critic(global_feat).squeeze(-1)

    def action_distribution(
        self, usv_feat: torch.Tensor, auv_feat: torch.Tensor
    ) -> tuple[Normal, Normal, torch.Tensor, torch.Tensor]:
        usv_mean, auv_mean = self.means(usv_feat, auv_feat)
        usv_std = self.usv_log_std.exp().expand_as(usv_mean)
        auv_std = self.auv_log_std.exp().expand_as(auv_mean)
        return Normal(usv_mean, usv_std), Normal(auv_mean, auv_std), usv_mean, auv_mean

    def sample_action(
        self, usv_feat: torch.Tensor, auv_feat: torch.Tensor, global_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        usv_dist, auv_dist, _, _ = self.action_distribution(usv_feat, auv_feat)
        usv_action = usv_dist.sample()
        auv_action = auv_dist.sample()
        log_prob = usv_dist.log_prob(usv_action).sum() + auv_dist.log_prob(auv_action).sum()
        entropy = usv_dist.entropy().sum() + auv_dist.entropy().sum()
        value = self.value(global_feat).squeeze(0)
        return usv_action.squeeze(0), auv_action, log_prob, entropy, value

    def evaluate_actions(
        self,
        usv_feat: torch.Tensor,
        auv_feat: torch.Tensor,
        global_feat: torch.Tensor,
        usv_action: torch.Tensor,
        auv_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        usv_dist, auv_dist, _, _ = self.action_distribution(usv_feat, auv_feat)
        log_prob = usv_dist.log_prob(usv_action.unsqueeze(0)).sum() + auv_dist.log_prob(auv_action).sum()
        entropy = usv_dist.entropy().sum() + auv_dist.entropy().sum()
        value = self.value(global_feat).squeeze(0)
        return log_prob, entropy, value


class TorchHACoPolicy(BasePolicy):
    name = "haco_safemarl_mappo"

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        deterministic: bool = True,
        use_acoustic: bool | None = None,
        use_shield: bool | None = None,
        name: str | None = None,
    ):
        self.checkpoint = Path(checkpoint)
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(self.checkpoint, map_location="cpu")
        hidden = int(payload.get("hidden", 128))
        args = payload.get("args", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        use_attention = bool(args.get("use_attention", True))
        self.model = HACoActorCritic(hidden=hidden, use_attention=use_attention).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.deterministic = deterministic
        self.use_acoustic = bool(args.get("use_acoustic", True)) if use_acoustic is None else use_acoustic
        self.use_shield = bool(args.get("use_shield", True)) if use_shield is None else use_shield
        self.name = name or str(args.get("policy_name", self.name))

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        with torch.no_grad():
            usv_feat, auv_feat, _ = features_to_tensors(obs, self.device, use_acoustic=self.use_acoustic)
            usv_dist, auv_dist, usv_mean, auv_mean = self.model.action_distribution(usv_feat, auv_feat)
            if self.deterministic:
                usv_action = usv_mean.squeeze(0)
                auv_action = auv_mean
            else:
                usv_action = usv_dist.sample().squeeze(0)
                auv_action = auv_dist.sample()
        return usv_action.cpu().numpy(), auv_action.cpu().numpy(), self.use_shield
