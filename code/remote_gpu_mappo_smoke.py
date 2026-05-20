from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from haco.env import HACoPilotEnv, summarize_episode
from haco.experiment import scenario_config
from haco.policies import HACoSafeHeuristicPolicy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "torch_mappo"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_teacher() -> HACoSafeHeuristicPolicy:
    for path in [ROOT / "results" / "cem" / "best_policy.json", ROOT / "results" / "cem_local" / "best_policy.json"]:
        if path.exists():
            payload = json.loads(path.read_text())
            return HACoSafeHeuristicPolicy(params=payload.get("params", payload.get("best_params", {})), name="haco_safemarl_trained")
    return HACoSafeHeuristicPolicy(name="haco_safemarl_trained")


def feat(obs: dict[str, np.ndarray]) -> np.ndarray:
    usv = obs["usv_pos"] / 2000.0
    auv = ((obs["auv_pos"] - obs["usv_pos"]) / 1000.0).reshape(-1)
    task = ((obs["tasks"] - obs["auv_pos"]) / 1000.0).reshape(-1)
    depth = obs["auv_depth"] / 300.0
    return np.concatenate([usv, auv, task, obs["packet_probs"], obs["task_done"].astype(float), depth]).astype("float32")


def action_vec(usv: np.ndarray, auv: np.ndarray) -> np.ndarray:
    return np.concatenate([usv / 6.0, auv.reshape(-1) / 2.0]).astype("float32")


class Actor(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, dout))
        self.log_std = nn.Parameter(torch.full((dout,), -0.7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def vec_to_actions(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.clip(v, -1.0, 1.0)
    return v[:2] * 6.0, v[2:].reshape(3, 2) * 2.0


def rollout(policy: Actor, seed: int, explore: bool = False):
    cfg = scenario_config(["survey", "traffic", "acoustic_degradation", "emergency"][seed % 4], seed=seed, num_auvs=3)
    env = HACoPilotEnv(cfg)
    obs = env.reset()
    metrics, logps, rewards = [], [], []
    done = False
    while not done:
        x = torch.tensor(feat(obs), device=DEVICE).unsqueeze(0)
        mean = policy(x).squeeze(0)
        if explore:
            dist = Normal(mean, policy.log_std.exp())
            y = dist.rsample()
            logps.append(dist.log_prob(y).sum())
            v = torch.tanh(y).detach().cpu().numpy()
        else:
            v = mean.detach().cpu().numpy()
        usv, auv = vec_to_actions(v)
        obs, m, done = env.step(usv, auv, shield=True)
        metrics.append(m)
        rewards.append(m.task_progress / 120.0 - 0.25 * m.outage_count - 2.0 * m.auv_collision_count - float(m.usv_collision))
    s = summarize_episode(cfg, metrics, env.task_done, env.t)
    return s, logps, rewards


def main() -> None:
    torch.manual_seed(7)
    teacher = load_teacher()
    cfg = scenario_config("survey", seed=1, num_auvs=3)
    dim = feat(HACoPilotEnv(cfg).reset()).size
    actor = Actor(dim, 8).to(DEVICE)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    xs, ys = [], []
    for seed in range(12):
        cfg = scenario_config(["survey", "traffic", "acoustic_degradation", "emergency"][seed % 4], seed=1000 + seed, num_auvs=3)
        env = HACoPilotEnv(cfg)
        obs, done = env.reset(), False
        while not done:
            u, a, sh = teacher.act(obs)
            xs.append(feat(obs))
            ys.append(action_vec(u, a))
            obs, _, done = env.step(u, a, shield=sh)
    x = torch.tensor(np.stack(xs), device=DEVICE)
    y = torch.tensor(np.stack(ys), device=DEVICE)
    bc = []
    for epoch in range(10):
        pred = actor(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        bc.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu()), "device": DEVICE})

    hist = []
    for upd in range(1, 13):
        losses = []
        for seed in range(6):
            _, logps, rewards = rollout(actor, seed=2000 + upd * 10 + seed, explore=True)
            ret = torch.tensor(np.cumsum(rewards[::-1])[::-1].copy(), device=DEVICE)
            ret = (ret - ret.mean()) / (ret.std() + 1e-6)
            pg = -(torch.stack(logps) * ret).mean()
            reg = ((actor(x[:512]) - y[:512]) ** 2).mean()
            loss = pg + 0.15 * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        evals = [rollout(actor, seed=3000 + upd * 10 + i, explore=False)[0] for i in range(8)]
        rec = {
            "ppo_update": upd,
            "loss": float(np.mean(losses)),
            "mean_task_completion": float(np.mean([e["task_completion"] for e in evals])),
            "mean_outage_rate": float(np.mean([e["outage_rate"] for e in evals])),
            "mean_worst_packet": float(np.mean([e["worst_agent_packet"] for e in evals])),
            "mean_auv_collisions": float(np.mean([e["auv_collision_count"] for e in evals])),
            "device": DEVICE,
        }
        hist.append(rec)
        (OUT / "ppo_history.json").write_text(json.dumps(hist, indent=2))
    (OUT / "training_history.json").write_text(json.dumps({"bc": bc, "ppo": hist, "device": DEVICE}, indent=2))
    torch.save(actor.state_dict(), OUT / "checkpoint.pt")
    print(json.dumps(hist[-1], indent=2))


if __name__ == "__main__":
    main()
