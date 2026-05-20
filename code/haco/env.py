from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ScenarioConfig:
    seed: int = 0
    world_size: float = 2000.0
    num_auvs: int = 3
    num_surface_vessels: int = 3
    num_obstacles: int = 8
    steps: int = 300
    dt: float = 2.0
    acoustic_range: float = 520.0
    packet_threshold: float = 0.78
    noise_level: float = 0.15
    dropout_bias: float = 0.0
    traffic: bool = True
    emergency: bool = False
    scenario_name: str = "survey"


@dataclass
class StepMetrics:
    task_progress: float
    outage_count: int
    packet_probs: np.ndarray
    usv_collision: bool
    auv_collision_count: int
    colregs_violation: bool
    energy: float
    smoothness: float
    shield_interventions: int


def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec / norm * max_norm


class HACoPilotEnv:
    """A lightweight, deterministic-by-seed simulator for USV-AUV studies.

    This environment is intentionally compact. It is used to validate scenario
    generation, metrics, baselines, and table generation before GPU MARL runs.
    """

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self.usv_pos = np.zeros(2)
        self.usv_prev_action = np.zeros(2)
        self.auv_pos = np.zeros((cfg.num_auvs, 2))
        self.auv_prev_action = np.zeros((cfg.num_auvs, 2))
        self.auv_depth = np.zeros(cfg.num_auvs)
        self.tasks = np.zeros((cfg.num_auvs, 2))
        self.task_done = np.zeros(cfg.num_auvs, dtype=bool)
        self.obstacles = np.zeros((cfg.num_obstacles, 3))
        self.vessel_pos = np.zeros((cfg.num_surface_vessels, 2))
        self.vessel_vel = np.zeros((cfg.num_surface_vessels, 2))
        self.energy = 0.0
        self.shield_interventions = 0
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        c = self.cfg
        self.t = 0
        center = c.world_size / 2.0
        self.usv_pos = np.array([center, center], dtype=float)
        self.usv_prev_action = np.zeros(2)
        angles = np.linspace(0, 2 * np.pi, c.num_auvs, endpoint=False)
        radius = 180.0
        self.auv_pos = np.column_stack(
            [center + radius * np.cos(angles), center + radius * np.sin(angles)]
        )
        self.auv_pos += self.rng.normal(0, 30, size=self.auv_pos.shape)
        self.auv_prev_action = np.zeros((c.num_auvs, 2))
        self.auv_depth = self.rng.uniform(60.0, 180.0, size=c.num_auvs)
        self.tasks = self.rng.uniform(250.0, c.world_size - 250.0, size=(c.num_auvs, 2))
        self.task_done[:] = False
        self.obstacles = np.column_stack(
            [
                self.rng.uniform(250.0, c.world_size - 250.0, size=c.num_obstacles),
                self.rng.uniform(250.0, c.world_size - 250.0, size=c.num_obstacles),
                self.rng.uniform(35.0, 80.0, size=c.num_obstacles),
            ]
        )
        self.vessel_pos = self.rng.uniform(100.0, c.world_size - 100.0, size=(c.num_surface_vessels, 2))
        headings = self.rng.uniform(0, 2 * np.pi, size=c.num_surface_vessels)
        speeds = self.rng.uniform(1.0, 3.5, size=c.num_surface_vessels)
        self.vessel_vel = np.column_stack([np.cos(headings), np.sin(headings)]) * speeds[:, None]
        if not c.traffic:
            self.vessel_vel[:] = 0.0
        self.energy = 0.0
        self.shield_interventions = 0
        return self.observe()

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            "usv_pos": self.usv_pos.copy(),
            "auv_pos": self.auv_pos.copy(),
            "auv_depth": self.auv_depth.copy(),
            "tasks": self.tasks.copy(),
            "task_done": self.task_done.copy(),
            "obstacles": self.obstacles.copy(),
            "vessel_pos": self.vessel_pos.copy(),
            "vessel_vel": self.vessel_vel.copy(),
            "packet_probs": self.packet_probs(),
        }

    def packet_probs(self) -> np.ndarray:
        c = self.cfg
        horizontal = np.linalg.norm(self.auv_pos - self.usv_pos[None, :], axis=1)
        slant = np.sqrt(horizontal**2 + self.auv_depth**2)
        beam_margin = c.acoustic_range * 0.95 - horizontal
        depth_penalty = np.maximum(0.0, self.auv_depth - 120.0) / 400.0
        snr_margin = (c.acoustic_range - slant) / max(c.acoustic_range, 1.0) - c.noise_level - depth_penalty
        prob = 1.0 / (1.0 + np.exp(-8.0 * snr_margin))
        if c.dropout_bias > 0:
            prob *= np.clip(1.0 - c.dropout_bias, 0.0, 1.0)
        prob = np.where(beam_margin >= -80.0, prob, prob * 0.35)
        return np.clip(prob, 0.0, 1.0)

    def apply_shield(self, usv_action: np.ndarray, auv_actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        c = self.cfg
        interventions = 0
        safe_usv = _clip_norm(usv_action, 6.0)
        if np.linalg.norm(safe_usv - usv_action) > 1e-6:
            interventions += 1
        for p, v in zip(self.vessel_pos, self.vessel_vel):
            rel = p - self.usv_pos
            dist = np.linalg.norm(rel)
            if dist < 130.0:
                avoid = -rel / max(dist, 1e-6) * (130.0 - dist) / 25.0
                safe_usv = _clip_norm(safe_usv + avoid, 6.0)
                interventions += 1
        for ox, oy, rad in self.obstacles:
            rel = np.array([ox, oy]) - self.usv_pos
            dist = np.linalg.norm(rel)
            if dist < rad + 45.0:
                safe_usv = _clip_norm(safe_usv - rel / max(dist, 1e-6) * 4.0, 6.0)
                interventions += 1

        safe_auv = auv_actions.copy()
        for i in range(c.num_auvs):
            original = safe_auv[i].copy()
            safe_auv[i] = _clip_norm(safe_auv[i], 2.0)
            if self.packet_probs()[i] < c.packet_threshold + 0.05:
                to_usv = self.usv_pos - self.auv_pos[i]
                safe_auv[i] += _clip_norm(to_usv, 1.0)
            for ox, oy, rad in self.obstacles:
                rel = np.array([ox, oy]) - self.auv_pos[i]
                dist = np.linalg.norm(rel)
                if dist < rad + 35.0:
                    safe_auv[i] -= rel / max(dist, 1e-6) * 2.0
            safe_auv[i] = _clip_norm(safe_auv[i], 2.0)
            if np.linalg.norm(safe_auv[i] - original) > 1e-6:
                interventions += 1
        return safe_usv, safe_auv, interventions

    def step(self, usv_action: np.ndarray, auv_actions: np.ndarray, shield: bool = True) -> Tuple[Dict[str, np.ndarray], StepMetrics, bool]:
        c = self.cfg
        if shield:
            usv_action, auv_actions, interventions = self.apply_shield(usv_action, auv_actions)
        else:
            interventions = 0
            usv_action = _clip_norm(usv_action, 6.0)
            auv_actions = np.array([_clip_norm(a, 2.0) for a in auv_actions])

        prev_task_dist = np.linalg.norm(self.auv_pos - self.tasks, axis=1)
        self.usv_pos = np.clip(self.usv_pos + usv_action * c.dt, 0.0, c.world_size)
        self.auv_pos = np.clip(self.auv_pos + auv_actions * c.dt, 0.0, c.world_size)
        self.vessel_pos = (self.vessel_pos + self.vessel_vel * c.dt) % c.world_size
        if c.emergency and self.t > c.steps // 2:
            self.auv_depth[0] = min(280.0, self.auv_depth[0] + 0.2 * c.dt)
        new_task_dist = np.linalg.norm(self.auv_pos - self.tasks, axis=1)
        newly_done = (new_task_dist < 45.0) & (~self.task_done)
        self.task_done |= newly_done
        task_progress = float(np.maximum(prev_task_dist - new_task_dist, 0.0).sum() + newly_done.sum() * 100.0)

        probs = self.packet_probs()
        outage_count = int(np.sum(probs < c.packet_threshold))
        usv_collision = bool(np.any(np.linalg.norm(self.vessel_pos - self.usv_pos[None, :], axis=1) < 35.0))
        auv_collision_count = 0
        for i in range(c.num_auvs):
            for ox, oy, rad in self.obstacles:
                if np.linalg.norm(self.auv_pos[i] - np.array([ox, oy])) < rad + 12.0:
                    auv_collision_count += 1
        colregs_violation = False
        for p, v in zip(self.vessel_pos, self.vessel_vel):
            rel = p - self.usv_pos
            if np.linalg.norm(rel) < 160.0 and np.cross(np.append(usv_action, 0.0), np.append(rel, 0.0))[2] > 0:
                colregs_violation = True
        energy = float(np.linalg.norm(usv_action) ** 2 + 0.5 * np.sum(np.linalg.norm(auv_actions, axis=1) ** 2))
        smoothness = float(np.linalg.norm(usv_action - self.usv_prev_action) + np.sum(np.linalg.norm(auv_actions - self.auv_prev_action, axis=1)))
        self.energy += energy
        self.usv_prev_action = usv_action.copy()
        self.auv_prev_action = auv_actions.copy()
        self.shield_interventions += interventions
        self.t += 1
        done = self.t >= c.steps or bool(np.all(self.task_done))
        metrics = StepMetrics(
            task_progress=task_progress,
            outage_count=outage_count,
            packet_probs=probs,
            usv_collision=usv_collision,
            auv_collision_count=auv_collision_count,
            colregs_violation=colregs_violation,
            energy=energy,
            smoothness=smoothness,
            shield_interventions=interventions,
        )
        return self.observe(), metrics, done


def summarize_episode(cfg: ScenarioConfig, step_metrics: List[StepMetrics], task_done: np.ndarray, elapsed_steps: int) -> Dict[str, float]:
    if not step_metrics:
        raise ValueError("empty episode")
    packets = np.stack([m.packet_probs for m in step_metrics], axis=0)
    outage_loss = np.array([m.outage_count / cfg.num_auvs for m in step_metrics])
    connected = packets >= cfg.packet_threshold
    per_agent_conn = connected.mean(axis=0)
    fairness = float((per_agent_conn.sum() ** 2) / (cfg.num_auvs * np.sum(per_agent_conn**2) + 1e-9))
    return {
        "success": float(np.all(task_done) and not any(m.usv_collision for m in step_metrics) and sum(m.auv_collision_count for m in step_metrics) == 0),
        "task_completion": float(task_done.mean()),
        "mission_time": float(elapsed_steps),
        "outage_rate": float(outage_loss.mean()),
        "mean_packet": float(packets.mean()),
        "worst_agent_packet": float(packets.mean(axis=0).min()),
        "cvar_outage_90": float(np.sort(outage_loss)[int(0.9 * len(outage_loss)) :].mean()),
        "comm_fairness": fairness,
        "usv_collision_rate": float(np.mean([m.usv_collision for m in step_metrics])),
        "auv_collision_count": float(sum(m.auv_collision_count for m in step_metrics)),
        "colregs_violation_rate": float(np.mean([m.colregs_violation for m in step_metrics])),
        "energy": float(sum(m.energy for m in step_metrics)),
        "smoothness": float(sum(m.smoothness for m in step_metrics)),
        "shield_interventions": float(sum(m.shield_interventions for m in step_metrics)),
    }
