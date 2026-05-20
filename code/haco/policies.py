from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np


def _unit_to(target: np.ndarray, source: np.ndarray, speed: float) -> np.ndarray:
    vec = target - source
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros_like(vec)
    return vec / norm * speed


class BasePolicy:
    name = "base"

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        raise NotImplementedError


class FixedAUVRelayPolicy(BasePolicy):
    name = "fixed_auv_ga_pso_tlbo_proxy"

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        auv_pos = obs["auv_pos"]
        tasks = obs["tasks"]
        usv_pos = obs["usv_pos"]
        centroid = auv_pos.mean(axis=0)
        usv_action = _unit_to(centroid, usv_pos, 4.0)
        auv_actions = np.vstack([_unit_to(tasks[i], auv_pos[i], 1.3) for i in range(len(auv_pos))])
        return usv_action, auv_actions, False


class IndependentGreedyPolicy(BasePolicy):
    name = "independent_greedy"

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        auv_pos = obs["auv_pos"]
        tasks = obs["tasks"]
        usv_pos = obs["usv_pos"]
        usv_action = _unit_to(tasks.mean(axis=0), usv_pos, 4.5)
        auv_actions = np.vstack([_unit_to(tasks[i], auv_pos[i], 1.7) for i in range(len(auv_pos))])
        return usv_action, auv_actions, False


class CommunicationAwarePolicy(BasePolicy):
    name = "communication_aware"

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        auv_pos = obs["auv_pos"]
        tasks = obs["tasks"]
        usv_pos = obs["usv_pos"]
        packets = obs["packet_probs"]
        weights = 1.0 + 3.0 * np.maximum(0.0, 0.8 - packets)
        relay = (auv_pos * weights[:, None]).sum(axis=0) / weights.sum()
        usv_action = _unit_to(relay, usv_pos, 5.0)
        auv_actions = []
        for i in range(len(auv_pos)):
            task_vec = _unit_to(tasks[i], auv_pos[i], 1.5)
            if packets[i] < 0.75:
                link_vec = _unit_to(usv_pos, auv_pos[i], 1.0)
                action = 0.55 * task_vec + 0.45 * link_vec
            else:
                action = task_vec
            auv_actions.append(action)
        return usv_action, np.vstack(auv_actions), False


DEFAULT_HACO_PARAMS = {
    "relay_link_weight": 5.0,
    "relay_task_mix": 0.28,
    "usv_speed": 5.3,
    "auv_task_speed": 1.75,
    "auv_return_speed": 0.70,
    "auv_link_speed": 1.70,
    "link_pressure_gain": 0.55,
    "packet_target": 0.84,
}


PARAM_BOUNDS = {
    "relay_link_weight": (0.0, 10.0),
    "relay_task_mix": (0.0, 0.65),
    "usv_speed": (3.0, 6.0),
    "auv_task_speed": (0.9, 2.0),
    "auv_return_speed": (0.3, 1.4),
    "auv_link_speed": (0.8, 2.0),
    "link_pressure_gain": (0.0, 0.9),
    "packet_target": (0.70, 0.92),
}


def clip_params(params: Mapping[str, float]) -> Dict[str, float]:
    out = dict(DEFAULT_HACO_PARAMS)
    out.update({k: float(v) for k, v in params.items() if k in DEFAULT_HACO_PARAMS})
    for key, (lo, hi) in PARAM_BOUNDS.items():
        out[key] = float(np.clip(out[key], lo, hi))
    return out


class HACoSafeHeuristicPolicy(BasePolicy):
    name = "haco_safemarl_pilot"

    def __init__(
        self,
        no_acoustic: bool = False,
        no_shield: bool = False,
        params: Mapping[str, float] | None = None,
        name: str | None = None,
    ):
        self.no_acoustic = no_acoustic
        self.no_shield = no_shield
        self.params = clip_params(params or {})
        suffix = []
        if no_acoustic:
            suffix.append("no_acoustic")
        if no_shield:
            suffix.append("no_shield")
        if suffix:
            self.name = "haco_safemarl_pilot_" + "_".join(suffix)
        if name is not None:
            self.name = name

    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        auv_pos = obs["auv_pos"]
        tasks = obs["tasks"]
        done = obs["task_done"]
        usv_pos = obs["usv_pos"]
        packets = obs["packet_probs"]
        active = ~done
        if not np.any(active):
            active = np.ones(len(auv_pos), dtype=bool)

        p = self.params
        if self.no_acoustic:
            weights = np.where(active, 1.0, 0.2)
        else:
            weights = np.where(
                active,
                1.0 + p["relay_link_weight"] * np.maximum(0.0, p["packet_target"] - packets),
                0.2,
            )
        relay = (auv_pos * weights[:, None]).sum(axis=0) / weights.sum()
        task_center = tasks[active].mean(axis=0)
        mix = p["relay_task_mix"]
        usv_target = (1.0 - mix) * relay + mix * task_center
        usv_action = _unit_to(usv_target, usv_pos, p["usv_speed"])

        auv_actions = []
        for i in range(len(auv_pos)):
            if done[i]:
                target = usv_pos
                task_vec = _unit_to(target, auv_pos[i], p["auv_return_speed"])
            else:
                task_vec = _unit_to(tasks[i], auv_pos[i], p["auv_task_speed"])
            if self.no_acoustic:
                action = task_vec
            else:
                link_pressure = max(0.0, p["packet_target"] - float(packets[i])) / max(p["packet_target"], 1e-6)
                link_vec = _unit_to(usv_pos, auv_pos[i], p["auv_link_speed"])
                blend = p["link_pressure_gain"] * link_pressure
                action = (1.0 - blend) * task_vec + blend * link_vec
            auv_actions.append(action)
        return usv_action, np.vstack(auv_actions), not self.no_shield


def policy_suite():
    return [
        FixedAUVRelayPolicy(),
        IndependentGreedyPolicy(),
        CommunicationAwarePolicy(),
        HACoSafeHeuristicPolicy(no_acoustic=True),
        HACoSafeHeuristicPolicy(no_shield=True),
        HACoSafeHeuristicPolicy(),
    ]
