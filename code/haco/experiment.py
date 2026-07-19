from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np

from haco.env import HACoPilotEnv, ScenarioConfig, summarize_episode
from haco.policies import policy_suite


def scenario_config(name: str, seed: int, num_auvs: int = 3) -> ScenarioConfig:
    base = dict(seed=seed, num_auvs=num_auvs, scenario_name=name)
    if name == "survey":
        return ScenarioConfig(**base, noise_level=0.12, dropout_bias=0.02, traffic=False)
    if name == "traffic":
        return ScenarioConfig(**base, noise_level=0.15, dropout_bias=0.05, traffic=True, num_surface_vessels=5)
    if name == "acoustic_degradation":
        return ScenarioConfig(**base, acoustic_range=430.0, noise_level=0.28, dropout_bias=0.18, traffic=True)
    if name == "emergency":
        return ScenarioConfig(**base, acoustic_range=460.0, noise_level=0.22, dropout_bias=0.12, traffic=True, emergency=True)
    if name == "generalization":
        return ScenarioConfig(**base, acoustic_range=500.0, noise_level=0.18, dropout_bias=0.08, traffic=True, num_obstacles=12)
    if name == "colregs_stress":
        return ScenarioConfig(
            **base,
            noise_level=0.15,
            dropout_bias=0.05,
            traffic=True,
            num_surface_vessels=8,
            encounter_stress=True,
        )
    if name == "dynamics_current":
        return ScenarioConfig(
            **base,
            noise_level=0.18,
            dropout_bias=0.08,
            traffic=True,
            usv_time_constant=8.0,
            auv_time_constant=4.0,
            max_usv_turn_rate=np.deg2rad(6.0),
            max_auv_turn_rate=np.deg2rad(12.0),
            current_speed=0.5,
        )
    raise ValueError(f"unknown scenario {name}")


def run_episode(policy, cfg: ScenarioConfig):
    env = HACoPilotEnv(cfg)
    obs = env.reset()
    metrics = []
    done = False
    while not done:
        usv_action, auv_actions, shield = policy.act(obs)
        obs, step_metrics, done = env.step(usv_action, auv_actions, shield=shield)
        metrics.append(step_metrics)
    summary = summarize_episode(cfg, metrics, env.task_done, env.t)
    summary.update({"policy": policy.name, "scenario": cfg.scenario_name, "seed": cfg.seed, "num_auvs": cfg.num_auvs})
    return summary


def aggregate(rows):
    keys = [
        "success",
        "task_completion",
        "outage_rate",
        "mean_packet",
        "worst_agent_packet",
        "cvar_outage_90",
        "comm_fairness",
        "usv_collision_rate",
        "auv_collision_count",
        "colregs_violation_rate",
        "energy",
        "smoothness",
        "shield_interventions",
        "mission_time",
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["policy"]), []).append(row)
    out = []
    for (scenario, policy), items in sorted(grouped.items()):
        rec = {"scenario": scenario, "policy": policy, "episodes": len(items)}
        for key in keys:
            vals = [float(x[key]) for x in items]
            rec[f"{key}_mean"] = mean(vals)
            rec[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_latex_table(path: Path, agg_rows):
    selected = [r for r in agg_rows if r["scenario"] in {"survey", "traffic", "acoustic_degradation"}]
    policies = [
        "fixed_auv_ga_pso_tlbo_proxy",
        "independent_greedy",
        "communication_aware",
        "haco_safemarl_pilot_no_acoustic",
        "haco_safemarl_pilot_no_shield",
        "haco_safemarl_pilot",
        "haco_safemarl_trained",
        "haco_safemarl_mappo",
    ]
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Scenario & Method & Success $\\uparrow$ & Task $\\uparrow$ & Outage $\\downarrow$ & Worst pkt $\\uparrow$ & Crossing $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for scenario in ["survey", "traffic", "acoustic_degradation"]:
        for pol in policies:
            match = [r for r in selected if r["scenario"] == scenario and r["policy"] == pol]
            if not match:
                continue
            r = match[0]
            lines.append(
                f"{scenario} & {pol.replace('_', '-')} & "
                f"{r['success_mean']:.2f} & {r['task_completion_mean']:.2f} & "
                f"{r['outage_rate_mean']:.2f} & {r['worst_agent_packet_mean']:.2f} & "
                f"{r['colregs_violation_rate_mean']:.2f} \\\\"
            )
        lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def default_policies():
    return policy_suite()
