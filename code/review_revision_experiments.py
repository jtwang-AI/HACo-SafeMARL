from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np

from haco.env import HACoPilotEnv, summarize_episode
from haco.experiment import aggregate, run_episode, scenario_config, write_csv, write_json
from haco.policies import HACoSafeHeuristicPolicy, IndependentGreedyPolicy


MAIN_SCENARIOS = ["survey", "traffic", "acoustic_degradation", "emergency", "generalization"]
KEY_METRICS = ["task_completion", "outage_rate", "worst_agent_packet", "auv_collision_count"]


def load_params(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return payload.get("params", payload.get("best_params", {}))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def paired_inference(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, float | list[float]]:
    delta = a - b
    n = len(delta)
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    boot = delta[indices].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutation_samples, n))
    permuted = (signs * delta[None, :]).mean(axis=1)
    observed = float(delta.mean())
    p_value = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (permutation_samples + 1))
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean_delta": observed,
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "paired_effect_dz": observed / sd if sd > 1e-12 else 0.0,
        "p_value": p_value,
    }


def analyze_main_results(
    episode_csv: Path,
    out_dir: Path,
    table_dir: Path,
    bootstrap_samples: int,
    permutation_samples: int,
) -> None:
    rows = read_rows(episode_csv)
    proposed = "haco_safemarl_trained"
    comparisons = [
        "independent_greedy",
        "communication_aware",
        "haco_safemarl_pilot_no_acoustic",
        "haco_safemarl_pilot_no_shield",
    ]
    rng = np.random.default_rng(20260719)
    records: list[dict] = []
    for baseline in comparisons:
        for metric in KEY_METRICS:
            pmap = {
                (r["scenario"], r["seed"], r["num_auvs"]): float(r[metric])
                for r in rows
                if r["policy"] == proposed
            }
            bmap = {
                (r["scenario"], r["seed"], r["num_auvs"]): float(r[metric])
                for r in rows
                if r["policy"] == baseline
            }
            keys = sorted(pmap.keys() & bmap.keys())
            result = paired_inference(
                np.array([pmap[k] for k in keys]),
                np.array([bmap[k] for k in keys]),
                rng,
                bootstrap_samples,
                permutation_samples,
            )
            result.update({"comparison": baseline, "metric": metric})
            records.append(result)
    adjusted = holm_adjust([float(r["p_value"]) for r in records])
    for rec, value in zip(records, adjusted):
        rec["p_holm"] = value
    write_json(out_dir / "paired_statistics.json", records)

    label = {
        "independent_greedy": "Independent greedy",
        "communication_aware": "Communication-aware",
        "haco_safemarl_pilot_no_acoustic": "HACo-Safe w/o acoustic",
        "haco_safemarl_pilot_no_shield": "HACo-Safe w/o shield",
    }
    metric_label = {
        "task_completion": "Task completion",
        "outage_rate": "Outage rate",
        "worst_agent_packet": "Worst-agent packet",
        "auv_collision_count": "AUV collisions",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Paired comparison (HACo-Safe minus baseline) & Metric & $\Delta$ & 95\% CI & Holm $p$ \\",
        r"\midrule",
    ]
    for rec in records:
        lo, hi = rec["ci95"]
        lines.append(
            f"{label[rec['comparison']]} & {metric_label[rec['metric']]} & "
            f"{rec['mean_delta']:.3f} & [{lo:.3f}, {hi:.3f}] & {rec['p_holm']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "review_paired_statistics.tex").write_text("\n".join(lines))

    shield_rows = []
    for scenario in MAIN_SCENARIOS:
        proposed_rows = [r for r in rows if r["scenario"] == scenario and r["policy"] == proposed]
        raw_rows = [r for r in rows if r["scenario"] == scenario and r["policy"] == "haco_safemarl_pilot_no_shield"]
        if not proposed_rows or not raw_rows:
            continue
        steps = np.array([float(r["mission_time"]) for r in proposed_rows])
        agents = np.array([float(r["num_auvs"]) + 1.0 for r in proposed_rows])
        interventions = np.array([float(r["shield_interventions"]) for r in proposed_rows])
        shield_rows.append(
            {
                "scenario": scenario,
                "episodes": len(proposed_rows),
                "interventions_per_episode": float(interventions.mean()),
                "interventions_per_100_agent_steps": float(np.mean(100.0 * interventions / (steps * agents))),
                "shielded_auv_collisions": mean(float(r["auv_collision_count"]) for r in proposed_rows),
                "raw_auv_collisions": mean(float(r["auv_collision_count"]) for r in raw_rows),
            }
        )
    write_json(out_dir / "shield_scenario_audit.json", shield_rows)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Interventions/episode & Interventions/100 agent-steps & Shielded AUV coll. & Raw AUV coll. \\",
        r"\midrule",
    ]
    for rec in shield_rows:
        lines.append(
            f"{rec['scenario'].replace('_', ' ').title()} & {rec['interventions_per_episode']:.0f} & "
            f"{rec['interventions_per_100_agent_steps']:.1f} & {rec['shielded_auv_collisions']:.1f} & "
            f"{rec['raw_auv_collisions']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "review_shield_scenario.tex").write_text("\n".join(lines))

    mission_rows = []
    for scenario in MAIN_SCENARIOS:
        items = [r for r in rows if r["scenario"] == scenario and r["policy"] == proposed]
        if not items:
            continue
        task_complete = [float(r["task_completion"]) == 1.0 for r in items]
        collision_free = [
            float(r["usv_collision_rate"]) == 0.0 and float(r["auv_collision_count"]) == 0.0
            for r in items
        ]
        mission_rows.append(
            {
                "scenario": scenario,
                "all_tasks": float(np.mean(task_complete)),
                "collision_free": float(np.mean(collision_free)),
                "strict_success": mean(float(r["success"]) for r in items),
                "outage": mean(float(r["outage_rate"]) for r in items),
                "fairness": mean(float(r["comm_fairness"]) for r in items),
                "cvar": mean(float(r["cvar_outage_90"]) for r in items),
            }
        )
    write_json(out_dir / "mission_success_decomposition.json", mission_rows)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Scenario & All tasks & Collision-free & Strict success & Outage & Fairness & CVaR$_{0.9}$ \\",
        r"\midrule",
    ]
    for rec in mission_rows:
        lines.append(
            f"{rec['scenario'].replace('_', ' ').title()} & {rec['all_tasks']:.2f} & "
            f"{rec['collision_free']:.2f} & {rec['strict_success']:.2f} & {rec['outage']:.2f} & "
            f"{rec['fairness']:.2f} & {rec['cvar']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "review_mission_decomposition.tex").write_text("\n".join(lines))


def copy_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value) for key, value in obs.items()}


def tdma_cycle_seconds(obs: dict[str, np.ndarray], slot_airtime: float = 1.0) -> float:
    horizontal = np.linalg.norm(obs["auv_pos"] - obs["usv_pos"][None, :], axis=1)
    slant = np.sqrt(horizontal**2 + obs["auv_depth"] ** 2)
    # Stop-and-wait TDMA sensitivity: one payload plus a round-trip propagation
    # allowance per AUV. This is deliberately conservative and not a modem claim.
    return float(len(slant) * (slot_airtime + 2.0 * slant.mean() / 1500.0))


def run_delayed_episode(policy, cfg, mode: str) -> dict[str, float]:
    env = HACoPilotEnv(cfg)
    obs = env.reset()
    history: deque[dict[str, np.ndarray]] = deque(maxlen=64)
    metrics = []
    ages = []
    service_intervals = []
    done = False
    while not done:
        history.append(copy_observation(obs))
        if mode == "ideal":
            age_seconds = 0.0
            service_interval = cfg.dt
        elif mode == "propagation":
            horizontal = np.linalg.norm(obs["auv_pos"] - obs["usv_pos"][None, :], axis=1)
            slant = np.sqrt(horizontal**2 + obs["auv_depth"] ** 2)
            age_seconds = float(slant.max() / 1500.0)
            service_interval = cfg.dt
        elif mode == "tdma":
            service_interval = tdma_cycle_seconds(obs)
            age_seconds = service_interval
        else:
            raise ValueError(mode)
        age_steps = int(math.ceil(age_seconds / cfg.dt)) if age_seconds > 0 else 0
        stale = history[max(0, len(history) - 1 - age_steps)]
        policy_obs = copy_observation(obs)
        for key in ["auv_pos", "auv_depth", "task_done", "packet_probs"]:
            policy_obs[key] = stale[key].copy()
        usv_action, auv_action, shield = policy.act(policy_obs)
        obs, step_metric, done = env.step(usv_action, auv_action, shield=shield)
        metrics.append(step_metric)
        ages.append(age_seconds)
        service_intervals.append(service_interval)
    summary = summarize_episode(cfg, metrics, env.task_done, env.t)
    summary.update(
        {
            "policy": policy.name,
            "scenario": cfg.scenario_name,
            "seed": cfg.seed,
            "num_auvs": cfg.num_auvs,
            "communication_mode": mode,
            "message_age_seconds": mean(ages),
            "service_interval_seconds": mean(service_intervals),
            "per_auv_update_rate_hz": 1.0 / mean(service_intervals),
        }
    )
    return summary


def run_sensitivity_experiments(params: dict[str, float], episodes: int, out_dir: Path, table_dir: Path) -> None:
    policy = HACoSafeHeuristicPolicy(params=params, name="haco_safe_shielded")
    delayed_rows = []
    for scenario in ["survey", "traffic", "acoustic_degradation", "emergency"]:
        for ep in range(episodes):
            seed = 910000 + 10000 * ep + 101 * MAIN_SCENARIOS.index(scenario)
            cfg = scenario_config(scenario, seed=seed, num_auvs=3)
            for mode in ["ideal", "propagation", "tdma"]:
                delayed_rows.append(run_delayed_episode(policy, cfg, mode))
    write_csv(out_dir / "communication_delay_episode_metrics.csv", delayed_rows)
    delayed_agg = []
    for (scenario, mode) in sorted({(r["scenario"], r["communication_mode"]) for r in delayed_rows}):
        items = [r for r in delayed_rows if r["scenario"] == scenario and r["communication_mode"] == mode]
        delayed_agg.append(
            {
                "scenario": scenario,
                "mode": mode,
                "episodes": len(items),
                "task_completion": mean(r["task_completion"] for r in items),
                "outage_rate": mean(r["outage_rate"] for r in items),
                "worst_agent_packet": mean(r["worst_agent_packet"] for r in items),
                "service_interval_seconds": mean(r["service_interval_seconds"] for r in items),
                "per_auv_update_rate_hz": mean(r["per_auv_update_rate_hz"] for r in items),
            }
        )
    write_json(out_dir / "communication_delay_summary.json", delayed_agg)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Scenario & Information model & Service interval (s) & Update rate (Hz) & Task & Outage & Worst pkt \\",
        r"\midrule",
    ]
    for rec in delayed_agg:
        lines.append(
            f"{rec['scenario'].replace('_', ' ').title()} & {rec['mode'].title()} & "
            f"{rec['service_interval_seconds']:.2f} & {rec['per_auv_update_rate_hz']:.2f} & "
            f"{rec['task_completion']:.2f} & {rec['outage_rate']:.2f} & {rec['worst_agent_packet']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "review_delay_sensitivity.tex").write_text("\n".join(lines))

    robustness_rows = []
    raw_policy = HACoSafeHeuristicPolicy(params=params, no_shield=True, name="haco_safe_raw")
    policies = [policy, raw_policy, IndependentGreedyPolicy()]
    for condition in ["traffic", "dynamics_current", "colregs_stress"]:
        for ep in range(episodes):
            seed = 930000 + 10000 * ep
            cfg = scenario_config(condition, seed=seed, num_auvs=3)
            for candidate in policies:
                robustness_rows.append(run_episode(candidate, cfg))
    write_csv(out_dir / "robustness_episode_metrics.csv", robustness_rows)
    robustness_agg = aggregate(robustness_rows)
    write_json(out_dir / "robustness_summary.json", robustness_agg)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Condition & Controller & Success & Task & AUV coll. & Crossing & Shield \\",
        r"\midrule",
    ]
    policy_label = {
        "haco_safe_shielded": "HACo-Safe shielded",
        "haco_safe_raw": "HACo-Safe raw",
        "independent_greedy": "Independent greedy",
    }
    condition_label = {
        "colregs_stress": "Crossing stress",
        "dynamics_current": "Dynamics + current",
        "traffic": "Traffic",
    }
    for rec in robustness_agg:
        lines.append(
            f"{condition_label[rec['scenario']]} & {policy_label[rec['policy']]} & "
            f"{rec['success_mean']:.2f} & {rec['task_completion_mean']:.2f} & "
            f"{rec['auv_collision_count_mean']:.1f} & {rec['colregs_violation_rate_mean']:.3f} & "
            f"{rec['shield_interventions_mean']:.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (table_dir / "review_robustness_sensitivity.tex").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=Path("results/cem_local/best_policy.json"))
    parser.add_argument(
        "--main-episodes",
        type=Path,
        default=Path("results/final_eval_local40/episode_metrics.csv"),
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("results/reviewer_revision"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    analyze_main_results(
        args.main_episodes,
        args.out,
        args.table_dir,
        args.bootstrap_samples,
        args.permutation_samples,
    )
    run_sensitivity_experiments(load_params(args.policy), args.episodes, args.out, args.table_dir)
    manifest = {
        "policy": str(args.policy),
        "main_episode_source": str(args.main_episodes),
        "episodes_per_sensitivity_cell": args.episodes,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed_policy": "fixed deterministic seeds recorded in episode CSV files",
    }
    write_json(args.out / "manifest.json", manifest)
    print(f"wrote reviewer-revision artifacts to {args.out}")


if __name__ == "__main__":
    main()
