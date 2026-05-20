from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from haco.experiment import scenario_config, write_json
from haco.policies import DEFAULT_HACO_PARAMS, PARAM_BOUNDS, HACoSafeHeuristicPolicy, clip_params
from run_pilot_experiments import run_episode


PARAM_NAMES = list(DEFAULT_HACO_PARAMS.keys())


def vector_to_params(vec: np.ndarray) -> dict:
    return clip_params({k: float(v) for k, v in zip(PARAM_NAMES, vec)})


def objective(summary: dict) -> float:
    return (
        4.0 * summary["success"]
        + 2.5 * summary["task_completion"]
        + 1.5 * summary["worst_agent_packet"]
        + 0.7 * summary["comm_fairness"]
        - 2.4 * summary["outage_rate"]
        - 2.0 * summary["cvar_outage_90"]
        - 3.0 * summary["usv_collision_rate"]
        - 0.5 * summary["auv_collision_count"]
        - 0.8 * summary["colregs_violation_rate"]
        - 0.00018 * summary["energy"]
        - 0.00035 * summary["smoothness"]
        - 0.0010 * summary["shield_interventions"]
    )


def evaluate_candidate(params: dict, scenarios: list[str], episodes: int, seed_offset: int) -> tuple[float, dict]:
    policy = HACoSafeHeuristicPolicy(params=params, name="haco_safemarl_trained")
    rows = []
    scores = []
    for scenario in scenarios:
        for ep in range(episodes):
            cfg = scenario_config(scenario, seed=seed_offset + 1009 * ep + 37 * len(rows))
            row = run_episode(policy, cfg)
            rows.append(row)
            scores.append(objective(row))
    return float(np.mean(scores)), {"score": float(np.mean(scores)), "params": params, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--out", type=Path, default=Path("results/cem"))
    parser.add_argument("--scenarios", nargs="*", default=["survey", "traffic", "acoustic_degradation", "emergency"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lows = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES], dtype=float)
    highs = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES], dtype=float)
    mean = np.array([DEFAULT_HACO_PARAMS[k] for k in PARAM_NAMES], dtype=float)
    std = (highs - lows) * 0.25
    elite_n = max(2, int(math.ceil(args.population * args.elite_frac)))
    history = []
    best = {"score": -1e9, "params": vector_to_params(mean), "rows": []}

    for gen in range(args.generations):
        samples = rng.normal(mean, std, size=(args.population, len(PARAM_NAMES)))
        samples = np.clip(samples, lows, highs)
        scored = []
        for idx, vec in enumerate(samples):
            params = vector_to_params(vec)
            score, payload = evaluate_candidate(
                params,
                args.scenarios,
                args.episodes,
                seed_offset=args.seed + gen * 100000 + idx * 1000,
            )
            scored.append((score, vec, payload))
            if score > best["score"]:
                best = payload
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = np.vstack([x[1] for x in scored[:elite_n]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), (highs - lows) * 0.035)
        rec = {
            "generation": gen,
            "best_score": float(scored[0][0]),
            "mean_score": float(np.mean([x[0] for x in scored])),
            "best_params": vector_to_params(scored[0][1]),
        }
        history.append(rec)
        write_json(args.out / "cem_history.json", history)
        write_json(args.out / "best_policy.json", best)
        print(f"generation={gen} best={rec['best_score']:.4f} mean={rec['mean_score']:.4f}")

    print(f"best score: {best['score']:.4f}")
    print(f"best policy: {args.out / 'best_policy.json'}")


if __name__ == "__main__":
    main()
