from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from haco.experiment import aggregate, run_episode, scenario_config, write_csv, write_json
from haco.policies import HACoSafeHeuristicPolicy, clip_params


SCENARIOS = ["survey", "traffic", "acoustic_degradation", "emergency", "generalization"]


def objective(row: dict) -> float:
    return (
        5.0 * row["task_completion"]
        + 2.0 * row["success"]
        + 0.8 * row["worst_agent_packet"]
        - 0.8 * row["outage_rate"]
        - 0.08 * row["auv_collision_count"]
        - 0.5 * row["colregs_violation_rate"]
        - 0.0002 * row["shield_interventions"]
    )


def scenario_auv_counts(scenario: str) -> list[int]:
    return [2, 6, 8] if scenario == "generalization" else [3]


def evaluate_params(params: dict, episodes: int, seed_base: int) -> tuple[float, dict]:
    policy = HACoSafeHeuristicPolicy(params=params, name="haco_safemarl_task_priority")
    rows = []
    scores = []
    for scenario_idx, scenario in enumerate(SCENARIOS):
        for num_auvs in scenario_auv_counts(scenario):
            for ep in range(episodes):
                seed = seed_base + scenario_idx * 100000 + num_auvs * 1000 + ep * 97
                row = run_episode(policy, scenario_config(scenario, seed=seed, num_auvs=num_auvs))
                rows.append(row)
                scores.append(objective(row))
    return sum(scores) / len(scores), {"score": sum(scores) / len(scores), "params": params, "rows": rows}


def candidate_params(rng: random.Random) -> dict:
    return clip_params(
        {
            "relay_link_weight": rng.uniform(0.0, 4.0),
            "relay_task_mix": rng.uniform(0.35, 0.65),
            "usv_speed": rng.uniform(4.5, 6.0),
            "auv_task_speed": rng.uniform(1.6, 2.0),
            "auv_return_speed": rng.uniform(0.3, 0.8),
            "auv_link_speed": rng.uniform(0.8, 1.5),
            "link_pressure_gain": rng.uniform(0.0, 0.35),
            "packet_target": rng.uniform(0.70, 0.82),
        }
    )


def grid_candidates(limit: int | None = None) -> list[dict]:
    out = []
    for relay_link_weight in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
        for relay_task_mix in [0.35, 0.45, 0.55, 0.65]:
            for link_pressure_gain in [0.0, 0.08, 0.15, 0.25, 0.35]:
                for packet_target in [0.70, 0.74, 0.78, 0.80]:
                    out.append(
                        clip_params(
                            {
                                "relay_link_weight": relay_link_weight,
                                "relay_task_mix": relay_task_mix,
                                "usv_speed": 5.5,
                                "auv_task_speed": 1.95,
                                "auv_return_speed": 0.3,
                                "auv_link_speed": 1.1,
                                "link_pressure_gain": link_pressure_gain,
                                "packet_target": packet_target,
                            }
                        )
                    )
                    if limit is not None and len(out) >= limit:
                        return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-episodes", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--random-candidates", type=int, default=160)
    parser.add_argument("--grid-candidates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--out", type=Path, default=Path("results/task_priority_policy"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    candidates = grid_candidates(args.grid_candidates) + [candidate_params(rng) for _ in range(args.random_candidates)]
    history = []
    best = {"score": -1e9, "params": None, "rows": []}
    for idx, params in enumerate(candidates):
        score, payload = evaluate_params(params, args.search_episodes, args.seed + idx * 1000000)
        rec = {"candidate": idx, "score": score, "params": params}
        history.append(rec)
        if score > best["score"]:
            best = payload
        if (idx + 1) % 25 == 0 or idx == len(candidates) - 1:
            write_json(args.out / "search_history.json", history)
            write_json(args.out / "best_policy.json", best)
            print(f"searched {idx + 1}/{len(candidates)} best={best['score']:.4f}", flush=True)

    final_score, final_payload = evaluate_params(best["params"], args.eval_episodes, args.seed + 990000000)
    final_rows = final_payload["rows"]
    agg = aggregate(final_rows)
    write_csv(args.out / "episode_metrics.csv", final_rows)
    write_csv(args.out / "aggregate_metrics.csv", agg)
    write_json(args.out / "aggregate_metrics.json", agg)
    write_json(args.out / "best_policy.json", {"score": final_score, "params": best["params"], "rows": best["rows"]})
    print(json.dumps({"final_score": final_score, "params": best["params"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
