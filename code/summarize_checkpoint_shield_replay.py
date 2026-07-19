from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    ("success_mean", "Success", 2),
    ("task_completion_mean", "Task", 2),
    ("outage_rate_mean", "Outage", 2),
    ("worst_agent_packet_mean", "Worst pkt", 2),
    ("auv_collision_count_mean", "AUV pen.", 1),
    ("colregs_violation_rate_mean", "Crossing", 3),
    ("shield_interventions_mean", "Shield", 0),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(values: list[float], digits: int) -> str:
    return f"${mean(values):.{digits}f}\\pm{stdev(values):.{digits}f}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/review_same_actor_shield_replay"))
    parser.add_argument(
        "--out-json", type=Path, default=Path("results/review_same_actor_shield_replay_summary.json")
    )
    parser.add_argument(
        "--out-table", type=Path, default=Path("tables/review_same_actor_shield_replay.tex")
    )
    args = parser.parse_args()

    policies = ["mappo_same_actor_shielded", "mappo_same_actor_raw"]
    by_policy: dict[str, list[dict[str, float | str]]] = {policy: [] for policy in policies}
    for seed_dir in sorted(args.root.glob("seed_*")):
        rows = read_csv(seed_dir / "aggregate_metrics.csv")
        for policy in policies:
            subset = [row for row in rows if row["policy"] == policy]
            if not subset:
                continue
            record: dict[str, float | str] = {"seed": seed_dir.name.replace("seed_", "")}
            for key, _, _ in METRICS:
                record[key] = mean(float(row[key]) for row in subset)
            by_policy[policy].append(record)

    if any(len(records) < 2 for records in by_policy.values()):
        raise RuntimeError("at least two completed seeds are required for replay summary")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(by_policy, indent=2), encoding="utf-8")
    labels = {
        "mappo_same_actor_shielded": "Same actor + shield",
        "mappo_same_actor_raw": "Same actor, raw actions",
    }
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Execution & Success $\uparrow$ & Task $\uparrow$ & Outage $\downarrow$ & Worst pkt $\uparrow$ & AUV pen. $\downarrow$ & Crossing $\downarrow$ & Shield $\downarrow$ \\",
        r"\midrule",
    ]
    for policy in policies:
        records = by_policy[policy]
        cells = []
        for key, _, digits in METRICS:
            cells.append(fmt([float(record[key]) for record in records], digits))
        lines.append(f"{labels[policy]} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_table.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_table}")


if __name__ == "__main__":
    main()
