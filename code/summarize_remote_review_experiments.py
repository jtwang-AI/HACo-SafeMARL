from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = [
    ("success_mean", "Success"),
    ("task_completion_mean", "Task"),
    ("outage_rate_mean", "Outage"),
    ("worst_agent_packet_mean", "Worst pkt"),
    ("auv_collision_count_mean", "AUV coll."),
    ("colregs_violation_rate_mean", "COLREGs"),
    ("shield_interventions_mean", "Shield"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_seed(path: Path, policy_name: str) -> dict[str, float] | None:
    rows = [r for r in read_csv(path) if r["policy"] == policy_name]
    if not rows:
        return None
    out = {"episodes": sum(float(r["episodes"]) for r in rows)}
    for key, _ in METRICS:
        out[key] = mean(float(r[key]) for r in rows)
    return out


def fmt(values: list[float], digits: int = 2) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"${mean(values):.{digits}f}\\pm{stdev(values):.{digits}f}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/review_marl_baselines"))
    parser.add_argument("--out-table", type=Path, default=Path("tables/review_marl_baselines.tex"))
    parser.add_argument("--out-json", type=Path, default=Path("results/review_marl_baselines_summary.json"))
    args = parser.parse_args()

    variants = []
    for variant_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        policy_name = variant_dir.name
        seeds = []
        for seed_dir in sorted(variant_dir.glob("seed_*")):
            agg = seed_dir / "aggregate_metrics.csv"
            if agg.exists():
                rec = summarize_seed(agg, policy_name)
                if rec is not None:
                    rec["seed"] = seed_dir.name.replace("seed_", "")
                    seeds.append(rec)
        if seeds:
            variants.append({"variant": variant_dir.name, "seeds": seeds})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(variants, indent=2), encoding="utf-8")

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Remote MARL variant & Success $\uparrow$ & Task $\uparrow$ & Outage $\downarrow$ & Worst pkt $\uparrow$ & AUV coll. $\downarrow$ & COLREGs $\downarrow$ & Shield $\downarrow$ \\",
        r"\midrule",
    ]
    labels = {
        "mappo_acoustic_shield": "MAPPO + acoustic + shield",
        "mappo_no_acoustic_shield": "MAPPO w/o acoustic + shield",
        "mappo_acoustic_no_shield": "MAPPO + acoustic w/o shield",
    }
    for variant in variants:
        seeds = variant["seeds"]
        values = {key: [float(row[key]) for row in seeds] for key, _ in METRICS}
        lines.append(
            f"{labels.get(variant['variant'], variant['variant'])} & "
            f"{fmt(values['success_mean'])} & {fmt(values['task_completion_mean'])} & "
            f"{fmt(values['outage_rate_mean'])} & {fmt(values['worst_agent_packet_mean'])} & "
            f"{fmt(values['auv_collision_count_mean'], 1)} & {fmt(values['colregs_violation_rate_mean'])} & "
            f"{fmt(values['shield_interventions_mean'], 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_table.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_table}")


if __name__ == "__main__":
    main()
