"""Generate result tables and figures used by the paper."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from haco.env import HACoPilotEnv
from haco.experiment import scenario_config
from haco.policies import HACoSafeHeuristicPolicy

ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULTS = ROOT / "results" / "final_eval_local40" / "aggregate_metrics.csv"
FALLBACK_FINAL_RESULTS = ROOT / "results" / "final_eval_local20" / "aggregate_metrics.csv"
PILOT_RESULTS = ROOT / "results" / "local_eval_20" / "aggregate_metrics.csv"
RESULTS = FINAL_RESULTS if FINAL_RESULTS.exists() else FALLBACK_FINAL_RESULTS if FALLBACK_FINAL_RESULTS.exists() else PILOT_RESULTS
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"


POLICY_LABELS = {
    "fixed_auv_ga_pso_tlbo_proxy": "Fixed relay",
    "independent_greedy": "Independent",
    "communication_aware": "Comm.-aware",
    "haco_safemarl_pilot_no_acoustic": "HACo w/o acoustic",
    "haco_safemarl_pilot_no_shield": "HACo w/o shield",
    "haco_safemarl_pilot": "HACo-SafeMARL",
    "haco_safemarl_trained": "HACo-SafeMARL",
}

POLICY_ORDER = [
    "fixed_auv_ga_pso_tlbo_proxy",
    "independent_greedy",
    "communication_aware",
    "haco_safemarl_pilot_no_acoustic",
    "haco_safemarl_pilot_no_shield",
    "haco_safemarl_trained",
]

SCENARIO_LABELS = {
    "survey": "Survey",
    "traffic": "Traffic",
    "acoustic_degradation": "Acoustic",
    "emergency": "Emergency",
    "generalization": "Generalization",
}

SCENARIO_ORDER = [
    "survey",
    "traffic",
    "acoustic_degradation",
    "emergency",
    "generalization",
]

METHOD_COLORS = {
    "Fixed relay": "AIGray!62",
    "Independent": "AIAmber!70",
    "Comm.-aware": "AICyan!70",
    "HACo w/o acoustic": "AIViolet!62",
    "HACo w/o shield": "AIRed!62",
    "HACo-SafeMARL": "AITeal!86!black",
    "Balanced": "AIBlue!62",
}


def _ai_tikz_begin(options: str = "font=\\scriptsize,x=1cm,y=1cm") -> list[str]:
    """Shared AI/RL-paper visual grammar for TikZ figures."""
    return [
        fr"\begin{{tikzpicture}}[{options}]",
        r"\definecolor{AINavy}{RGB}{24,35,58}",
        r"\definecolor{AIBlue}{RGB}{43,100,214}",
        r"\definecolor{AICyan}{RGB}{0,153,188}",
        r"\definecolor{AITeal}{RGB}{0,137,123}",
        r"\definecolor{AIAmber}{RGB}{232,160,33}",
        r"\definecolor{AIRed}{RGB}{214,82,91}",
        r"\definecolor{AIViolet}{RGB}{126,87,194}",
        r"\definecolor{AIGray}{RGB}{112,122,138}",
        r"\definecolor{AIPanel}{RGB}{248,250,253}",
        r"\tikzset{aiPanel/.style={rounded corners=2pt,draw=AINavy!55,line width=0.45pt,fill=AIPanel},aiGrid/.style={draw=AINavy!10,line width=0.25pt},aiAxis/.style={draw=AINavy!72,line width=0.45pt},aiLink/.style={densely dashed,line width=0.55pt,draw=AICyan!72},aiMean/.style={line width=1.2pt,line cap=round,line join=round},aiSeed/.style={line width=0.38pt,opacity=0.32,draw=AIGray}}",
    ]


def read_rows() -> list[dict[str, str]]:
    with RESULTS.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def grouped(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def generate_main_table(rows: list[dict[str, str]]) -> None:
    by_policy = grouped(rows, "policy")
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Success $\uparrow$ & Task $\uparrow$ & Outage $\downarrow$ & Worst pkt $\uparrow$ & AUV coll. $\downarrow$ & COLREGs $\downarrow$ & Shield $\downarrow$ \\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        prow = by_policy[policy]
        vals = {
            "success": mean([float(r["success_mean"]) for r in prow]),
            "task": mean([float(r["task_completion_mean"]) for r in prow]),
            "outage": mean([float(r["outage_rate_mean"]) for r in prow]),
            "worst": mean([float(r["worst_agent_packet_mean"]) for r in prow]),
            "auv": mean([float(r["auv_collision_count_mean"]) for r in prow]),
            "colregs": mean([float(r["colregs_violation_rate_mean"]) for r in prow]),
            "shield": mean([float(r["shield_interventions_mean"]) for r in prow]),
        }
        lines.append(
            f"{POLICY_LABELS[policy]} & "
            f"{vals['success']:.2f} & {vals['task']:.2f} & {vals['outage']:.2f} & "
            f"{vals['worst']:.2f} & {vals['auv']:.1f} & {vals['colregs']:.2f} & "
            f"{vals['shield']:.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / "final_main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_scenario_table(rows: list[dict[str, str]]) -> None:
    main_policy = "haco_safemarl_trained"
    if not any(r["policy"] == main_policy for r in rows):
        main_policy = "haco_safemarl_pilot"
    keep = ["independent_greedy", "communication_aware", main_policy]
    by_pair = {(r["scenario"], r["policy"]): r for r in rows}
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Method & Success $\uparrow$ & Task $\uparrow$ & Outage $\downarrow$ & Worst pkt $\uparrow$ \\",
        r"\midrule",
    ]
    for scenario in SCENARIO_ORDER:
        for policy in keep:
            r = by_pair[(scenario, policy)]
            lines.append(
                f"{SCENARIO_LABELS[scenario]} & {POLICY_LABELS[policy]} & "
                f"{float(r['success_mean']):.2f} & {float(r['task_completion_mean']):.2f} & "
                f"{float(r['outage_rate_mean']):.2f} & {float(r['worst_agent_packet_mean']):.2f} \\\\"
            )
        if scenario != SCENARIO_ORDER[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / "scenario_results.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_tradeoff_figure(rows: list[dict[str, str]]) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    by_policy = grouped(rows, "policy")
    metrics = []
    for policy in POLICY_ORDER:
        prow = by_policy[policy]
        metrics.append(
            (
                POLICY_LABELS[policy],
                mean([float(r["worst_agent_packet_mean"]) for r in prow]),
                mean([float(r["outage_rate_mean"]) for r in prow]),
                mean([float(r["auv_collision_count_mean"]) for r in prow]),
            )
        )

    # A compact TikZ figure avoids a runtime dependency on matplotlib while keeping
    # the plotted values generated from the experiment CSV.
    lines = _ai_tikz_begin("x=1cm,y=0.45cm,font=\\scriptsize")
    lines.extend(
        [
            r"\node[anchor=west,font=\small\bfseries,text=AINavy] at (0,1.0) {Worst-agent packet delivery};",
            r"\node[anchor=west,font=\small\bfseries,text=AINavy] at (7.0,1.0) {Outage rate};",
            r"\draw[aiAxis] (0,0.45) -- (5.2,0.45);",
            r"\draw[aiAxis] (7,0.45) -- (12.2,0.45);",
        ]
    )
    for i, (label, worst, outage, collisions) in enumerate(metrics):
        y = -i
        lines.append(fr"\node[anchor=east,text=AINavy] at (-0.15,{y}) {{{label}}};")
        lines.append(fr"\draw[rounded corners=1pt,fill=AITeal!72,draw=AITeal!95!black] (0,{y-0.16:.2f}) rectangle ({5*worst:.2f},{y+0.16:.2f});")
        lines.append(fr"\node[anchor=west] at ({5*worst+0.08:.2f},{y}) {{{worst:.2f}}};")
        lines.append(fr"\draw[rounded corners=1pt,fill=AIRed!55,draw=AIRed!92!black] (7,{y-0.16:.2f}) rectangle ({7+5*outage:.2f},{y+0.16:.2f});")
        lines.append(fr"\node[anchor=west] at ({7+5*outage+0.08:.2f},{y}) {{{outage:.2f}}};")
        lines.append(fr"\node[anchor=west,text=AIGray] at (12.6,{y}) {{AUV coll. {collisions:.1f}}};")
    lines.extend(
        [
            r"\draw[-{Latex[length=1.8mm]},draw=AINavy!65] (0,-6.0) -- (5.2,-6.0) node[right] {higher is better};",
            r"\draw[-{Latex[length=1.8mm]},draw=AINavy!65] (7,-6.0) -- (12.2,-6.0) node[right] {higher is worse};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "pilot_comm_tradeoff_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def _axis_points(values: list[tuple[float, float]], xmin: float, xmax: float, ymin: float, ymax: float, width: float, height: float) -> str:
    coords = []
    for x, y in values:
        px = (x - xmin) / max(xmax - xmin, 1e-9) * width
        py = (y - ymin) / max(ymax - ymin, 1e-9) * height
        coords.append(f"({px:.2f},{py:.2f})")
    return " -- ".join(coords)


def _axis_coords(values: list[tuple[float, float]], xmin: float, xmax: float, ymin: float, ymax: float, width: float, height: float) -> list[tuple[float, float]]:
    coords = []
    for x, y in values:
        px = (x - xmin) / max(xmax - xmin, 1e-9) * width
        py = (y - ymin) / max(ymax - ymin, 1e-9) * height
        coords.append((px, py))
    return coords


def _tikz_path(coords: list[tuple[float, float]]) -> str:
    return " -- ".join(f"({x:.2f},{y:.2f})" for x, y in coords)


def _band_path(
    xs: list[float],
    means: list[float],
    stds: list[float],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: float,
    height: float,
) -> str:
    upper_vals = [min(max(m + s, ymin), ymax) for m, s in zip(means, stds)]
    lower_vals = [min(max(m - s, ymin), ymax) for m, s in zip(means, stds)]
    upper = _axis_coords(list(zip(xs, upper_vals)), xmin, xmax, ymin, ymax, width, height)
    lower = _axis_coords(list(zip(xs, lower_vals)), xmin, xmax, ymin, ymax, width, height)
    return _tikz_path(upper + list(reversed(lower)))


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _mappo_seed_dirs() -> list[Path]:
    base_100 = ROOT / "results" / "mappo_multiseed_100"
    base = base_100 if base_100.exists() else ROOT / "results" / "mappo_multiseed"
    if not base.exists():
        return []
    return sorted(p for p in base.glob("seed_*") if (p / "ppo_history.json").exists())


def _load_multiseed_ppo() -> tuple[list[float], dict[str, tuple[list[float], list[float]]], int]:
    seed_dirs = _mappo_seed_dirs()
    histories = []
    for seed_dir in seed_dirs:
        ppo = json.loads((seed_dir / "ppo_history.json").read_text(encoding="utf-8"))
        histories.append({int(r["ppo_update"]): r for r in ppo})
    if not histories:
        return [], {}, 0
    updates = sorted(set.intersection(*(set(h.keys()) for h in histories)))
    metrics = {}
    for key in ["mean_task_completion", "mean_outage_rate", "loss"]:
        means = []
        stds = []
        for update in updates:
            vals = [float(h[update][key]) for h in histories]
            m, s = _mean_std(vals)
            means.append(m)
            stds.append(s)
        metrics[key] = (means, stds)
    return [float(u) for u in updates], metrics, len(histories)


def _load_multiseed_series(filename: str, x_key: str, y_key: str) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    histories = []
    for seed_dir in _mappo_seed_dirs():
        path = seed_dir / filename
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        histories.append({int(r[x_key]): float(r[y_key]) for r in rows})
    if not histories:
        return [], [], [], []
    xs_int = sorted(set.intersection(*(set(h.keys()) for h in histories)))
    traces = [[hist[x] for x in xs_int] for hist in histories]
    means, stds = [], []
    for idx, _ in enumerate(xs_int):
        m, s = _mean_std([trace[idx] for trace in traces])
        means.append(m)
        stds.append(s)
    return [float(x) for x in xs_int], traces, means, stds


def _nice_ticks(ymin: float, ymax: float, count: int = 3) -> list[float]:
    if ymax <= ymin:
        return [ymin]
    if ymin >= 0 and ymax <= 1:
        return [0.0, 0.5, 1.0]
    return [ymin + i * (ymax - ymin) / max(count - 1, 1) for i in range(count)]


def _metric_panel(
    lines: list[str],
    xshift: float,
    yshift: float,
    title: str,
    xs: list[float],
    traces: list[list[float]],
    means: list[float],
    stds: list[float],
    color: str,
    ymin: float,
    ymax: float,
    ylabel: str,
    tag: str,
    width: float = 4.15,
    height: float = 2.25,
    show_xtick_labels: bool = True,
) -> None:
    xmin, xmax = min(xs), max(xs)
    mean_path = _axis_points(list(zip(xs, means)), xmin, xmax, ymin, ymax, width, height)
    band = _band_path(xs, means, stds, xmin, xmax, ymin, ymax, width, height)
    seed_paths = [
        _axis_points(list(zip(xs, trace)), xmin, xmax, ymin, ymax, width, height)
        for trace in traces
    ]
    lines.append(fr"\begin{{scope}}[shift={{({xshift:.2f},{yshift:.2f})}}]")
    lines.append(fr"\draw[aiPanel] (0,0) rectangle ({width:.2f},{height:.2f});")
    for tick in range(int(xmin), int(xmax) + 1, max(1, int((xmax - xmin) // 3) or 1)):
        x = (tick - xmin) / max(xmax - xmin, 1e-9) * width
        lines.append(fr"\draw[aiGrid] ({x:.2f},0) -- ({x:.2f},{height:.2f});")
        if show_xtick_labels:
            lines.append(fr"\node[below] at ({x:.2f},-0.03) {{{tick}}};")
    for val in _nice_ticks(ymin, ymax):
        y = (val - ymin) / max(ymax - ymin, 1e-9) * height
        lines.append(fr"\draw[aiGrid] (0,{y:.2f}) -- ({width:.2f},{y:.2f});")
        label = f"{val:.1f}" if abs(val) < 10 else f"{val:.0f}"
        lines.append(fr"\node[anchor=east] at (-0.06,{y:.2f}) {{{label}}};")
    for seed_path in seed_paths:
        lines.append(fr"\draw[aiSeed] {seed_path};")
    lines.append(fr"\fill[{color},opacity=0.18] {band} -- cycle;")
    lines.append(fr"\draw[{color},aiMean] {mean_path};")
    lines.append(fr"\node[anchor=south west,font=\bfseries\scriptsize,text=AINavy] at (0.02,{height + 0.14:.2f}) {{{tag} {title}}};")
    lines.append(fr"\node[anchor=south,rotate=90,text=AINavy] at (-0.48,{height / 2:.2f}) {{{ylabel}}};")
    lines.append(r"\end{scope}")


def generate_training_curve_figure() -> None:
    task_xs, task_traces, task, task_std = _load_multiseed_series(
        "ppo_history.json", "ppo_update", "mean_task_completion"
    )
    outage_xs, outage_traces, outage, outage_std = _load_multiseed_series(
        "ppo_history.json", "ppo_update", "mean_outage_rate"
    )
    loss_xs, loss_traces, loss, loss_std = _load_multiseed_series("ppo_history.json", "ppo_update", "loss")
    bc_xs, bc_traces, bc, bc_std = _load_multiseed_series("bc_history.json", "bc_epoch", "loss")
    if task_xs and outage_xs and loss_xs and bc_xs:
        loss_ymin = min(min(trace) for trace in loss_traces)
        loss_ymax = max(max(trace) for trace in loss_traces)
        loss_pad = 0.06 * (loss_ymax - loss_ymin + 1e-6)
        bc_ymin = min(min(trace) for trace in bc_traces)
        bc_ymax = max(max(trace) for trace in bc_traces)
        bc_pad = 0.08 * (bc_ymax - bc_ymin + 1e-6)
        lines = _ai_tikz_begin()
        _metric_panel(lines, 0.0, 3.05, "warm-start imitation", bc_xs, bc_traces, bc, bc_std, "AIViolet",
                      bc_ymin - bc_pad, bc_ymax + bc_pad, "MSE loss", "(A)", show_xtick_labels=False)
        _metric_panel(lines, 5.20, 3.05, "task completion", task_xs, task_traces, task, task_std, "AITeal",
                      0.0, 1.0, "rate", "(B)", show_xtick_labels=False)
        _metric_panel(lines, 0.0, 0.0, "communication outage", outage_xs, outage_traces, outage, outage_std, "AIRed",
                      0.0, 1.0, "rate", "(C)")
        _metric_panel(lines, 5.20, 0.0, "PPO objective", loss_xs, loss_traces, loss, loss_std, "AIBlue",
                      loss_ymin - loss_pad, loss_ymax + loss_pad, "loss", "(D)")
        lines.extend(
            [
                r"\node[anchor=west,align=left,text width=9.4cm] at (0,-0.88) {Thin gray traces show individual seeds; colored curves show the four-seed mean and translucent bands show one standard deviation.};",
                r"\end{tikzpicture}",
                "",
            ]
        )
        (FIG_DIR / "training_curves_tikz.tex").write_text("\n".join(lines), encoding="utf-8")
        return

    ppo_path = ROOT / "results" / "torch_mappo" / "ppo_history.json"
    full_path = ROOT / "results" / "torch_mappo" / "training_history.json"
    if full_path.exists() or ppo_path.exists():
        payload = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else {}
        ppo = payload.get("ppo") or json.loads(ppo_path.read_text(encoding="utf-8"))
        if ppo:
            xs = [float(r["ppo_update"]) for r in ppo]
            task = [float(r["mean_task_completion"]) for r in ppo]
            outage = [float(r["mean_outage_rate"]) for r in ppo]
            loss = [float(r["loss"]) for r in ppo]
            xmin, xmax = min(xs), max(xs)
            width, height = 8.4, 3.2
            task_path = _axis_points(list(zip(xs, task)), xmin, xmax, 0.0, 1.0, width, height)
            outage_path = _axis_points(list(zip(xs, outage)), xmin, xmax, 0.0, 1.0, width, height)
            ymin, ymax = min(loss) - 0.05 * (max(loss) - min(loss) + 1e-6), max(loss) + 0.05 * (max(loss) - min(loss) + 1e-6)
            loss_path = _axis_points(list(zip(xs, loss)), xmin, xmax, ymin, ymax, width, height)
            lines = [
                r"\begin{tikzpicture}[font=\scriptsize]",
                fr"\draw[->,gray!80] (0,0) -- ({width + 0.35:.2f},0) node[right] {{MAPPO update}};",
                fr"\draw[->,gray!80] (0,0) -- (0,{height + 0.35:.2f}) node[above] {{task/outage}};",
                fr"\draw[->,gray!80] ({width + 0.15:.2f},0) -- ({width + 0.15:.2f},{height + 0.35:.2f}) node[above] {{loss}};",
            ]
            for tick in range(int(xmin), int(xmax) + 1):
                x = (tick - xmin) / max(xmax - xmin, 1e-9) * width
                lines.append(fr"\draw[gray!30] ({x:.2f},0) -- ({x:.2f},{height:.2f});")
                lines.append(fr"\node[below] at ({x:.2f},0) {{{tick}}};")
            for frac, label in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
                y = frac * height
                lines.append(fr"\draw[gray!30] (0,{y:.2f}) -- ({width:.2f},{y:.2f});")
                lines.append(fr"\node[anchor=east] at (-0.08,{y:.2f}) {{{label}}};")
            lines.extend(
                [
                    fr"\draw[very thick,green!55!black] {task_path};",
                    fr"\draw[very thick,red!65] {outage_path};",
                    fr"\draw[very thick,blue!65,dashed] {loss_path};",
                    fr"\node[draw,fill=white,anchor=north west,align=left] at (5.35,{height - 0.10:.2f}) {{\textcolor{{green!55!black}}{{task completion}}\\\textcolor{{red!65}}{{outage rate}}\\\textcolor{{blue!65}}{{PPO loss}}}};",
                    r"\node[anchor=north west,align=left,text width=8.4cm] at (0,-0.62) {Neural graph-attention MAPPO fine-tuning curve generated from remote training logs.};",
                    r"\end{tikzpicture}",
                    "",
                ]
            )
            (FIG_DIR / "training_curves_tikz.tex").write_text("\n".join(lines), encoding="utf-8")
            return

    history_path = ROOT / "results" / "cem_local" / "cem_history.json"
    if not history_path.exists():
        return
    hist = json.loads(history_path.read_text(encoding="utf-8"))
    gens = [float(r["generation"]) for r in hist]
    best = [float(r["best_score"]) for r in hist]
    mean_scores = [float(r["mean_score"]) for r in hist]
    ymin = min(best + mean_scores) - 0.25
    ymax = max(best + mean_scores) + 0.25
    xmin = min(gens)
    xmax = max(gens)
    width, height = 8.4, 3.2
    best_path = _axis_points(list(zip(gens, best)), xmin, xmax, ymin, ymax, width, height)
    mean_path = _axis_points(list(zip(gens, mean_scores)), xmin, xmax, ymin, ymax, width, height)
    lines = [
        r"\begin{tikzpicture}[font=\scriptsize]",
        fr"\draw[->,gray!80] (0,0) -- ({width + 0.35:.2f},0) node[right] {{generation}};",
        fr"\draw[->,gray!80] (0,0) -- (0,{height + 0.35:.2f}) node[above] {{training objective}};",
    ]
    for tick in range(int(xmin), int(xmax) + 1):
        x = (tick - xmin) / max(xmax - xmin, 1e-9) * width
        lines.append(fr"\draw[gray!35] ({x:.2f},0) -- ({x:.2f},{height:.2f});")
        lines.append(fr"\node[below] at ({x:.2f},0) {{{tick}}};")
    for frac, label in [(0.0, f"{ymin:.1f}"), (0.5, f"{(ymin + ymax) / 2:.1f}"), (1.0, f"{ymax:.1f}")]:
        y = frac * height
        lines.append(fr"\draw[gray!35] (0,{y:.2f}) -- ({width:.2f},{y:.2f});")
        lines.append(fr"\node[anchor=east] at (-0.08,{y:.2f}) {{{label}}};")
    lines.extend(
        [
            fr"\draw[very thick,green!55!black] {best_path};",
            fr"\draw[very thick,blue!65,dashed] {mean_path};",
            fr"\node[draw,fill=white,anchor=north west,align=left] at (5.55,{height - 0.15:.2f}) {{\textcolor{{green!55!black}}{{best candidate}}\\\textcolor{{blue!65}}{{population mean}}}};",
            fr"\node[anchor=north west,align=left,text width=8.2cm] at (0,-0.62) {{Warm-start policy optimization before graph-attention MAPPO fine-tuning. The curve reports the same constrained objective used by the simulator.}};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "training_curves_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def _summary_by_policy(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_policy = grouped(rows, "policy")
    out = {}
    for policy, prow in by_policy.items():
        out[policy] = {
            "task": mean([float(r["task_completion_mean"]) for r in prow]),
            "outage": mean([float(r["outage_rate_mean"]) for r in prow]),
            "worst": mean([float(r["worst_agent_packet_mean"]) for r in prow]),
            "auv": mean([float(r["auv_collision_count_mean"]) for r in prow]),
            "shield": mean([float(r["shield_interventions_mean"]) for r in prow]),
        }
    balanced = _summarize_file(ROOT / "results" / "final_eval_balanced40" / "aggregate_metrics.csv")
    if balanced is not None:
        out["balanced"] = {
            "task": balanced["task"],
            "outage": balanced["outage"],
            "worst": balanced["worst"],
            "auv": balanced["auv"],
            "shield": balanced["shield"],
        }
    task_priority = _summarize_file(
        ROOT / "results" / "task_priority_policy" / "aggregate_metrics.csv",
        "haco_safemarl_task_priority",
    )
    if task_priority is not None:
        out["task_priority"] = {
            "task": task_priority["task"],
            "outage": task_priority["outage"],
            "worst": task_priority["worst"],
            "auv": task_priority["auv"],
            "shield": task_priority["shield"],
        }
    return out


def generate_ablation_figure(rows: list[dict[str, str]]) -> None:
    s = _summary_by_policy(rows)
    methods = [
        ("haco_safemarl_trained", "HACo-SafeMARL"),
        ("haco_safemarl_pilot_no_acoustic", "w/o acoustic"),
        ("haco_safemarl_pilot_no_shield", "w/o shield"),
        ("balanced", "balanced pref."),
        ("task_priority", "task pref."),
    ]
    metrics = [
        ("worst", "Worst-agent packet", 1.0, "AITeal!82", r"$\uparrow$"),
        ("task", "Task completion", 1.0, "AIBlue!72", r"$\uparrow$"),
        ("auv", "AUV collisions", 70.0, "AIRed!72", r"$\downarrow$"),
    ]
    lines = _ai_tikz_begin()
    panel_w = 3.6
    for i, (_, label) in enumerate(methods):
        y = -0.45 * i
        lines.append(fr"\node[anchor=east] at (2.35,{y:.2f}) {{{label}}};")
    for mi, (key, title, max_v, color, arrow) in enumerate(metrics):
        x0 = 2.55 + mi * 4.35
        lines.append(fr"\node[anchor=west,font=\small] at ({x0:.1f},1.15) {{{title} {arrow}}};")
        lines.append(fr"\draw[gray!35] ({x0:.1f},0.78) -- ({x0 + panel_w:.1f},0.78);")
        for i, (policy_key, label) in enumerate(methods):
            if policy_key not in s:
                continue
            val = s[policy_key][key]
            y = -0.45 * i
            bar = min(val / max_v, 1.0) * panel_w
            fill = color if policy_key == "haco_safemarl_trained" else "AIGray!28"
            if policy_key == "balanced":
                fill = "AICyan!40"
            if policy_key == "task_priority":
                fill = "AIAmber!60"
            lines.append(fr"\draw[rounded corners=1pt,fill={fill},draw=AINavy!35] ({x0:.1f},{y - 0.13:.2f}) rectangle ({x0 + bar:.2f},{y + 0.13:.2f});")
            label_value = f"{val:.2f}" if key != "auv" else f"{val:.1f}"
            lines.append(fr"\node[anchor=west] at ({x0 + bar + 0.05:.2f},{y:.2f}) {{{label_value}}};")
        lines.append(fr"\draw[-{{Latex[length=1.6mm]}},draw=AINavy!55] ({x0:.1f},-2.05) -- ({x0 + panel_w:.1f},-2.05);")
    lines.extend(
        [
            r"\node[anchor=west,align=left,text width=14.2cm] at (0,-2.75) {Ablations isolate the acoustic link features and the safety shield. Removing acoustic feedback hurts the weakest link; removing the shield causes a large underwater-collision increase.};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "ablation_bars_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"L {x:.1f} {y:.1f}" for x, y in points[1:])


def _load_best_params() -> dict:
    path = ROOT / "results" / "cem_local" / "best_policy.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("params", payload.get("best_params", {}))


def _simulate_emergency_trace() -> tuple[object, list[dict[str, object]]]:
    cfg = scenario_config("emergency", seed=2026042907, num_auvs=3)
    env = HACoPilotEnv(cfg)
    policy = HACoSafeHeuristicPolicy(params=_load_best_params(), name="haco_safemarl_trained")
    obs = env.reset()
    frames = []
    done = False
    while not done:
        frames.append(
            {
                "t": env.t,
                "usv": env.usv_pos.copy(),
                "auv": env.auv_pos.copy(),
                "depth": env.auv_depth.copy(),
                "tasks": env.tasks.copy(),
                "obstacles": env.obstacles.copy(),
                "vessels": env.vessel_pos.copy(),
                "packets": env.packet_probs().copy(),
            }
        )
        usv_action, auv_actions, shield = policy.act(obs)
        obs, _, done = env.step(usv_action, auv_actions, shield=shield)
    return cfg, frames


def _map_point(pt: np.ndarray, size: float = 3.4) -> tuple[float, float]:
    return float(pt[0]) / 2000.0 * size, float(pt[1]) / 2000.0 * size


def generate_emergency_sequence_figure() -> None:
    _, frames = _simulate_emergency_trace()
    selected_steps = [0, 90, 160, 240]
    lines = _ai_tikz_begin("font=\\scriptsize")
    panel = 3.4
    gap = 0.55
    for pi, step in enumerate(selected_steps):
        frame = min(frames, key=lambda f: abs(int(f["t"]) - step))
        xoff = pi * (panel + gap)
        lines.append(fr"\begin{{scope}}[shift={{({xoff:.2f},0)}}]")
        lines.append(fr"\draw[aiPanel,fill=AICyan!4] (0,0) rectangle ({panel:.2f},{panel:.2f});")
        for grid in [0.85, 1.70, 2.55]:
            lines.append(fr"\draw[aiGrid] ({grid:.2f},0) -- ({grid:.2f},{panel:.2f});")
            lines.append(fr"\draw[aiGrid] (0,{grid:.2f}) -- ({panel:.2f},{grid:.2f});")
        for ox, oy, rad in frame["obstacles"]:
            x, y = _map_point(np.array([ox, oy]), panel)
            rr = max(rad / 2000.0 * panel, 0.045)
            lines.append(fr"\draw[fill=AIGray!28,draw=AIGray!70] ({x:.2f},{y:.2f}) circle ({rr:.2f});")
        for task in frame["tasks"]:
            x, y = _map_point(task, panel)
            lines.append(fr"\node[star,star points=5,draw=AIAmber!90!black,fill=AIAmber!55,inner sep=0.8pt] at ({x:.2f},{y:.2f}) {{}};")
        for vessel in frame["vessels"]:
            x, y = _map_point(vessel, panel)
            lines.append(fr"\node[diamond,draw=AIRed!80!black,fill=AIRed!22,inner sep=1.1pt] at ({x:.2f},{y:.2f}) {{}};")
        start = 0
        trace = frames[start : int(frame["t"]) + 1 : 12]
        usv_path = [_map_point(f["usv"], panel) for f in trace]
        auv0_path = [_map_point(f["auv"][0], panel) for f in trace]
        if len(usv_path) > 1:
            lines.append(fr"\draw[AITeal!90!black,line width=1.0pt,line cap=round] ({usv_path[0][0]:.2f},{usv_path[0][1]:.2f}) -- " + " -- ".join(f"({x:.2f},{y:.2f})" for x, y in usv_path[1:]) + ";")
        if len(auv0_path) > 1:
            lines.append(fr"\draw[AIBlue,line width=1.0pt,line cap=round] ({auv0_path[0][0]:.2f},{auv0_path[0][1]:.2f}) -- " + " -- ".join(f"({x:.2f},{y:.2f})" for x, y in auv0_path[1:]) + ";")
        ux, uy = _map_point(frame["usv"], panel)
        lines.append(fr"\node[rectangle,draw=AITeal!90!black,fill=AITeal!24,rounded corners=1pt,inner sep=1.4pt] at ({ux:.2f},{uy:.2f}) {{U}};")
        for ai, auv in enumerate(frame["auv"]):
            x, y = _map_point(auv, panel)
            fill = "AIBlue!70" if ai == 0 else "AIBlue!18"
            lines.append(fr"\node[circle,draw=AIBlue!85!black,fill={fill},inner sep=1.2pt] at ({x:.2f},{y:.2f}) {{}};")
            lines.append(fr"\draw[aiLink] ({ux:.2f},{uy:.2f}) -- ({x:.2f},{y:.2f});")
        worst = float(np.min(frame["packets"]))
        emergency = " emergency" if int(frame["t"]) >= 150 else ""
        lines.append(fr"\node[anchor=west,font=\bfseries,text=AINavy] at (0,{panel + 0.25:.2f}) {{$t={int(frame['t'])}${emergency}}};")
        lines.append(fr"\node[anchor=east,text=AINavy!80] at ({panel:.2f},-0.24) {{min pkt={worst:.2f}}};")
        lines.append(r"\end{scope}")
    lines.extend(
        [
            r"\node[anchor=west,align=left,text width=14.5cm] at (0,-0.8) {Emergency response sequence. The first AUV enters a deeper, lower-link state after mid-mission; the relay and shielded AUV motion preserve the weakest acoustic link while avoiding obstacles and surface traffic.};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "emergency_rescue_sequence_tikz.tex").write_text("\n".join(lines), encoding="utf-8")
    generate_emergency_animation_svg(frames)


def generate_emergency_animation_svg(frames: list[dict[str, object]]) -> None:
    sample = frames[::10]
    width = 720
    height = 720
    scale = width / 2000.0

    def sx(pt):
        return float(pt[0]) * scale

    def sy(pt):
        return height - float(pt[1]) * scale

    def values_xy(seq, getter):
        xs = [f"{sx(getter(f)):.1f}" for f in seq]
        ys = [f"{sy(getter(f)):.1f}" for f in seq]
        return ";".join(xs), ";".join(ys)

    usv_x, usv_y = values_xy(sample, lambda f: f["usv"])
    auv_x, auv_y = values_xy(sample, lambda f: f["auv"][0])
    first = frames[0]
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ecfbff"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:18px}.trail{fill:none;stroke-width:3;opacity:.7}</style>',
    ]
    for ox, oy, rad in first["obstacles"]:
        items.append(f'<circle cx="{ox*scale:.1f}" cy="{height-oy*scale:.1f}" r="{rad*scale:.1f}" fill="#c7c7c7" stroke="#777"/>')
    for task in first["tasks"]:
        items.append(f'<circle cx="{sx(task):.1f}" cy="{sy(task):.1f}" r="7" fill="#ffd84d" stroke="#a77a00"/>')
    usv_poly = " ".join(f"{sx(f['usv']):.1f},{sy(f['usv']):.1f}" for f in sample)
    auv_poly = " ".join(f"{sx(f['auv'][0]):.1f},{sy(f['auv'][0]):.1f}" for f in sample)
    items.append(f'<polyline class="trail" stroke="#188a42" points="{usv_poly}"/>')
    items.append(f'<polyline class="trail" stroke="#1b5fc9" points="{auv_poly}"/>')
    items.append(f'<circle r="10" fill="#29a35a" stroke="#064d23"><animate attributeName="cx" values="{usv_x}" dur="8s" repeatCount="indefinite"/><animate attributeName="cy" values="{usv_y}" dur="8s" repeatCount="indefinite"/></circle>')
    items.append(f'<circle r="8" fill="#2f71d6" stroke="#073d91"><animate attributeName="cx" values="{auv_x}" dur="8s" repeatCount="indefinite"/><animate attributeName="cy" values="{auv_y}" dur="8s" repeatCount="indefinite"/></circle>')
    items.append('<text x="22" y="34">Emergency relay-recovery animation: green=USV relay, blue=AUV in emergency</text>')
    items.append("</svg>")
    (FIG_DIR / "emergency_rescue_animation.svg").write_text("\n".join(items), encoding="utf-8")


def _summarize_file(path: Path, policy: str = "haco_safemarl_trained") -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["policy"] == policy]
    if not rows:
        return None
    return {
        "success": mean([float(r["success_mean"]) for r in rows]),
        "task": mean([float(r["task_completion_mean"]) for r in rows]),
        "outage": mean([float(r["outage_rate_mean"]) for r in rows]),
        "worst": mean([float(r["worst_agent_packet_mean"]) for r in rows]),
        "auv": mean([float(r["auv_collision_count_mean"]) for r in rows]),
        "colregs": mean([float(r["colregs_violation_rate_mean"]) for r in rows]),
        "shield": mean([float(r["shield_interventions_mean"]) for r in rows]),
    }


def generate_preference_table() -> None:
    variants = [
        ("Communication-safety", ROOT / "results" / "final_eval_local40" / "aggregate_metrics.csv", "haco_safemarl_trained"),
        ("Balanced", ROOT / "results" / "final_eval_balanced40" / "aggregate_metrics.csv", "haco_safemarl_trained"),
        ("Task-priority", ROOT / "results" / "task_priority_policy" / "aggregate_metrics.csv", "haco_safemarl_task_priority"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Preference & Success $\uparrow$ & Task $\uparrow$ & Outage $\downarrow$ & Worst pkt $\uparrow$ & AUV coll. $\downarrow$ & Shield $\downarrow$ \\",
        r"\midrule",
    ]
    added = 0
    for name, path, policy in variants:
        s = _summarize_file(path, policy)
        if s is None:
            continue
        added += 1
        lines.append(
            f"{name} & {s['success']:.2f} & {s['task']:.2f} & {s['outage']:.2f} & "
            f"{s['worst']:.2f} & {s['auv']:.1f} & {s['shield']:.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    if added:
        (TABLE_DIR / "preference_results.tex").write_text("\n".join(lines), encoding="utf-8")


def _summarize_mappo_seed(seed_dir: Path) -> dict[str, float] | None:
    path = seed_dir / "aggregate_metrics.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["policy"] == "haco_safemarl_mappo"]
    if not rows:
        return None
    return {
        "seed": float(seed_dir.name.replace("seed_", "")),
        "success": mean([float(r["success_mean"]) for r in rows]),
        "task": mean([float(r["task_completion_mean"]) for r in rows]),
        "outage": mean([float(r["outage_rate_mean"]) for r in rows]),
        "worst": mean([float(r["worst_agent_packet_mean"]) for r in rows]),
        "auv": mean([float(r["auv_collision_count_mean"]) for r in rows]),
        "shield": mean([float(r["shield_interventions_mean"]) for r in rows]),
    }


def generate_neural_multiseed_table() -> None:
    summaries = [s for s in (_summarize_mappo_seed(p) for p in _mappo_seed_dirs()) if s is not None]
    if not summaries:
        return

    def fmt_mean_std(key: str, digits: int = 2) -> str:
        m, s = _mean_std([row[key] for row in summaries])
        return f"${m:.{digits}f}\\pm{s:.{digits}f}$"

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Seed & Success $\uparrow$ & Task $\uparrow$ & Outage $\downarrow$ & Worst pkt $\uparrow$ & AUV coll. $\downarrow$ & Shield $\downarrow$ \\",
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{int(row['seed'])} & {row['success']:.2f} & {row['task']:.2f} & {row['outage']:.2f} & "
            f"{row['worst']:.2f} & {row['auv']:.1f} & {row['shield']:.0f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        "Mean $\\pm$ std. & "
        f"{fmt_mean_std('success')} & {fmt_mean_std('task')} & {fmt_mean_std('outage')} & "
        f"{fmt_mean_std('worst')} & {fmt_mean_std('auv', 1)} & {fmt_mean_std('shield', 0)} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / "neural_multiseed_results.tex").write_text("\n".join(lines), encoding="utf-8")

    summary_payload = {
        "num_seeds": len(summaries),
        "seeds": [int(row["seed"]) for row in summaries],
        "mean_std": {
            key: {"mean": _mean_std([row[key] for row in summaries])[0], "std": _mean_std([row[key] for row in summaries])[1]}
            for key in ["success", "task", "outage", "worst", "auv", "shield"]
        },
        "per_seed": summaries,
    }
    (ROOT / "results" / "mappo_multiseed_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def generate_bc_loss_figure() -> None:
    histories = []
    for seed_dir in _mappo_seed_dirs():
        path = seed_dir / "bc_history.json"
        if path.exists():
            hist = json.loads(path.read_text(encoding="utf-8"))
            histories.append({int(r["bc_epoch"]): float(r["loss"]) for r in hist})
    if not histories:
        return
    epochs = sorted(set.intersection(*(set(h.keys()) for h in histories)))
    means, stds = [], []
    for epoch in epochs:
        m, s = _mean_std([h[epoch] for h in histories])
        means.append(m)
        stds.append(s)
    xmin, xmax = float(min(epochs)), float(max(epochs))
    all_vals = [value for hist in histories for value in hist.values()]
    raw_ymin = min(min(all_vals), min(m - s for m, s in zip(means, stds)))
    raw_ymax = max(max(all_vals), max(m + s for m, s in zip(means, stds)))
    ypad = max((raw_ymax - raw_ymin) * 0.08, 0.08)
    ymin = max(0.0, raw_ymin - ypad)
    ymax = raw_ymax + ypad
    width, height = 8.0, 3.35
    xs = [float(e) for e in epochs]
    path = _axis_points(list(zip(xs, means)), xmin, xmax, ymin, ymax, width, height)
    band = _band_path(xs, means, stds, xmin, xmax, ymin, ymax, width, height)
    seed_paths = [
        _axis_points([(float(e), hist[e]) for e in epochs], xmin, xmax, ymin, ymax, width, height)
        for hist in histories
    ]

    def y_coord(val: float) -> float:
        return (val - ymin) / max(ymax - ymin, 1e-9) * height

    lines = _ai_tikz_begin()
    lines.extend(
        [
            fr"\fill[white] (-0.72,-0.58) rectangle ({width + 0.62:.2f},{height + 0.58:.2f});",
            fr"\draw[aiPanel] (0,0) rectangle ({width:.2f},{height:.2f});",
        ]
    )
    x_tick_step = 20 if xmax >= 80 else max(1, int((xmax - xmin) // 5) or 1)
    x_ticks = list(range(int(xmin), int(xmax) + 1, x_tick_step))
    if int(xmax) not in x_ticks:
        x_ticks.append(int(xmax))
    minor_tick_step = 10 if xmax >= 80 else x_tick_step
    minor_ticks = list(range(int(xmin), int(xmax) + 1, minor_tick_step))

    for epoch in minor_ticks:
        x = (epoch - xmin) / max(xmax - xmin, 1e-9) * width
        grid_color = "AINavy!16" if epoch in x_ticks else "AINavy!8"
        lines.append(fr"\draw[{grid_color},line width=0.25pt] ({x:.2f},0) -- ({x:.2f},{height:.2f});")
    for epoch in x_ticks:
        x = (epoch - xmin) / max(xmax - xmin, 1e-9) * width
        lines.append(fr"\draw[AINavy!62,line width=0.35pt] ({x:.2f},0) -- ({x:.2f},-0.05);")
        lines.append(fr"\node[below] at ({x:.2f},-0.05) {{{epoch}}};")
    for val in _nice_ticks(ymin, ymax, count=5):
        y = y_coord(val)
        lines.append(fr"\draw[AINavy!14,line width=0.25pt] (0,{y:.2f}) -- ({width:.2f},{y:.2f});")
        lines.append(fr"\draw[AINavy!62,line width=0.35pt] (0,{y:.2f}) -- (-0.05,{y:.2f});")
        lines.append(fr"\node[anchor=east] at (-0.10,{y:.2f}) {{{val:.1f}}};")
    lines.append(r"\begin{scope}")
    lines.append(fr"\clip (0,0) rectangle ({width:.2f},{height:.2f});")
    for seed_path in seed_paths:
        lines.append(fr"\draw[AIGray,line width=0.45pt,opacity=0.36] {seed_path};")
    lines.extend(
        [
            fr"\fill[AIBlue,opacity=0.18] {band} -- cycle;",
            fr"\draw[AIBlue,aiMean] {path};",
        ]
    )
    marker_epochs = set(x_ticks)
    for epoch, (x, y) in zip(epochs, _axis_coords(list(zip(xs, means)), xmin, xmax, ymin, ymax, width, height)):
        if epoch in marker_epochs:
            lines.append(fr"\fill[AIBlue] ({x:.2f},{y:.2f}) circle (1.35pt);")
            lines.append(fr"\draw[white,line width=0.25pt] ({x:.2f},{y:.2f}) circle (1.35pt);")
    lines.append(r"\end{scope}")
    lines.extend(
        [
            fr"\node[anchor=north] at ({width / 2:.2f},-0.42) {{Behavior-cloning epoch}};",
            fr"\node[anchor=south,rotate=90] at (-0.62,{height / 2:.2f}) {{MSE loss}};",
            fr"\node[anchor=south west,font=\scriptsize\bfseries,text=AINavy] at (0.04,{height + 0.16:.2f}) {{Warm-start imitation convergence}};",
            fr"\node[draw=AINavy!35,fill=white,rounded corners=1pt,inner sep=3pt,anchor=north east,align=left] at ({width - 0.12:.2f},{height - 0.12:.2f}) {{\raisebox{{0.5pt}}{{\tikz{{\draw[AIBlue,line width=1.1pt] (0,0) -- (0.42,0); \fill[AIBlue] (0.21,0) circle (1.15pt);}}}} mean\\\raisebox{{0.5pt}}{{\tikz{{\draw[AIBlue,line width=3pt,opacity=0.22] (0,0) -- (0.42,0);}}}} std. band\\\raisebox{{0.5pt}}{{\tikz{{\draw[AIGray,line width=0.45pt,opacity=0.55] (0,0) -- (0.42,0);}}}} seed traces}};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "bc_loss_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_neural_checkpoint_bars() -> None:
    seed_rows = []
    for seed_dir in _mappo_seed_dirs():
        rows = _episode_rows_for_mappo(seed_dir)
        if rows:
            seed_rows.append((seed_dir.name.replace("seed_", "")[-2:], rows))
    if not seed_rows:
        return
    metrics = [
        ("worst_agent_packet", "Worst packet", "AITeal!88!black", 0.0, 1.0, r"$\uparrow$"),
        ("outage_rate", "Outage", "AIRed!82", 0.0, 1.0, r"$\downarrow$"),
        ("task_completion", "Task", "AIBlue!78", 0.0, 1.0, r"$\uparrow$"),
        ("auv_collision_count", "AUV collisions", "AIAmber!82!black", 0.0, None, r"$\downarrow$"),
    ]
    metric_values = {
        key: [float(row[key]) for _, rows in seed_rows for row in rows]
        for key, *_ in metrics
    }
    collision_hi = float(np.quantile(metric_values["auv_collision_count"], 0.95))
    metrics[-1] = ("auv_collision_count", "AUV collisions", "AIAmber!82!black", 0.0, max(collision_hi * 1.12, 1.0), r"$\downarrow$")

    def x_coord(value: float, xmin: float, xmax: float, width: float) -> float:
        value = min(max(value, xmin), xmax)
        return (value - xmin) / max(xmax - xmin, 1e-9) * width

    lines = _ai_tikz_begin()
    panel_w, panel_h = 4.35, 2.35
    positions = [(0.0, 3.05), (5.25, 3.05), (0.0, 0.0), (5.25, 0.0)]
    for mi, (key, title, color, xmin, xmax, arrow) in enumerate(metrics):
        x0, y0 = positions[mi]
        lines.append(fr"\begin{{scope}}[shift={{({x0:.2f},0)}}]")
        lines[-1] = fr"\begin{{scope}}[shift={{({x0:.2f},{y0:.2f})}}]"
        lines.append(fr"\draw[aiPanel] (0,0) rectangle ({panel_w:.2f},{panel_h:.2f});")
        lines.append(fr"\node[anchor=south west,font=\bfseries\scriptsize,text=AINavy] at (0.02,{panel_h + 0.14:.2f}) {{{chr(65 + mi)} {title} {arrow}}};")
        for val in _nice_ticks(xmin, xmax):
            x = x_coord(val, xmin, xmax, panel_w)
            label = f"{val:.1f}" if xmax <= 1.0 else f"{val:.0f}"
            lines.append(fr"\draw[aiGrid] ({x:.2f},0) -- ({x:.2f},{panel_h:.2f});")
            if mi >= 2:
                lines.append(fr"\node[below] at ({x:.2f},-0.03) {{{label}}};")
        row_gap = panel_h / (len(seed_rows) + 1)
        for si, (seed, rows) in enumerate(seed_rows):
            values = np.asarray([float(row[key]) for row in rows], dtype=float)
            q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            mean_v = float(values.mean())
            y = panel_h - (si + 1) * row_gap
            x05, x25, x50, x75, x95, xm = [
                x_coord(v, xmin, xmax, panel_w) for v in [q05, q25, q50, q75, q95, mean_v]
            ]
            lines.append(fr"\node[anchor=east] at (-0.08,{y:.2f}) {{{seed}}};")
            lines.append(fr"\draw[{color},line width=0.85pt] ({x05:.2f},{y:.2f}) -- ({x95:.2f},{y:.2f});")
            lines.append(fr"\draw[rounded corners=1pt,fill={color},draw={color},opacity=0.20] ({x25:.2f},{y-0.14:.2f}) rectangle ({x75:.2f},{y+0.14:.2f});")
            lines.append(fr"\draw[AINavy,line width=0.85pt] ({x50:.2f},{y-0.18:.2f}) -- ({x50:.2f},{y+0.18:.2f});")
            lines.append(fr"\node[circle,fill={color},inner sep=1.15pt] at ({xm:.2f},{y:.2f}) {{}};")
        lines.append(r"\end{scope}")
    lines.extend(
        [
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "neural_checkpoint_bars_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_scenario_heatmap(rows: list[dict[str, str]]) -> None:
    keep = ["independent_greedy", "communication_aware", "haco_safemarl_trained"]
    header_labels = {
        "independent_greedy": "Indep.",
        "communication_aware": "Comm.",
        "haco_safemarl_trained": "HACo",
    }
    by_pair = {(r["scenario"], r["policy"]): r for r in rows}
    cell_w, cell_h = 2.15, 0.58
    lines = _ai_tikz_begin()
    for j, policy in enumerate(keep):
        x = 2.7 + j * cell_w
        lines.append(fr"\node[align=center,font=\bfseries,text=AINavy] at ({x + cell_w/2:.2f},0.35) {{{header_labels[policy]}}};")
    for i, scenario in enumerate(SCENARIO_ORDER):
        y = -i * cell_h
        lines.append(fr"\node[anchor=east] at (2.55,{y - 0.32:.2f}) {{{SCENARIO_LABELS[scenario]}}};")
        for j, policy in enumerate(keep):
            r = by_pair[(scenario, policy)]
            value = float(r["worst_agent_packet_mean"])
            pct = int(12 + 78 * value)
            x = 2.7 + j * cell_w
            lines.append(fr"\fill[AITeal!{pct}!white] ({x:.2f},{y - cell_h:.2f}) rectangle ({x + cell_w:.2f},{y:.2f});")
            lines.append(fr"\draw[white,line width=0.6pt] ({x:.2f},{y - cell_h:.2f}) rectangle ({x + cell_w:.2f},{y:.2f});")
            text_color = "white" if value > 0.62 else "black"
            lines.append(fr"\node[text={text_color}] at ({x + cell_w/2:.2f},{y - cell_h/2:.2f}) {{{value:.2f}}};")
    lines.extend(
        [
            r"\node[anchor=west] at (2.7,-3.55) {worst-agent packet delivery; darker means better weakest-link communication};",
            r"\draw[fill=AITeal!15!white,draw=AIGray!45] (2.7,-3.95) rectangle (3.25,-3.72);",
            r"\node[anchor=west] at (3.32,-3.84) {low};",
            r"\draw[fill=AITeal!85!white,draw=AIGray!45] (4.15,-3.95) rectangle (4.70,-3.72);",
            r"\node[anchor=west] at (4.77,-3.84) {high};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "scenario_heatmap_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_task_safety_scatter(rows: list[dict[str, str]]) -> None:
    summary = _summary_by_policy(rows)
    width, height = 7.4, 4.2
    max_coll = max(summary[p]["auv"] for p in POLICY_ORDER if p in summary)
    lines = _ai_tikz_begin("font=\\scriptsize")
    lines.extend(
        [
            fr"\draw[-{{Latex[length=1.8mm]}},draw=AINavy!72] (0,0) -- ({width + 0.35:.2f},0) node[right] {{task completion}};",
            fr"\draw[-{{Latex[length=1.8mm]}},draw=AINavy!72] (0,0) -- (0,{height + 0.35:.2f}) node[above] {{worst packet}};",
        ]
    )
    for frac, label in [(0.0, "0"), (0.5, ".5"), (1.0, "1")]:
        x = frac * width
        y = frac * height
        lines.append(fr"\draw[aiGrid] ({x:.2f},0) -- ({x:.2f},{height:.2f});")
        lines.append(fr"\node[below] at ({x:.2f},0) {{{label}}};")
        lines.append(fr"\draw[aiGrid] (0,{y:.2f}) -- ({width:.2f},{y:.2f});")
        lines.append(fr"\node[anchor=east] at (-0.05,{y:.2f}) {{{label}}};")
    legend = []
    for idx, policy in enumerate(POLICY_ORDER, 1):
        vals = summary[policy]
        x = vals["task"] * width
        y = vals["worst"] * height
        radius = 0.10 + 0.23 * np.sqrt(vals["auv"] / max(max_coll, 1e-9))
        label = POLICY_LABELS[policy]
        color = METHOD_COLORS.get(label, "gray!50")
        lines.append(fr"\draw[fill={color},draw=AINavy!60,opacity=0.78] ({x:.2f},{y:.2f}) circle ({radius:.2f});")
        lines.append(fr"\node[font=\tiny] at ({x:.2f},{y:.2f}) {{{idx}}};")
        legend.append(f"{idx}: {label}")
    legend_text = (
        r"\begin{tabular}{lll}"
        + f"{legend[0]} & {legend[1]} & {legend[2]} "
        + r"\\ "
        + f"{legend[3]} & {legend[4]} & {legend[5]}"
        + r"\end{tabular}"
    )
    lines.extend(
        [
            fr"\node[draw=AINavy!32,fill=white,rounded corners=1pt,anchor=north west,align=left] at (0,-0.45) {{{legend_text}}};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "task_safety_scatter_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def _episode_rows_for_mappo(seed_dir: Path) -> list[dict[str, str]]:
    path = seed_dir / "episode_metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["policy"] == "haco_safemarl_mappo"]


def generate_neural_outage_boxplot() -> None:
    series = []
    for seed_dir in _mappo_seed_dirs():
        rows = _episode_rows_for_mappo(seed_dir)
        if not rows:
            continue
        values = np.asarray([float(r["outage_rate"]) for r in rows], dtype=float)
        series.append((seed_dir.name.replace("seed_", "")[-2:], values))
    if not series:
        return
    width, row_gap = 8.2, 0.7
    lines = _ai_tikz_begin()
    lines.append(fr"\draw[-{{Latex[length=1.8mm]}},draw=AINavy!72] (0,0.15) -- ({width + 0.35:.2f},0.15) node[right] {{episode outage rate}};")
    for frac, label in [(0.0, "0"), (0.25, ".25"), (0.5, ".5"), (0.75, ".75"), (1.0, "1")]:
        x = frac * width
        lines.append(fr"\draw[aiGrid] ({x:.2f},0.05) -- ({x:.2f},{row_gap * len(series) + 0.45:.2f});")
        lines.append(fr"\node[below] at ({x:.2f},0.08) {{{label}}};")
    for i, (seed, values) in enumerate(series):
        y = row_gap * (i + 1)
        q0, q1, med, q3, q4 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        mean_v = float(values.mean())
        x0, x1, xm, x3, x4 = [v * width for v in [q0, q1, med, q3, q4]]
        xmean = mean_v * width
        lines.append(fr"\node[anchor=east] at (-0.08,{y:.2f}) {{seed {seed}}};")
        lines.append(fr"\draw[AIRed!78,line width=0.85pt] ({x0:.2f},{y:.2f}) -- ({x4:.2f},{y:.2f});")
        lines.append(fr"\draw[rounded corners=1pt,fill=AIRed!22,draw=AIRed!75] ({x1:.2f},{y-0.16:.2f}) rectangle ({x3:.2f},{y+0.16:.2f});")
        lines.append(fr"\draw[AINavy,line width=0.85pt] ({xm:.2f},{y-0.20:.2f}) -- ({xm:.2f},{y+0.20:.2f});")
        lines.append(fr"\node[circle,fill=AINavy,inner sep=1.2pt] at ({xmean:.2f},{y:.2f}) {{}};")
        lines.append(fr"\node[anchor=west] at ({x4 + 0.08:.2f},{y:.2f}) {{$\mu$={mean_v:.2f}}};")
    lines.extend(
        [
            r"\node[anchor=west,align=left,text width=8.2cm] at (0,-0.65) {Boxes show interquartile ranges over evaluation episodes; whiskers show 5th--95th percentiles and dots mark means.};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    (FIG_DIR / "neural_outage_boxplot_tikz.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    generate_main_table(rows)
    generate_scenario_table(rows)
    generate_tradeoff_figure(rows)
    generate_training_curve_figure()
    generate_ablation_figure(rows)
    generate_emergency_sequence_figure()
    generate_preference_table()
    generate_neural_multiseed_table()
    generate_bc_loss_figure()
    generate_neural_checkpoint_bars()
    generate_scenario_heatmap(rows)
    generate_task_safety_scatter(rows)
    generate_neural_outage_boxplot()


if __name__ == "__main__":
    main()
