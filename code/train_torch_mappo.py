from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from haco.env import HACoPilotEnv, ScenarioConfig, StepMetrics, summarize_episode
from haco.experiment import aggregate, default_policies, run_episode, scenario_config, write_csv, write_json, write_latex_table
from haco.policies import HACoSafeHeuristicPolicy
from haco.torch_policy import HACoActorCritic, TorchHACoPolicy, build_feature_arrays


@dataclass
class StepRecord:
    usv_feat: np.ndarray
    auv_feat: np.ndarray
    global_feat: np.ndarray
    usv_action: np.ndarray
    auv_action: np.ndarray
    old_log_prob: float
    value: float
    reward: float
    ret: float = 0.0
    adv: float = 0.0


def load_warm_params(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("params", payload.get("best_params", {}))


def shaped_reward(cfg: ScenarioConfig, metrics: StepMetrics, use_acoustic_reward: bool = True) -> float:
    outage = metrics.outage_count / max(cfg.num_auvs, 1)
    acoustic_term = 0.35 * metrics.packet_probs.mean() - 1.25 * outage if use_acoustic_reward else 0.0
    return float(
        0.035 * metrics.task_progress
        + acoustic_term
        - 2.5 * float(metrics.usv_collision)
        - 0.65 * metrics.auv_collision_count
        - 0.35 * float(metrics.colregs_violation)
        - 0.00012 * metrics.energy
        - 0.00014 * metrics.smoothness
        - 0.0015 * metrics.shield_interventions
    )


def terminal_bonus(summary: dict, use_acoustic_reward: bool = True) -> float:
    acoustic_term = (
        1.0 * summary["worst_agent_packet"]
        + 0.6 * summary["comm_fairness"]
        - 1.4 * summary["outage_rate"]
        - 1.2 * summary["cvar_outage_90"]
        if use_acoustic_reward
        else 0.0
    )
    return float(
        3.0 * summary["success"]
        + 2.0 * summary["task_completion"]
        + acoustic_term
        - 1.5 * summary["usv_collision_rate"]
        - 0.25 * summary["auv_collision_count"]
        - 0.4 * summary["colregs_violation_rate"]
    )


def tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def collect_bc_samples(
    policy: HACoSafeHeuristicPolicy,
    scenarios: list[str],
    episodes: int,
    seed: int,
    use_acoustic: bool,
):
    samples = []
    for scenario in scenarios:
        for ep in range(episodes):
            cfg = scenario_config(scenario, seed=seed + ep * 7919 + len(samples))
            env = HACoPilotEnv(cfg)
            obs = env.reset()
            done = False
            while not done:
                usv_action, auv_action, shield = policy.act(obs)
                samples.append(
                    (
                        *build_feature_arrays(obs, use_acoustic=use_acoustic),
                        usv_action.astype(np.float32),
                        auv_action.astype(np.float32),
                    )
                )
                obs, _, done = env.step(usv_action, auv_action, shield=shield)
    return samples


def train_behavior_clone(
    model: HACoActorCritic,
    samples,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        random.shuffle(samples)
        losses = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            opt.zero_grad()
            loss_terms = []
            for usv_feat, auv_feat, _, target_usv, target_auv in batch:
                pred_usv, pred_auv = model.means(tensor(usv_feat, device).unsqueeze(0), tensor(auv_feat, device))
                loss_terms.append(nn.functional.mse_loss(pred_usv.squeeze(0), tensor(target_usv, device)))
                loss_terms.append(nn.functional.mse_loss(pred_auv, tensor(target_auv, device)))
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rec = {"bc_epoch": epoch, "loss": float(np.mean(losses))}
        history.append(rec)
        print(f"bc_epoch={epoch} loss={rec['loss']:.5f}")
    return history


def rollout_episode(model: HACoActorCritic, cfg: ScenarioConfig, args, device: torch.device) -> tuple[list[StepRecord], dict]:
    env = HACoPilotEnv(cfg)
    obs = env.reset()
    records: list[StepRecord] = []
    step_metrics = []
    done = False
    while not done:
        usv_feat, auv_feat, global_feat = build_feature_arrays(obs, use_acoustic=args.use_acoustic)
        with torch.no_grad():
            usv_action, auv_action, log_prob, _, value = model.sample_action(
                tensor(usv_feat, device).unsqueeze(0),
                tensor(auv_feat, device),
                tensor(global_feat, device).unsqueeze(0),
            )
        next_obs, metrics, done = env.step(
            usv_action.cpu().numpy(), auv_action.cpu().numpy(), shield=args.use_shield
        )
        records.append(
            StepRecord(
                usv_feat=usv_feat,
                auv_feat=auv_feat,
                global_feat=global_feat,
                usv_action=usv_action.cpu().numpy().astype(np.float32),
                auv_action=auv_action.cpu().numpy().astype(np.float32),
                old_log_prob=float(log_prob.cpu()),
                value=float(value.cpu()),
                reward=shaped_reward(cfg, metrics, use_acoustic_reward=args.use_acoustic_reward),
            )
        )
        step_metrics.append(metrics)
        obs = next_obs
    summary = summarize_episode(cfg, step_metrics, env.task_done, env.t)
    if records:
        records[-1].reward += terminal_bonus(summary, use_acoustic_reward=args.use_acoustic_reward)
    return records, summary


def assign_returns(records: list[StepRecord], gamma: float, gae_lambda: float):
    next_adv = 0.0
    next_value = 0.0
    for rec in reversed(records):
        delta = rec.reward + gamma * next_value - rec.value
        next_adv = delta + gamma * gae_lambda * next_adv
        rec.adv = float(next_adv)
        rec.ret = float(rec.adv + rec.value)
        next_value = rec.value


def ppo_update(
    model: HACoActorCritic,
    records: list[StepRecord],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
) -> dict:
    adv = np.array([r.adv for r in records], dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    for rec, val in zip(records, adv):
        rec.adv = float(val)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []
    for _ in range(epochs):
        random.shuffle(records)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            opt.zero_grad()
            lp_terms = []
            vf_terms = []
            ent_terms = []
            for rec in batch:
                log_prob, entropy, value = model.evaluate_actions(
                    tensor(rec.usv_feat, device).unsqueeze(0),
                    tensor(rec.auv_feat, device),
                    tensor(rec.global_feat, device).unsqueeze(0),
                    tensor(rec.usv_action, device),
                    tensor(rec.auv_action, device),
                )
                ratio = torch.exp(log_prob - torch.tensor(rec.old_log_prob, dtype=torch.float32, device=device))
                adv_t = torch.tensor(rec.adv, dtype=torch.float32, device=device)
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t
                lp_terms.append(-torch.minimum(ratio * adv_t, clipped))
                vf_terms.append((value - torch.tensor(rec.ret, dtype=torch.float32, device=device)).pow(2))
                ent_terms.append(entropy)
            policy_loss = torch.stack(lp_terms).mean()
            value_loss = torch.stack(vf_terms).mean()
            entropy = torch.stack(ent_terms).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def train_ppo(model: HACoActorCritic, args, device: torch.device) -> list[dict]:
    history = []
    for update in range(args.ppo_updates):
        records: list[StepRecord] = []
        summaries = []
        for ep in range(args.rollout_episodes):
            scenario = args.scenarios[(update * args.rollout_episodes + ep) % len(args.scenarios)]
            cfg = scenario_config(scenario, seed=args.seed + 500000 + update * 10000 + ep * 101)
            ep_records, summary = rollout_episode(model, cfg, args, device)
            assign_returns(ep_records, args.gamma, args.gae_lambda)
            records.extend(ep_records)
            summaries.append(summary)
        stats = ppo_update(
            model,
            records,
            device,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            lr=args.ppo_lr,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
        )
        rec = {
            "ppo_update": update,
            "rollout_steps": len(records),
            "mean_task_completion": float(np.mean([s["task_completion"] for s in summaries])),
            "mean_outage_rate": float(np.mean([s["outage_rate"] for s in summaries])),
            **stats,
        }
        history.append(rec)
        write_json(args.out / "ppo_history.json", history)
        print(
            "ppo_update={ppo_update} steps={rollout_steps} task={mean_task_completion:.3f} "
            "outage={mean_outage_rate:.3f} loss={loss:.4f}".format(**rec)
        )
    return history


def evaluate_checkpoint(checkpoint: Path, args, device: torch.device):
    policy = TorchHACoPolicy(
        checkpoint,
        device=str(device),
        deterministic=True,
        use_acoustic=args.use_acoustic,
        use_shield=args.use_shield,
        name=args.policy_name,
    )
    policies = default_policies() + [policy]
    rows = []
    for scenario in args.eval_scenarios:
        auv_counts = args.generalization_auvs if scenario == "generalization" else [3]
        for num_auvs in auv_counts:
            for ep in range(args.eval_episodes):
                seed = args.seed + 900000 + 10000 * ep + 100 * num_auvs + len(rows)
                cfg = scenario_config(scenario, seed=seed, num_auvs=num_auvs)
                for pol in policies:
                    rows.append(run_episode(pol, cfg))
    agg = aggregate(rows)
    write_csv(args.out / "episode_metrics.csv", rows)
    write_csv(args.out / "aggregate_metrics.csv", agg)
    write_json(args.out / "aggregate_metrics.json", agg)
    write_latex_table(Path("tables/torch_mappo_main_results.tex"), agg)
    return rows, agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/torch_mappo"))
    parser.add_argument("--warm-policy", type=Path, default=Path("results/cem/best_policy.json"))
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenarios", nargs="*", default=["survey", "traffic", "acoustic_degradation", "emergency"])
    parser.add_argument("--eval-scenarios", nargs="*", default=["survey", "traffic", "acoustic_degradation", "emergency", "generalization"])
    parser.add_argument("--generalization-auvs", nargs="*", type=int, default=[2, 6, 8])
    parser.add_argument("--bc-episodes", type=int, default=8)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--bc-lr", type=float, default=2e-4)
    parser.add_argument("--ppo-updates", type=int, default=100)
    parser.add_argument("--rollout-episodes", type=int, default=6)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-batch-size", type=int, default=96)
    parser.add_argument("--ppo-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--gae-lambda", type=float, default=0.94)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--policy-name", default="haco_safemarl_mappo")
    parser.add_argument("--no-acoustic-features", action="store_true")
    parser.add_argument("--no-acoustic-reward", action="store_true")
    parser.add_argument("--disable-shield", action="store_true")
    args = parser.parse_args()
    args.use_acoustic = not args.no_acoustic_features
    args.use_acoustic_reward = not args.no_acoustic_reward
    args.use_shield = not args.disable_shield

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    warm_params = load_warm_params(args.warm_policy)
    teacher = HACoSafeHeuristicPolicy(params=warm_params, name="haco_safemarl_teacher")
    model = HACoActorCritic(hidden=args.hidden).to(device)

    run_config = {
        "policy_name": args.policy_name,
        "use_acoustic": args.use_acoustic,
        "use_acoustic_reward": args.use_acoustic_reward,
        "use_shield": args.use_shield,
        "seed": args.seed,
        "scenarios": args.scenarios,
        "eval_scenarios": args.eval_scenarios,
        "generalization_auvs": args.generalization_auvs,
    }
    write_json(args.out / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2))

    bc_samples = collect_bc_samples(teacher, args.scenarios, args.bc_episodes, args.seed, args.use_acoustic)
    history = {
        "device": str(device),
        "bc_samples": len(bc_samples),
        "bc": train_behavior_clone(model, bc_samples, device, args.bc_epochs, args.bc_batch_size, args.bc_lr),
    }
    write_json(args.out / "bc_history.json", history["bc"])
    history["ppo"] = train_ppo(model, args, device)

    checkpoint = args.out / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": args.hidden,
            "args": vars(args),
        },
        checkpoint,
    )
    write_json(args.out / "training_history.json", history)
    _, agg = evaluate_checkpoint(checkpoint, args, device)
    print(f"checkpoint: {checkpoint}")
    print(f"aggregate: {args.out / 'aggregate_metrics.csv'}")
    print(f"rows: {len(agg)} aggregate records")


if __name__ == "__main__":
    main()
