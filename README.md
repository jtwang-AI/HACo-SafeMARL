# HACo-Safe

HACo-Safe 是一个面向 **USV-AUV 异构协同路径规划** 的可复现实验仓库。主基准评估 CEM 调参的可解释控制器与规则安全盾；图注意力 MAPPO 是单独的蒸馏和机制诊断阶段，不代表主结果或部署策略。

本项目研究无人水面艇（USV）与自主水下航行器（AUV）在水声通信丢包、动态水面交通、水下障碍、AUV 应急状态和安全约束下的协同路径规划问题。仓库仅发布代码、实验数据、生成图表和实验结果，不包含论文 PDF、LaTeX 源文件或投稿材料。

## 仓库结构

- `code/`：仿真环境、策略、训练、评估和结果生成脚本。
- `code/haco/`：核心环境、控制器、实验工具和 PyTorch 策略实现。
- `scripts/`：GPU 与远程实验运行脚本。
- `results/`：原始 episode 表、聚合指标、训练历史和运行配置。
- `figures/`：生成的非 LaTeX 图表文件，例如 PNG。
- `EXPERIMENT_MANIFEST.md`：已完成实验、复现命令和结果说明。

## 环境安装

安装最小 Python 依赖：

```bash
pip install -r requirements.txt
```

如果运行神经 MAPPO 实验，需要安装与本机 GPU 驱动匹配的 CUDA 版 PyTorch。

## 复现主要实验

运行 warm-start 搜索：

```bash
PYTHONPATH=code python3 code/train_cem_warmstart.py \
  --generations 8 --population 24 --episodes 3 \
  --out results/cem_local
```

运行主评估：

```bash
PYTHONPATH=code python3 code/evaluate_policy_params.py \
  --policy results/cem_local/best_policy.json \
  --episodes 40 \
  --out results/final_eval_local40
```

运行 review-driven MAPPO 诊断消融实验（图注意力、无声学、无安全盾）：

```bash
./scripts/run_remote_review_experiments_parallel.sh
```

该脚本会生成：

- `results/review_marl_baselines/*/seed_*/`
- `results/review_marl_baselines_summary.json`

运行无图注意力的均值聚合 MAPPO 基线（4 个 GPU/4 个随机种子）：

```bash
./scripts/run_no_gat_baseline_parallel.sh
```

四种学习变体使用同一预算：100 个 behavior-cloning epochs、100 个 PPO updates、每个随机种子 350 个 held-out episodes。

四种子均值聚合（无图注意力）结果为：success `0.00±0.00`、task `0.08±0.02`、outage `0.30±0.05`、worst packet `0.71±0.04`、AUV penetration count `3.8±0.3`。同批次图注意力基线为 `0.00±0.00 / 0.04±0.01 / 0.26±0.05 / 0.74±0.04 / 2.6±0.4`；均值聚合任务进度略高，但通信和侵入指标略差，不能据此声称图注意力全面优越。

对同一个已训练的 graph-attention actor 分别启用/关闭安全盾，避免把“无盾训练”与“同一策略的原始动作”混为一谈：

```bash
./scripts/run_checkpoint_shield_replay_parallel.sh
PYTHONPATH=code python3 code/summarize_checkpoint_shield_replay.py \
  --root results/review_same_actor_shield_replay
```

四种子汇总：启用安全盾时 outage/worst packet/AUV penetration count 为 `0.26±0.07 / 0.74±0.04 / 2.3±0.4`；同一 actor 关闭安全盾后为 `0.56±0.10 / 0.52±0.11 / 65.7±5.6`。这里的 collision/penetration count 是逐时间步累加的障碍侵入次数，不是唯一物理碰撞事件数。

## 审稿补充实验

生成配对置信区间/随机化检验、安全盾依赖、任务成功分解、TDMA/陈旧信息，以及会遇与动力学/海流敏感性结果：

```bash
PYTHONPATH=code python3 code/review_revision_experiments.py \
  --policy results/cem_local/best_policy.json \
  --main-episodes results/final_eval_local40/episode_metrics.csv \
  --episodes 40 \
  --out results/reviewer_revision \
  --table-dir tables
```

输出包括原始 episode CSV、JSON 汇总和以下论文表格：

- `review_paired_statistics.tex`
- `review_shield_scenario.tex`
- `review_mission_decomposition.tex`
- `review_delay_sensitivity.tex`
- `review_robustness_sensitivity.tex`

声学与动力学均为规划级敏感性模型：TDMA 测试不是调制解调器/MAC 验证，first-order lag/current 测试也不是 3-DOF 或 6-DOF 水动力验证。

## 重新生成实验汇总

```bash
PYTHONPATH=code python3 code/summarize_remote_review_experiments.py \
  --root results/review_marl_baselines
```
