# HACo-SafeMARL

HACo-SafeMARL 是一个面向 **USV-AUV 异构协同路径规划** 的可复现实验仓库，核心方向是声学通信感知的安全多智能体强化学习。

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

运行 review-driven MAPPO 诊断消融实验：

```bash
./scripts/run_remote_review_experiments_parallel.sh
```

该脚本会生成：

- `results/review_marl_baselines/*/seed_*/`
- `results/review_marl_baselines_summary.json`

## 重新生成实验汇总

```bash
PYTHONPATH=code python3 code/summarize_remote_review_experiments.py \
  --root results/review_marl_baselines
```
