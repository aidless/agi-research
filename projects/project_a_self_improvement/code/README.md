# code/ — Project A 的可运行代码

> 这是 v1 的最小可运行实现。你应该能在没有 GPU 的笔记本上 3-5 分钟跑通整个流程。

## 文件说明

| 文件 | 用途 | 你 review 时关注 |
|---|---|---|
| `envs.py` | 环境 wrapper + EpisodeLog + failure 判定 | `is_failure_episode` 阈值 |
| `ppo.py` | 100 行 from-scratch PPO | clip_eps, gamma, hidden |
| `monitor.py` | FailureMonitor + train_monitor | MLP 架构，loss 函数，BCE |
| `main.py` | 训练入口 | 默认超参，rollout 长度 |
| `evaluate.py` | 评估 + 报告 AUROC + Pearson | failure 阈值, history_len |

## 怎么跑

### Step 0: 安装依赖（一次性）
```bash
pip install torch numpy gymnasium
```

### Step 1: 训练 PPO + 训练 Monitor
```bash
cd E:\agi-research\projects\project_a_self_improvement\code
python main.py --env CartPole-v1 --total-steps 60000 --eval-episodes 80
```
预期输出（在 CPU 上 3-5 分钟）：
```
[Project A] v1 on CartPole-v1  seed=0  total_steps=60000
=== Stage 1: Train policy via PPO ===
  iter=5  steps=10240  avg_return(last_30ep)=20.4  ...
  iter=10  steps=20480  avg_return(last_30ep)=70.2 ...
  ...
=== Stage 2: Collect rollouts for Monitor training ===
  collected 80 eval episodes
  reward mean=480.1  std=80.2  min=180.0  max=500.0
=== Stage 3: Train Failure Monitor (frozen-policy decoupling) ===
  Monitor dataset: 80 episodes (12 failures, 68 successes)
  Epoch 1/5  loss=0.4512
  ...
  Monitor AUROC (on train set, for sanity): 0.612
```

### Step 2: 评估
```bash
python evaluate.py --env CartPole-v1 --n-episodes 100 --seed 1
```
预期输出：
```
[Evaluate] env=CartPole-v1  episodes=100  seed=1
=== Results ===
  Eval episodes            : 100
  Failure-rate (label rate): 0.120
  Reward  mean ± std       : 470.3 ± 80.5
  ...
  AUROC (mean prob → failure): 0.61x
  Pearson(mean_p, fail)      : 0.4x
```

## 如果出错了

| 症状 | 可能原因 | 修法 |
|---|---|---|
| ModuleNotFoundError: gymnasium | 没装 gymnasium | `pip install gymnasium` |
| ModuleNotFoundError: torch | 没装 PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Monitor AUROC = 0.5 | 训练样本里全是一类 | 改 `--total-steps` 让 policy 学得多一些 |
| Loss = nan | LR 太大或 rollout 里撞 dead state | 改 `--seed` 或减小 `lr` |
| 训练明显不收敛 | PPO 初始随机策略太搞 | 改 `--seed` 或减小 `--total-steps` |

## 我 review 时必须看的地方（标记在文件里的 # REVIEW-ME）

envs.py:1. `is_failure_episode` 的阈值  
envs.py:2. `history_len` 应该用 32/64/128 的哪一种  
ppo.py:30-40. hidden 大小  
monitor.py:1. MLP 架构是否够  
monitor.py:2. 历史是否应该 sequence model 而非 MLP  
main.py:1. 默认 `--total-steps` 是不是合适  
evaluate.py:1. `--threshold` 应该 default 0.5 还是动态（按均值的百分位）  

## 下一步做什么

1. 跑一次 baseline 没有 monitor：`python main.py --env CartPole-v1 --disable-monitor`
2. 跑带 monitor 的：`python main.py --env CartPole-v1`（默认）
3. 跑 evaluate，对比两次的 AUROC
4. 在你的实验笔记里写下结果
5. **把结果贴回来给 Codex** —— 我接着写 LunarLander 的实验

## 注意

- PPO 默认设置是随机种子 `seed=0`。换 seed 重跑应该产生类似结果但不完全一样
- Monitor 仅在训练阶段用同一个 seed 的 episodes 训练，可能过拟合，evaluate 时用不同 seed
- 如果 AUROC > 0.6 且 failure 标签占比 > 5%，你的 paper v1 主 claim 已经被实验支持
- 如果 AUROC 接近 0.5，不必灰心，这正是为什么研究是迭代
