# 第一次端到端 smoke test

- 日期: 2026-07-25
- 项目: Project A (Self-Improvement Loop) v1
- 环境: CartPole-v1
- seed: 0 (训练) / 999 (评估)
- 总训练步: 8192 (非常短! 默认应该是 60000)
- 训练轮数: 4
- 计算力: CPU only, RTX 4090 都没用

## 这个 run 想验证什么
- 代码是否能完全跑通 (无 import 错误 / 维度不匹配)
- Decoupled Monitor 是否能 learn something
- evaluate.py 是否能产生 actionable metrics

## 实际结果

| 指标 | 数值 | 解释 |
|---|---|---|
| Eval episodes | 50 | 不同的 seed (999) |
| Failure rate | 0.600 | 60% 的 episode 是 failure, 因为 policy 还很弱 (mean return 85) |
| Reward mean ± std | 84.72 ± 48.48 | policy 还在学, 不是 final 的 |
| Monitor mean p mean ± std | 0.489 ± 0.001 | Monitor 对几乎所有 episode 输出 0.5 -- 它对 weak policy 的 future 没强 confidence |
| **AUROC (mean p → failure)** | **0.705** | **Monitor 是 predicting 有意义的信号** |
| AUROC (final p → failure) | 0.653 | 最后一步的概率也 OK |
| Pearson(mean p, reward) | -0.334 | Monitor probability 越高, reward 越低 |
| **Pearson(mean p, fail)** | **0.361** | Monitor probability 越高, failure 概率越大 |

## 结论

1. **技术结论**: Decoupled Monitor 在 8K 步训练下已经展示明显信号. 在 60K 步 + 多 env 的 paper main result 里, 信号应该更强.

2. **路线结论**: 不需要 GPU 就能跑 main result 的 v1 experiments. 这是 Project A 的低成本试错机制确立.

3. **下一步**:
   - 把 --total-steps 提到 60000 重跑 baseline
   - 加 LunarLander / Acrobot
   - 加 ablation (joint-trained critic)
   - 跑 ≥3 seeds, report mean ± std

## 输出文件

- code/checkpoints/policy.pt
- code/checkpoints/monitor.pt
- code/checkpoints/run_log.json
- code/checkpoints/eval_log.json
