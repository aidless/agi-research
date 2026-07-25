# Acrobot-v1 Phase 2 (2026-07-25) - threshold-passing fixed, eval AUROC noisy

## Pipeline fix shipped
- envs.py: --percentile default 30 -> 10 + auto-detect n_actions in monitor
- monitor.py: FailureDataset.__init__ accepts threshold arg
  + uses ep.total_reward < threshold when threshold != None
- monitor.py: train_monitor() takes threshold kwarg
- monitor.py: docstring + def train_monitor duplicates fixed
- classic_phase2.py (was lunarlander_phase2.py): generalised env-agnostic
  runner with --env flag and --threshold-floor

## Acrobot-v1 result (PIPE FIXED, eval noisy)
- PPO 256K, p10=-117, threshold-floor=-1e9
- Train labels: 7 failures < -117, 193 successes
- Train Monitor AUROC = 0.984
- Eval failures = 2 / 100
- **Eval AUROC = 0.42 (BELOW chance)** — but with only 2 positives in eval,
  this is statistically meaningless (noise).

## Interpretation
- Pipeline is now working end-to-end. Threshold passes from Phase 2
  pipeline -> train_monitor -> FailureDataset -> labels.
- Acrobot's bimodal distribution (mean -82) gives natural failure/success
  labels at p10=-117.
- Train Monitor learns (0.984 AUROC) but doesn't generalize on 2-eval positives.
- The 0.42 eval AUROC is below chance -- needs either more eval episodes
  (50 -> 500) or more seeds.

## Implications
- H1 on LunarLander-v3 still the primary result (Eval AUROC 0.98)
- Acrobot is harder because PPO at 256K doesn't fully solve it; threshold
  -117 is close to mean -82; label distribution skewed.
- For paper, the LunarLander-v3 result is the cleanest H1 demonstration.
  Acrobot can be reported as cross-env exploratory.

## Code shipped
- monitor.py: dynamic threshold passed from Phase 2 -> FailureDataset
- classic_phase2.py: --env flag (env-agnostic Phase 2 runner)
- 4 corruption fixes (UTF-8 em-dash byte sequences, def duplicates)
