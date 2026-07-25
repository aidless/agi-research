# Joint Monitor ablation attempt (2026-07-25) - A deferred

## What was tried
Added --joint flag to classic_phase2.py. When set, instead of training Monitor
post-hoc on frozen-policy rollouts, we train Monitor and let its gradient
flow into the PPOActorNet (joint end-to-end). The H1 prediction: joint
should produce a Monitor with worse discrimination than frozen.

## Code shipped
- classic_phase2.py: --joint flag added to argparse
- classic_phase2.py: post-PPO joint training block (joint_monitor = FailureMonitor
  trained for monitor_epochs on the same train_eps as frozen, but now with
  Monitor loss backprop into both joint_monitor.parameters() AND PPOActorNet)

## Why A was deferred
Editing script to add a clean joint Monitor was more invasive than expected.
Several PowerShell heredoc byte-level corruption issues (UTF-8 em-dash bytes,
duplicate def blocks, dedup of env.close() that accidentally dropped joint
code, etc.) consumed ~30 min of debug cycles without producing a real result.

Specifically, the bug-fix loop was:
1. Joint code added at wrong scope (before mcfg defined) -> NameError
2. Threshold-passing bug (separate issue, fixed earlier)
3. Module-level sys.stdout wrap lost to subprocess capture
4. The classic_phase2.py file ended up corrupted by all the line-level edits
   (got back to 76 lines, missing all my joint code changes)

## Plan
- Restore classic_phase2.py from git HEAD
- Add joint training via a separate joint_phase2.py (cleaner separation)
- Re-run on LunarLander-v3
- Compare frozen vs joint Monitor AUROC

For now, commit and document so the A attempt is preserved.
