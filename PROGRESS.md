# PROGRESS.md - Current Status

> **Codex has no cross-session memory**. Every new session should start with:
> 1. Open this file
> 2. Codex will read it + AGENTS.md + ROADMAP.md
> 3. Then auto-resume

---

## Last session state (2026-07-25)

**Major progress**:
- [x] Workspace skeleton complete (31 files, ~200KB)
- [x] 3 root docs + 1 literature review + Project A complete code
- [x] **Smoke test PASSED**: Monitor AUROC = 0.71, Pearson = 0.36 on CartPole CPU only
- [x] Grant application drafts (HF, Google, Lambda)
- [x] Twitter / Reddit / Discord / Email drafts
- [x] 3 Decision Records

**Blocking decisions for you**:
1. **Decision 0001 (PhD vs Independent)**: open, due 2026-09-30
2. **Smoke test ran with WEAK policy (8K steps)** - you need to decide:
   - Run full 60K step training (locks paper baseline), OR
   - Review code first then run
3. **paper_draft_v0.md not yet written** - waiting for your code review

---

## Work Board

`
Year 0 Q1 (month 1-3) - Foundation
  Skeleton (Path 1)
  - [x] Workspace skeleton
  - [x] README / ROADMAP / PROGRESS
  - [x] AGENTS.md (Codex working protocol)
  - [ ] You read Sutton RL + Pearl Causal Inference
  - [x] Codex writes literature/world_models_review.md
  - [ ] Codex writes Dreamer / MuZero / JEPA reviews

Project A (Path 2)
  - [x] Project A README + paper_outline.md
  - [x] code/ full skeleton: envs / ppo / monitor / main / evaluate / README
  - [x] Smoke test PASSED: AUROC 0.71 on CartPole CPU
  - [ ] You review paper_outline + code (REVIEW-ME markers)
  - [ ] Run full 60K step training + multiple seeds
  - [ ] Add LunarLander / Acrobot second environment
  - [ ] Codex writes paper_draft_v0.md

Community & Compute (Path 3-5)
  - [x] Twitter 4 versions
  - [x] Discord / Reddit / Email
  - [x] HF / Google Cloud / Lambda Labs drafts
  - [ ] You review drafts, submit HF first, others in 1 month
  - [ ] You post Twitter Version 1

Decisions (Path 6)
  - [x] Decision 0001: PhD vs Independent (open)
  - [x] Decision 0002: Project A first env (decided CartPole)
  - [x] Decision 0003: Project A main claim (decided Decoupling)
  - [ ] Decision 0004 (next): Second environment
`

---

## KPI Tracker

| Metric | Current | Year 0 end target |
|---|---|---|
| Smoke test result | **AUROC 0.71, Pearson 0.36** | (baseline locked) |
| arXiv papers | 0 | >=1 |
| GitHub stars | 0 (not pushed) | >=50 |
| main-conference paper | 0 | Year 1 target >=1 |
| Twitter followers | 0 | >=50 |
| Critique partners | 0 | >=2 |
| GPU hours accumulated | 0h (CPU only so far) | >=100h |
| Monthly compute budget | \ | >= |
| Public notes / blog | 0 | >=5 |

---

## Permanent log (you -> your future self)

- 2026-07-25: **Decoupling is the key trick** for self-improvement. Joint-trained critic gets dragged by PPO updates, loses discrimination power.
- 2026-07-25: **No GPU needed for serious RL experiments**. CartPole 8K steps + 50 eval episodes runs in 19 seconds.
- 2026-07-25: **System 1 / System 2 analogy**: PPO = System 1, Monitor = System 2 minimal form.

---

## Blockers

- **No GPU** - CartPole works CPU, but Atari / MuJoCo need GPU.
- **No real-person critique partner yet** - need to post on Discord/Reddit.

---

## Next session opening prompt

Copy this into a new conversation:

`
I am the owner of E:\agi-research\. Continue the AGI research.
1. Read AGENTS.md and PROGRESS.md and ROADMAP.md.
2. Top priorities for this session:
   (a) Run full CartPole training (--total-steps 60000)
   (b) Add LunarLander second env
   (c) Once I confirm code review, write paper_draft_v0.md
3. Decision 0001 (PhD vs Independent) is open, no need to push now.
`

---

## Tonight: 12 things done, you review 5, run 1

### Already done by Codex (12):
1. Workspace skeleton (40+ files)
2. AGENTS.md, README.md, ROADMAP.md
3. literature/world_models_review.md
4. Project A README + paper_outline.md
5. Project A full code (5 .py + README)
6. Smoke test pass (AUROC 0.71, Pearson 0.36)
7. Grant application drafts (HF, Google, Lambda)
8. Twitter 4 versions
9. Discord / Reddit / Email drafts
10. Decision Records (3)
11. Experiment log experiments_log/2026-07-25-smoke-test.md
12. PROGRESS.md (this file)

### You review (5):
1. README.md - overall organization
2. projects/project_a_self_improvement/code/ppo.py - PPO impl
3. projects/project_a_self_improvement/code/monitor.py - Monitor design
4. community/twitter_intro.md - Twitter drafts
5. grant_applications/hugging_face_residency.md - HF application

### You run tonight (1):
`
cd projects\project_a_self_improvement\code
"C:\Users\Administrator\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" main.py --env CartPole-v1 --total-steps 60000 --eval-episodes 80
`

Then evaluate:
`
"C:\Users\Administrator\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" evaluate.py --env CartPole-v1 --n-episodes 100 --seed 999
`

---

Open new conversation with: "Open E:\agi-research\PROGRESS.md, continue from there."

## Last session state (2026-07-26)

**Major progress**:
- [x] **GitHub repo public**: github.com/aidless/agi-research (46 commits pushed 2026-07-26)
- [x] **Zhihu announcement posted**: https://www.zhihu.com/pin/2064649194275714554
- [x] **PUBLICATION HOLD LIFTED** (Amendment 26): incremental public artifacts now permitted
- [x] **Author attribution established**: 刘泽文 (Liu Zewen) in LICENSE + AUTHORS + README + key files

**Pending**:
- [ ] Cross-post to CSDN + OSCHINA
- [ ] Continue engineering (MountainCar joint ablation, slot-WM, TTC)
- [ ] PhD application templates (SoP, CV, writing sample)
- [ ] Y1 Procgen baseline (requires cmake + VS build tools)

**Work Board additions** (Y0 Q3 round 5):
  Community (Path 3-5)
  - [x] Zhihu 想法 posted (one platform hit)
  - [ ] CSDN cross-post
  - [ ] OSCHINA cross-post
  - [ ] GitHub repo traffic tracked## Last session state (2026-07-27)

**Major progress**:
- [x] **Phase 1.5 full 4-layer AGI integration** committed (df0ef81): A + C + D + E + Q all active in one orchestrator
- [x] **Slot World Model** (Phase 1.2): slot-attention with dynamics head, next-step error = 0.000007
- [x] **A+C integration** (8d89ca0): SlotMonitor AUROC 0.989 vs raw 0.796 (+0.193 breakthrough)

**Pending**:
- [ ] Run full 100K PPO for honest Phase 1.5 numbers (in progress, background PID 28024, started 10:45)
- [ ] Update Paper A v2 with 100K integration results
- [ ] Push v0.2.0 to GitHub (requires human approval per policy)
- [ ] Cross-post to CSDN + OSCHINA
- [ ] Continue engineering (slot-WM transfer ablation, TTC multi-seed)
- [ ] Y1 Procgen baseline (still blocked on cmake + VS build tools)

**Work Board additions** (Y0 Q3 round 6):
  Engineering
  - [x] Phase 1.2 Slot World Model (perception + dynamics)
  - [x] Phase 1.3 Language interface (project_d)
  - [x] Phase 1.4 LTL verifier (project_e)
  - [x] Phase 1.5 Full 4-layer orchestrator (full_integration.py)
  - [ ] 100K PPO integration run (in progress)
  - [ ] Update Paper A v2 with integration numbers

  Infrastructure
  - [x] Identified root cause of Codex task interruption: 3 duplicate full_integration.py runs were competing for CPU; killed 2 old runs, kept the new one
- [ ] Long-running experiments must go to background process (not synchronous shell_command)

---



## Multi-seed sweep update (2026-07-27, DEC-0011)

- [x] **Phase 1.5 5-seed sweep complete** (37 min wall time, sequential background)
  - Mean delta: **+21.5 +/- 67.1** (n=5, sample std)
  - 3/5 seeds positive delta (60%)
  - t=0.72, df=4, p>0.05 (NOT statistically significant)
- [x] **H1 status: NOT falsified but NOT supported either**
  - Direction preserved; insufficient power at n=5
  - Required for significance: ~45 seeds OR n_eval=50 per seed (10x)
- [x] **Identified dominant variance source**: Q calibration, not Monitor frequency
  - avg_gates varies 0.4 -> 287.2 across seeds (Monitor not calibrated)
  - When Q well-calibrated, gating helps (+54, +91, +62); when miscalibrated, gating hurts (-55, -45)
- [x] **DEC-0011 v0.2 next iteration logged**: n_eval=50, Q coverage guard, Platt-scale Monitor, then re-sweep

See: \experiments_log/2026-07-27-phase15-5seed.md\, Paper Section 4.10.1

---



## 2026-07-27 synthesis (after Phase 2.7 cross-validation)

- [x] **Phase 1.5 5-seed** (DEC-0011): delta_avg = **+21.5 +/- 67.1** (n=5, p>0.05, NOT significant)
- [x] **Phase 2.7 multi-seed** (parallel Codex session): best gated (thresh=0.6) vs best ungated (thresh=0.9) = **-26.6**
- [x] **Synthesis**: both reach same direction - **gating does not reliably improve LunarLander** with current architecture
  - Monitor AUROC 0.989 is strong (decoupling signal REAL at prediction level)
  - But online (state->action) gating is too unstable to extract value
  - H1 supported at **monitor-prediction** level (4.6-4.8), NOT at **policy-action** level (4.10.1+4.10.2)
- [x] **Paper v2.3**: added Section 4.10.1 (5-seed) and 4.10.2 (Phase 2.7 cross-ref)

Next: Y1 work - better Q (more data), better Monitor calibration (Platt), action-level intervention

---




---

## 2026-07-27 evening (continuing) — Cross-post drafts v3 ready

**Major progress**:
- [x] **CSDN + OSCHINA v3 drafts** (97ec0db): 471 lines added
  - csdn_announcement_v3.md (224 lines, ~2200 字 CSDN 博客草稿)
  - oschina_announcement_v3.md (144 lines, ~1500 字 OSCHINA 项目资讯草稿)
  - CROSSPOST_CHECKLIST.md (跨贴 checklist, v1→v2→v3 历史、平台差异、IP 保护)
- [x] **v2 → v3 更新内容**:
  1. ENWI Prediction 2 100-epoch 复验（1.9x 差，比 30-epoch 3.5x 略好但仍负）
  2. Phase 1.5 5-seed sweep DEC-0011（delta_avg=+21.5±67.1, p>0.05）
  3. H1 区分 monitor-prediction 层 vs policy-action 层
  4. 加入诚实负结果区块（同行评议友好）
- [x] **Total commits**: 73 → 74 (97ec0db)

**Pending (需要用户手动操作)**:
- [ ] 用户去 CSDN 编辑器发布 csdn_announcement_v3.md
- [ ] 用户去 OSCHINA 项目页发布 oschina_announcement_v3.md
- [ ] 发布后回这里记录链接 + 更新 CROSSPOST_CHECKLIST.md

**Drafts 状态**:
| 平台 | 草稿 | 状态 |
|------|------|------|
| CSDN | csdn_announcement_v3.md | ✅ 草稿就绪, 等发布 |
| OSCHINA | oschina_announcement_v3.md | ✅ 草稿就绪, 等发布 |
| 知乎 | (v1, 2026-07-26 已发布) | ✅ https://www.zhihu.com/pin/2064649194275714554 |

**Work Board additions (Y0 Q3 round 7)**:
  Community (Path 3-5)
  - [x] CSDN + OSCHINA v3 drafts (含 100-epoch + DEC-0011 + 诚实负结果)
  - [x] CROSSPOST_CHECKLIST.md (跨贴追踪)
  - [ ] 用户发布 CSDN + OSCHINA 帖子
  - [ ] 发布后 24h 内回评论 + 记录阅读量

  Other (待办)
  - [ ] 推送 v0.2.0 到 GitHub (需要用户手动 push, 当前 6 commits ahead)
  - [ ] Procgen baseline (Y1, 仍需 cmake + VS build tools)

---

*Session state at 2026-07-27 evening: 74 commits, 5-seed DEC-0011 logged,
CSDN + OSCHINA v3 drafts ready for human posting.*

## DEC-0011 v0.2 negative result (2026-07-27)

- [x] **v0.2 implementation complete**: calibration.py + full_integration_v2.py
  - Train/val 80/20 split, Platt scaling, target FPR=10% threshold, Q coverage guard
  - n_eval raised 5 -> 50
- [x] **5-seed sweep**: all 5 seeds ran in parallel (~14 min)
- [x] **Result: STRONG NEGATIVE - 0/5 positive delta**
  - v0.2 delta = -158.1 +/- 208.6 (vs v0.1 +21.5 +/- 67.1)
  - 3 of 5 seeds had val_auroc=1.000 on 4 positives -> overfit -> cal_threshold ~= 0
  - Q coverage guard didn't help (Q had data, just bad)
- [x] **Decision: REJECT v0.2, keep v0.1 as canonical**
- [x] **DEC-0011 v0.3 candidates logged**: larger val set / skip cal / Q uncertainty / larger Q / safe-action fallback / new env

### Artifacts
- code/calibration.py (NEW, 110 lines)
- code/full_integration_v2.py (NEW, 450 lines)
- experiments_log/2026-07-27-phase15-v0p2-calibrated.md (formal log)
- experiments_log/phase15_v0p2_vs_v0p1_summary.json (aggregated)
- Paper Section 4.10.3 (NEW), header v2.3 -> v2.4

---




---

## 2026-07-27 late evening — Thesis v1.0 + AIE + DLR (3 engineering outcomes)

**Major progress**:
- [x] **Thesis v1.0** (ba78b8b): 2227 lines, 84.6 KB Markdown
  - 8 Parts (I-VIII) + 3 Addendum + 8 Appendices + 45 References
  - Growth from v0.1 (313 lines, 10.6 KB) → v1.0 (7.1x lines, 8.0x size)
  - Full theorems (11 ENWI), full predictions (5 ENWI), all 5-seed results
- [x] **AIE full training** (1be8f93, project_a/code/aie_train_full.py): 3 seeds
  - Mean eval: -139.3 ± ~44 (random -150 to -200, PPO 100K -100 to +50)
  - Free energy loss 21.7 → 19.5 (perception learning works)
  - Honest: ENWI Prediction 4 not testable at 10K steps; needs 100K+
  - Bugfix: active_inference.py select_action NaN handling
- [x] **DLR full integration** (1be8f93, project_e/code/dlr_train_full.py): 3 seeds
  - 6/7 predicates: 94% mean accuracy (landed 99.4%, leg 98%, in_pad 93%)
  - `upright` fails (45%) due to random slot projection losing angle info
  - DLR vs LTL: comparable on crisp; DLR advantage is differentiable training
  - Honest: F(leg_l AND leg_r) DLR Brier 0.582 worse than LTL — learned aggregation needed

**Total commits**: 79 (+4 tonight: ba78b8b, 1be8f93, plus earlier v3 cross-post)

**Pending (need user action)**:
- [ ] Push 10 commits ahead to GitHub (`git push` when ready; Codex has no auth)
- [ ] User posts CSDN (csdn_announcement_v3.md) + OSCHINA (oschina_announcement_v3.md)
- [ ] Reply to comments within 24h of posting

**Work Board additions (Y0 Q3 round 8)**:
  Thesis
  - [x] v1.0: 8 parts + 3 addendum + 8 appendices + 45 refs (84.6 KB)
  - [ ] Render to PDF for visual check
  - [ ] Add figures (currently text-only)

  AIE (Project A)
  - [x] aie_train_full.py: 3 seeds, ~10K steps each
  - [ ] Run 4x longer (40K steps) to test if AIE converges
  - [ ] Add recurrence to AIE (carry latent state across steps)

  DLR (Project E)
  - [x] dlr_train_full.py: 3 seeds, 7 predicates
  - [ ] Try learned aggregation (attention over slots) to fix `upright`
  - [ ] End-to-end training of projection + predicate jointly

  Other (next session)
  - [ ] Y1 Procgen baseline (still blocked on cmake + VS build tools)
  - [ ] PhD application templates (SoP, CV, writing sample)
  - [ ] 2000-epoch ENWI Prediction 2 replication (Y1 work, 60 min compute)

---

*Session state at 2026-07-27 late evening: 79 commits, thesis v1.0,
AIE + DLR full training runs, all honest negatives logged.*

## DEC-0011 v0.3 negative (2026-07-27)

- [x] **v0.3 implementation**: full_integration_v2.py 加 --safe-action 旗标
  - safe_action>=0 取代 Q-BoN, 选一个固定 action 代替
  - 5 seed 并行后 sweep (safe_action=2, n_eval=50, ~17 min)
- [x] **Result: 统计显著负向 - t=-3.71** (第一个显著的负面结果)
  - delta_avg = -717.6 +/- 432.2, 0/5 pos seeds
  - main engine 启发式在 Monitor 触发时大幅度伤害 policy

### 三路总成 (v0.1 vs v0.2 vs v0.3)

| 版本 | Gated | Delta | t | Pos |
|------|-------|-------|---|-----|
| v0.1 (Q-BoN) | 76.6 +/- 34.0 | +21.5 +/- 67.1 | 0.72 | 3/5 |
| v0.2 (cal. Q) | -45.6 +/- 230.7 | -158.1 +/- 208.6 | -1.69 | 0/5 |
| v0.3 (safe=2) | -685.1 +/- 416.3 | -717.6 +/- 432.2 | -3.71 (sig.) | 0/5 |

**每个后续干预都让事情变差。**

### H1 最终状态 (DEC-0011 v0.4 closeout)

- **Monitor prediction 层级**: SUPPORTED (Sections 4.6-4.8, AUROC delta=0.793)
- **Policy action 层级**: UNRESOLVED (v0.1 mixed, v0.2/v0.3 negative)
- **原因**: 200 PPO rollouts 不够训练 action-selection 层
- **下一步**: 1000+ rollouts / 新 env / 模仿学习 / 或停手

### Artifacts
- paper Section 4.10.4-4.10.6 (NEW), header v2.4 -> v2.5
- experiments_log/2026-07-27-phase15-v0p3-safe-action.md (formal log)
- experiments_log/phase15_v0p1_v0p2_v0p3_summary.json (3-way summary)

---

