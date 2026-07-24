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
