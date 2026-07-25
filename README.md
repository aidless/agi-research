# AGI Research Workspace (Archimedes Project)

> **A Self-Improving AGI Substrate: Decoupled Monitors, Causal World Models,
> and Typed Language Interfaces**

**Copyright (c) 2026 刘泽文 (Liu Zewen)** — see [LICENSE](./LICENSE) and [AUTHORS](./AUTHORS).

Independent 5-year research program toward a self-improving, cross-domain,
causally-grounded, language-queryable AGI substrate.
Started: 2026-07-25.

---

## What this is

An open-source research workspace containing:

- **Project A**: Decoupled failure-prediction Monitor for RL policies (H1 supported, 5/5 seeds, delta=0.724)
- **Project B**: Cross-domain transfer architecture (cross-domain VLA, LLaVA-style)
- **Project C**: Causal world model with slot-attention perception (PoC shipped, Procgen Y1)
- **Project D**: Language-as-type-system over slot latents (sketch)
- **Project E**: Neuro-symbolic verification (three-layer alignment sketch)
- **Project F**: Multi-agent coordination with decentralized monitors (doc)

**43 deep paper notes**, 5 project outlines, 30+ commits, joint ablation
result that empirically isolates "decoupling" as the mechanism behind
self-monitoring success.

## One-line summary

> Build a 4-layer self-improving AGI substrate (Self-Model + World Model +
> LLM-as-type-system + VLA), ground it on the 5-route fusion (Scaling +
> Neuro-Symbolic + World Models + Embodied + Cognitive Arch), measure it
> by **learning rate on novel tasks** (not absolute performance), ship at
> least one reference implementation cited by >= 3 frontier labs within
> 5 years.

## Quickstart

```
# Clone
git clone https://github.com/[USER]/agi-research.git
cd agi-research

# Install deps (Python 3.11+)
pip install torch numpy gymnasium

# Run LunarLander H1 ablation (5 seeds, ~13 min on CPU)
python projects/project_a_self_improvement/code/joint_phase2.py \
  --n-ppo-steps 100000 --n-train-episodes 200 --n-eval-episodes 100 \
  --history-len 32 --monitor-interval 4 --monitor-epochs-per-step 2 \
  --n-monitor-rollouts 20 --seed 0
```

See `projects/project_a_self_improvement/code/README.md` for full protocol.

## Key results

| Project | Status | Evidence |
|---------|--------|----------|
| A (Monitor) | **H1 supported 5/5 seeds** | Frozen AUROC 0.796 vs Joint 0.072, delta=0.724 |
| C (Slot-WM) | PoC | Diversity loss 0.39 -> 0.22 on CPU |
| D (LLM-type) | Sketch | Hindley-Milner over slot latents |
| E (Verifier) | Sketch | Outer/Inner/Corrigibility three-layer framing |

## Project structure

```
E:\agi-research\
+- AUTHORS, LICENSE, README (this file)
+- TASKBOOK_v1.md             # project charter (5-year program)
+- ROADMAP.md                 # 5-year vision + reading list
+- PROGRESS.md                # cross-session state
+- AGENTS.md                  # collaboration protocol with Codex
+- PUBLICATION_HOLD.md        # publication strategy
+- CHANGELOG.md               # amendments to TASKBOOK_v1
+- decisions/                 # Decision Records (P1/P2/P3)
+- literature/                # 43 paper deep notes + 2 TMLR syntheses
+- projects/                  # A/B/C/D/E project folders
+- experiments_log/           # one .md per experiment
+- community/                 # Twitter/Discord/Reddit drafts
+- bin/                       # 7 workspace CLI tools
+- prompts/                   # multi-agent prompts
+- 00_daily/                  # daily review notes
```

## Citation

See [AUTHORS](./AUTHORS). Cite as:

> Liu Zewen (2026). Archimedes: A Self-Improving AGI Substrate.
> Independent 5-year research program, AGI-2026-001.

## License

MIT — see [LICENSE](./LICENSE). Copyright (c) 2026 刘泽文 (Liu Zewen).

---

*This workspace is the output of an independent researcher + AI assistant
collaboration. All AI-generated content is reviewed by the PI before
inclusion. The intent is open publication with explicit attribution
to prevent IP misappropriation while allowing free reuse.*