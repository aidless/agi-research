# PUBLICATION HOLD - 2026-07-25 (user directive)

> **Status**: ACTIVE HOLD on all public-facing publications
> **Decision date**: 2026-07-25
> **Decided by**: user
> **Strategy**: 闆嗘潫鍙戝竷 (bundled publication) 鈥?wait for a single 100+ page
> comprehensive thesis-style paper rather than incremental arXiv / Twitter / Discord posts.

---

## 1. What is on hold

Do NOT post publicly until this hold is lifted:

### 1.1 Paper drafts (internal only)
- `projects/project_a_self_improvement/paper_v2_full.md` (28 KB, 374 lines)
  - Joint ablation 5-seed result (delta=0.724, 5/5 H1 supported)
  - frozen-critic family framing (STaR/ReAct/Reflexion/Self-Refine/CRITIC/PRM)
- `projects/project_d_language/paper_outline_v0.md` (6.3 KB)
- `projects/project_e_verification/paper_outline_v0.md` (9.2 KB)
- `projects/project_c_causal_world/paper_outline_v0.md` + `paper_outline_v1.md`

### 1.2 Community posts (drafts only, do NOT post)
- `community/twitter_joint_ablation.md` (4 Twitter versions, ready)
- `community/discord_joint_ablation.md` (Discord/Reddit draft, ready)
- `community/twitter_intro.md` (4 intro versions, ready)
- `community/discord_reddit_email.md` (Discord + Reddit + email drafts, ready)
- `community/finding_critique_partners.md` (critique partner playbook)

### 1.3 Grants (do NOT submit until hold lifted)
- `grant_applications/hugging_face_residency.md`
- `grant_applications/google.md`
- `grant_applications/lambda_labs.md`

---

## 2. What is NOT on hold (continue as normal)

- Code commits to local git (no remote push)
- Internal paper deep notes
- Internal experiment logs
- Internal decision records
- Internal synthesis documents (TMLR H+I series, etc.)
- ROADMAP / TASKBOOK / CHANGELOG / PROGRESS internal updates
- Daily session logs (.experience_log/)

These are PRIVATE workspace artifacts. No external audience.

---

## 3. Why the hold (rationale)

User said: "鍏堜繚鐣欙紝浠ュ悗绛夌潃鍙戜竴绡囦笂鐧鹃〉鐨勫ぇ璁烘枃" (Hold off, wait for a single 100+ page comprehensive paper).

Benefits of bundling:
1. **Single impact event**: one large paper > many small posts
2. **Avoid scooping risk**: if we publish joint ablation alone, others
   might replicate + publish before our comprehensive work
3. **Thesis-style authority**: a 100+ page document commands more
   citation weight than 5 incremental arXiv posts
4. **Coherent narrative**: projects A/B/C/D/E integrated as one
   4-layer story, not five disconnected papers
5. **Avoid premature "AIKR" framing fatigue**: 5 small posts would
   dilute the AIKR framing; one big paper amplifies it

Costs of holding:
- Slower external impact
- Risk that other researchers publish similar work first
- Reduced critique-partner acquisition rate
- Tighter coupling between Projects A-E (any one blocks all)

---

## 4. The big paper 鈥?target structure

Working title (provisional):

> **A Self-Improving AGI Substrate: Decoupled Monitors, Causal World
> Models, and Typed Language Interfaces**

Estimated 100-160 pages, structured as:

| Part | Section | Pages | Source material |
|------|---------|-------|-----------------|
| I    | Foundations: AIKR, 4-layer arch, related work | ~15 | TASKBOOK v1, 43 paper notes |
| II   | Project A: Self-Improvement via Decoupled Monitors | ~25 | paper_v2_full.md (expand) |
| III  | Project B: Cross-Domain VLA Transfer | ~15 | paper_outline_v0/v1 (Y1 work) |
| IV   | Project C: Causal World Models with Slot Attention | ~20 | paper_outline_v0/v1, slot_attention.py (Y1 work) |
| V    | Project D: Language as Type System | ~10 | paper_outline_v0 (Y1 work) |
| VI   | Project E: Neuro-Symbolic Verification | ~15 | paper_outline_v0 (Y1 work) |
| VII  | Integration: end-to-end demo + cross-domain + verification | ~10 | (Y2 work) |
| VIII | Discussion: limitations, future, AGI timeline | ~10 | all projects |
| Refs | References (~80 entries) | ~10 | 43 paper notes + 2025-2026 wave |
| App  | Appendices (architecture details, hyperparams, ablations) | ~15 | experiment logs |
| **Total** | | **~145** | |

---

## 5. Timeline (provisional)

- **2026 Q3-Q4 (Y0 remaining)**: Paper A joint ablation shipped;
  Project C/D/E sketches documented; ROADMAP reading list complete.
- **2027 Q1-Q2 (Y1 H1)**: Project C Procgen baseline (DIAMOND vs
  slot-WM); Project D type system PoC; Project E LTL verifier PoC.
- **2027 Q3-Q4 (Y1 H2)**: cross-domain transfer eval (B+C+D+E
  integration); TTC extension (ADR 0011); first integrated
  end-to-end demo.
- **2028 Q1-Q2 (Y2)**: scale experiments; multi-seed runs;
  ablation studies; write the comprehensive paper draft.
- **2028 Q3-Q4**: submit big paper. Target venue: arXiv monograph
  (preferred), PhD thesis (if DEC-001 鈫?PhD), or Springer book.

Decision deadline: lift hold at Y2 Q3 (~2028-07), unless earlier if
all projects hit their Y1 milestones.

---

## 6. Lift criteria (any of these triggers review)

1. **Y1 H2 milestones all hit**: A Monitor Procgen AUROC > 0.85
   AND C slot-WM transfer > 0.6 AND D type system 80% consistency
   AND E LTL verifier 95% symbolic agreement
2. **External trigger**: a competing paper publishes similar
   findings (joint ablation, frozen-critic pattern, slot-WM
   cross-domain) before our Y2 submission
3. **User decision**: user explicitly lifts the hold
4. **Risk of obsolescence**: hold too long and the field moves on,
   making our bundled paper stale

---

## 7. Decision log

- **2026-07-25**: hold established. All public-facing drafts frozen.
- **Y1 Q4 review** (2027-09): first scheduled hold-review.
- **Y2 Q3 review** (2028-07): default lift-or-pivot decision point.

---

*This file is the single source of truth on the publication hold.
Update it whenever the hold status changes.*

---

## 8. HOLD LIFTED (2026-07-26)

User directive: "完全 lift 我害怕别人盗用我的东西 帮我建一个
aidless 建不了那就我自己建一个"

### 8.1 Why lifted

User motivation: fear of IP theft. Open publication with explicit
attribution (LICENSE + AUTHORS + README + code comments) provides:
1. Public timestamp (GitHub commit history)
2. Clear authorship trace (LICENSE copyright + AUTHORS file)
3. Citation requirement (LICENSE attribution clause)
4. Prior art evidence for any future patent claim

The original "wait for 100+ page thesis" strategy was about impact
amplification; that strategy is incompatible with IP protection.
We prioritize IP protection over impact amplification.

### 8.2 What is now allowed

All previous holds are released:
- Paper A v2 draft can be pushed to GitHub
- Twitter/Discord drafts can be posted
- Grants can be submitted
- Critique partner outreach can proceed

We still aim for the 100+ page comprehensive thesis as the *primary*
academic output, but incremental public artifacts are now permitted
to establish authorship priority.

### 8.3 Attribution requirement

Every public artifact MUST carry the attribution:
- Author: 刘泽文 (Liu Zewen)
- Project: Archimedes (AGI-2026-001)
- Copyright (c) 2026

Attribution is embedded in:
- LICENSE file (MIT, with copyright)
- AUTHORS file (full citation block)
- README.md (top-level attribution)
- Key Python file headers (project code only)
- TASKBOOK_v1.md signature block
- Paper v2 draft author block

### 8.4 Push strategy

1. Create empty GitHub repo (user-created, since Codex has no auth)
2. Codex prepares local repo (LICENSE, AUTHORS, README, attribution comments)
3. User provides repo URL
4. Codex adds remote + git push -u origin main
5. Codex publishes draft posts (Twitter/Discord) with attribution

### 8.5 Status

- LICENSE: shipped
- AUTHORS: shipped
- README: updated
- Code comments: in progress (key files only)
- TASKBOOK signature: in progress
- GitHub repo creation: user action required (provide URL when ready)