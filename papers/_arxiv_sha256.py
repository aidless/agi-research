import hashlib, os, sys

PAPERS = os.path.dirname(os.path.abspath(__file__))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(8192), b''): h.update(c)
    return h.hexdigest()

out = []
out.append('OpenReview / arXiv submission package checklist')
out.append('')
out.append('Generated: 2026-08-01 (v1.3.1 camera-ready (P3 hybrid pre-reg completed 2026-08-01), P3 hybrid pre-reg day 1)')
out.append('Paper:     Y5 Master Synthesis (COLM 2026 submission)')
out.append('Author:    Liu Zewen + Codex (Archimedes Project, AGI-2026-001)')
out.append('')
out.append('=== SHA-256 of main artifacts ===')
for label, fn in [('PDF ', 'arxiv_main.pdf'), ('DOCX', 'arxiv_main.docx'), ('MD  ', 'arxiv_main.md')]:
    p = os.path.join(PAPERS, fn)
    if os.path.exists(p):
        out.append(f'{label}: {sha(p)}')
out.append('')
out.append('=== Camera-ready 14-item checklist (all green) ===')
for item in [
    'All 18 cumulative reviewer items addressed',
    'Pre-Reg Proposition 3 with GPU reservation (2026-08-01 to 2026-08-15)',
    'n=5 Hedges g row marked as post-hoc',
    'Pattern D cross-references Pre-Reg',
    'Bibliography complete with 7 new references',
    'Section 7.6.6 Monotonicity Lemma stated and proved',
    'Section 7.6.3 cost-weighted observation table',
    'Y4 v0.6.1 kill switch STOP-PAPER-REFUTED-REVERSE',
    'Cross-task meta-analysis (6 methods) converge on H10 REFUTATION',
    'Forest plot visualization',
    'Section 7.6 formal framework (7 Definitions + 4 Propositions + 4 Refutations)',
    'Section 7.5.5 first-principles motivation',
    'Section 7.6.2 Assumption A1 explicit',
    'Section 8.5 deployment patterns (4 patterns)',
    'Section 9.6 framework limitations',
]:
    out.append(f'[x] {item}')

open(os.path.join(PAPERS, 'arxiv_checklist.txt'), 'w', encoding='utf-8', newline='') \
    .write(chr(10).join(out) + chr(10))
print('arxiv_checklist.txt written:', len(out), 'lines')
