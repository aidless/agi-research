# Venue Plan for Y3 and Y4 papers (2026-07-31)

## Y3 paper

Title: Monitor Signal vs DLR Predicates in Cooperative MARL: A 6-Pathway Systematic Investigation

Current state: cover letter for AAMAS 2027 already drafted at papers/cover_letter_aamas2027.md. Package validated. v8 dlr_only n=100 plus 3-seed independent replication now included in Section 3.6.

Recommended venue: AAMAS 2027 (Mar 2027 deadline, paper submission around Sep 2026).
- Strength: MARL is the AAMAS main topic; the 6-pathway ablation design is a methodological contribution the community values.
- Strength: cs.MA is endorsement-free, so we can submit without finding an endorser.
- Strength: v8 dlr_only is the only publishable positive result (~0.09% relative improvement) but it is statistically robust (Bonferroni p<0.05) and reproducible from a fresh seed; that is exactly the kind of negative-finding + small-positive paper that AAMAS workshops prefer.
- Alternative: AAMAS MARL workshop (less competitive, more discussion time for negative results).

## Y4 paper

Title: Project G: H10 LLM Self-Monitoring (Stratified Split + n=5/n=20/n=100)

Current state: H10 REFUTED at chance level at n=100. The paper is a NEGATIVE result paper. Cover letter for COLM 2026 not yet written.

Recommended venue: COLM 2026 (Conference on Language Modeling, ICLR workshop style; submissions around Oct 2026).
- Strength: COLM specifically welcomes rigorous empirical studies, including negative results.
- Strength: H10 is a hypothesis test, not a method; COLM has a track for evaluation and reproducibility papers.
- Alternative: ACL Findings (broader audience, ACL Oct 2026).
- Alternative: NeurIPS Workshop on Aligned AI (workshop track, less competitive, AI-safety audience).

## arXiv first, then venue

Recommended order:
1. Submit Y3 to arXiv NOW (only blocker is ARXIV_TOKEN).
2. Get arXiv number; cite it in Y4.
3. Submit Y3 to AAMAS 2027 (Sep 2026).
4. Submit Y4 to arXiv (only blocker is also ARXIV_TOKEN).
5. Submit Y4 to COLM 2026 (Oct 2026).

Both packages are prepared. Both are blocked on ARXIV_TOKEN.

## Summary of next actions

| Action | Who | When | Blocker |
|---|---|---|---|
| Y3 to arXiv | script run | now | ARXIV_TOKEN |
| Y4 to arXiv | script run | now | ARXIV_TOKEN |
| Y3 to AAMAS 2027 | user submission | by Sep 2026 | arXiv number |
| Y4 to COLM 2026 | user submission | by Oct 2026 | arXiv number, COLM cover letter |
| Y4 cover letter for COLM | Codex | when ready | Y4 final v0.7 (already done) |

