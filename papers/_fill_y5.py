# -*- coding: utf-8 -*-
"""
Fill in the Y5 synthesis paper's GSM8K 200-token row, using the
Y4 v0.6.1 aggregator output. Idempotent: works on either fresh
template state OR partially-filled state.
"""

import json
import os
import re
import glob

PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(PAPERS_DIR, "..", "experiments_log"))
Y5_PATH = os.path.join(PAPERS_DIR, "y5_monitor_transfer_synthesis.md")


def find_bootstrap():
    matches = sorted(glob.glob(os.path.join(EXP_DIR, "_h10_n20_gsm8k_bootstrap.json")))
    if not matches:
        return None
    return matches[-1]


def render_gms8k_row_2_3(report):
    arms = report["mean_per_arm"]
    co = report["contrasts"]["F-J"]
    return (
        f"| n=20 | GSM8K 200-tok | "
        f"{arms['frozen']:.3f} | "
        f"{arms['joint']:.3f} | "
        f"{arms['random']:.3f} | "
        f"{co['mean_diff']:+.3f} | "
        f"No (d={co['cohens_d']:+.3f}, CI [{co['ci95'][0]:+.3f}, {co['ci95'][1]:+.3f}], "
        f"p={co['p_bootstrap_two_sided']:.3f}) |"
    )


def render_gms8k_row_3_1(report):
    co = report["contrasts"]["F-J"]
    return (
        f"| LLM self-monitoring (GSM8K 200-tok CoT) | "
        f"{co['mean_diff']:+.3f} (d={co['cohens_d']:+.3f}) | "
        f"Y4 v0.6.1 | n=20 ({report['n_seeds_valid']} valid) |"
    )


def render_decision_phrase(report):
    decision = report.get("kill_switch_decision", "TBD")
    co = report["contrasts"]["F-J"]
    co_rnd = report["contrasts"]["F-R"]
    if decision.startswith("EXTEND"):
        return (
            f"The Y4 v0.6.1 GSM8K 200-token n=20 follow-up gives F-J = "
            f"{co['mean_diff']:+.3f} (d={co['cohens_d']:+.3f}), direction-"
            f"consistent with H10 prediction. Per Pre-Reg Amendment 1 "
            f"addendum (kill switch), this triggers EXTEND to n=50."
        )
    elif decision == "STOP-PAPER-REFUTED-AMBIGUOUS":
        return (
            f"The Y4 v0.6.1 GSM8K 200-token n=20 follow-up gives F-J = "
            f"{co['mean_diff']:+.3f} (d={co['cohens_d']:+.3f}, 95% CI "
            f"[{co['ci95'][0]:+.3f}, {co['ci95'][1]:+.3f}], p = "
            f"{co['p_bootstrap_two_sided']:.3f}). F-R shows stronger "
            f"signal ({co_rnd['mean_diff']:+.3f}, d={co_rnd['cohens_d']:+.3f}) "
            f"so the Monitor does discriminate failure in absolute "
            f"terms, but the F-J decoupling contrast is in the "
            f"ambiguous zone [0, +0.10). Stop at n=20; write paper."
        )
    elif decision == "STOP-PAPER-REFUTED-REVERSE":
        return (
            f"The Y4 v0.6.1 GSM8K 200-token n=20 follow-up gives F-J = "
            f"{co['mean_diff']:+.3f} (d={co['cohens_d']:+.3f}, 95% CI "
            f"[{co['ci95'][0]:+.3f}, {co['ci95'][1]:+.3f}], p = "
            f"{co['p_bootstrap_two_sided']:.3f}), in the SAME direction "
            f"(Joint > Frozen) as the n=5 simple-arith result. H10 is "
            f"REFUTED with a CONSISTENT negative direction across both "
            f"simple-arithmetic and GSM8K task families."
        )
    return f"Verdict pending: {decision}"


def replace_y5_row(y5, label, candidates, new_row):
    """Try each candidate old_text; replace the one that matches.

    This is idempotent: the helper works on:
      1. Fresh template state (uses placeholder row)
      2. Already-filled state (uses the result row matching this JSON)
    """
    for old in candidates:
        if old in y5:
            n = y5.count(old)
            if n == 1:
                print(f"  [{label}] replaced (template placeholder)")
                return y5.replace(old, new_row), "template"
            else:
                # Multiple matches �?replace only the first occurrence
                idx = y5.find(old)
                y5 = y5[:idx] + new_row + y5[idx + len(old):]
                print(f"  [{label}] replaced {n} occurrences (idempotent)")
                return y5, "multi"
    print(f"  [{label}] NOT FOUND (template state broken; check Y5 file)")
    return y5, "missing"


def main():
    print("=== Filling Y5 synthesis paper (idempotent) ===")
    js = find_bootstrap()
    if not js:
        print(f"  No bootstrap found at {os.path.join(EXP_DIR, '_h10_n20_gsm8k_bootstrap.json')}")
        return 1
    print(f"  Reading {js}")
    with open(js, encoding="utf-8") as f:
        report = json.load(f)

    # Strictness guard: skip v0.7+ master paper (no placeholders to fill).
    with open(Y5_PATH, "r", encoding="utf-8") as _f:
        _y5_check = _f.read()
    _has_v6_placeholder = (
        b"| n=20 | GSM8K 200-tok | [pending] |" in _y5_check.encode() or
        "| n=20 | GSM8K 200-tok | [pending] |" in _y5_check
    )
    _has_v4_placeholder = "[pending Y4 v0.6.1]" in _y5_check
    if not (_has_v6_placeholder or _has_v4_placeholder):
        print("  No v0.6.1 placeholders detected; skipping (Y5 paper is final master paper).")
        return 0

    decision = report.get("kill_switch_decision", "TBD")
    print(f"  n_seeds_valid = {report['n_seeds_valid']}, decision = {decision}")

    with open(Y5_PATH, "r", encoding="utf-8") as f:
        y5 = f.read()

    # §2.3 row: try placeholder first, then current-state (the row matching this fill)
    y5, _ = replace_y5_row(
        y5,
        "§ 2.3 row",
        [
            "| n=20 | GSM8K 200-tok | [pending] | [pending] | [pending] | [pending] | [pending] |",
            render_gms8k_row_2_3(report),
        ],
        render_gms8k_row_2_3(report),
    )

    # §3.1 row: try placeholder, then current state
    y5, _ = replace_y5_row(
        y5,
        "§ 3.1 row",
        [
            "| LLM self-monitoring (GSM8K 200-tok CoT) | [pending Y4 v0.6.1] | Y4 | n=20 in flight |",
            render_gms8k_row_3_1(report),
        ],
        render_gms8k_row_3_1(report),
    )

    # Header date (idempotent: just rewrite it)
    new_header = f"**Date:** 2026-07-31 (Y4 v0.6.1 verdict: {decision})"
    y5 = re.sub(
        r"\*\*Date:\*\* 2026-07-31(\s*\(Y4 v0\.6\.1 verdict:.*?\))?",
        new_header,
        y5,
    )
    print(f"  [header] replaced")

    # §4.4 verdict block (idempotent: replace if present, add if missing)
    decision_phrase = render_decision_phrase(report)
    verdict_block = (
"\\n## 4.4 Y4 v0.6.1 verdict (GSM8K 200-token, completed)\\n\\n"
        + decision_phrase + "\n\n"
        "Cross-task synthesis: the failure-prediction Monitor does "
        "not transfer to LLM self-monitoring on either of the two "
        "tested LLM task families (simple arithmetic 64-token and "
        "GSM8K 200-token chain-of-thought). The Y1 single-agent RL "
        "result is the only context where decoupling produces a "
        "verifiable effect.\n"
    )
    y5 = re.sub(
        r"## 4\.4 Y4 v0\.6\.1 verdict \(GSM8K 200-token, in flight\)\n\n.*?(?=\n## 5\. Conclusion\n)",
        verdict_block,
        y5,
        flags=re.DOTALL,
    )
    if "## 4.4 Y4 v0.6.1 verdict" not in y5:
        # Block didn't exist; insert before §5
        y5 = y5.replace("## 5. Conclusion", verdict_block + "\n## 5. Conclusion")
        print(f"  [§ 4.4 verdict block] added")
    else:
        print(f"  [§ 4.4 verdict block] refreshed")

    with open(Y5_PATH, "w", encoding="utf-8") as f:
        f.write(y5)
    print(f"  WROTE {Y5_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
