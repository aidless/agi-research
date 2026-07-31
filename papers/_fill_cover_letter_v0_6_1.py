# -*- coding: utf-8 -*-
"""
Fill in the [TBD: verdict per Pre-Reg Amendment 1 kill switch]
placeholder in the COLM 2026 cover letter v0.6.1 after the
GSM8K aggregation completes.

Reads:
    experiments_log/_h10_n20_gsm8k_bootstrap.json

Writes:
    papers/cover_letter_colm2026_v0_6_1.md  (in place)

Run from any cwd once aggregator has produced bootstrap JSON:
    python papers/_fill_cover_letter_v0_6_1.py
"""

import json
import os
import re
import glob

PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(PAPERS_DIR, "..", "experiments_log"))
COVER_PATH = os.path.join(PAPERS_DIR, "cover_letter_colm2026_v0_6_1.md")

VERDICT_TEXT = {
    "EXTEND-N50": "EXTEND to n=50 (F-J >= +0.10; direction-consistent signal triggers the pre-registered kill switch to extend the study)",
    "STOP-PAPER-REFUTED-AMBIGUOUS": "REFUTED with F-J in [0, +0.10); the simple-arithmetic AND GSM8K 200-token results both show no detectable decoupling benefit",
    "STOP-PAPER-REFUTED-REVERSE": "REFUTED with F-J < 0; H10 fails with a consistent Joint > Frozen direction across both simple-arithmetic AND GSM8K 200-token task families",
}


def find_bootstrap():
    matches = sorted(glob.glob(os.path.join(EXP_DIR, "_h10_n20_gsm8k_bootstrap.json")))
    if not matches:
        return None
    return matches[-1]


def main():
    print("=== Fill COLM 2026 cover letter v0.6.1 with GSM8K verdict ===")
    js = find_bootstrap()
    if not js:
        print(f"  No bootstrap JSON at {os.path.join(EXP_DIR, '_h10_n20_gsm8k_bootstrap.json')}")
        print(f"  Run aggregator first.")
        return 1
    print(f"  Reading {js}")
    with open(js, encoding="utf-8") as f:
        report = json.load(f)
    decision = report.get("kill_switch_decision", "TBD")
    verdict_str = VERDICT_TEXT.get(decision, f"Kill-switch verdict: {decision} (interpret per Pre-Reg Amendment 1)")

    with open(COVER_PATH, "r", encoding="utf-8") as f:
        cover = f.read()

    old = "[TBD: verdict per Pre-Reg Amendment 1 kill switch]"
    if old not in cover:
        print(f"  Placeholder {old!r} not found in cover letter (already filled?)")
        # If not present, look for the verdict-style text and report
        return 0

    new_cover = cover.replace(old, verdict_str)
    with open(COVER_PATH, "w", encoding="utf-8") as f:
        f.write(new_cover)
    print(f"  WROTE {COVER_PATH}")
    print(f"  Replaced [TBD] with: {verdict_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())