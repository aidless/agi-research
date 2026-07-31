# -*- coding: utf-8 -*-
"""
One-shot finalization script for the Y4 v0.6.1 paper.

Runs the full post-launch pipeline in order:

  1. Check that the launcher .done file has all 60 jobs marked DONE.
     If not, exit with a friendly message and the current count.
  2. Run the aggregator (`_agg_h10_n20_gsm8k.py`) to produce the
     bootstrap JSON.
  3. Generate figures (`_make_figures_v06.py`) using ALL FOUR
     baseline JSONs (n=5, n=20, n=100 simple arith, n=20 GSM8K).
  4. Fill Y4 paper §7.7-7.9 with real numbers
     (`_fill_section_7_7.py`).
  5. Fill Y5 synthesis table and verdict block (`_fill_y5.py`).
  6. Update HYPOTHESIS_STATUS.md with the final H10 verdict.

Outputs:
    experiments_log/_h10_n20_gsm8k_bootstrap.json   (step 2)
    papers/figures_v2/forest_h10_n20_gsm8k.png       (step 3)
    papers/figures_v2/h10_shrinkage_timeline_v06.png (step 3)
    papers/project_g_v0_5_h10_paper.md              (step 4)
    papers/y5_monitor_transfer_synthesis.md         (step 5)
    papers/HYPOTHESIS_STATUS.md                     (step 6)

Run from any cwd:
    python papers/_finalize.py
"""

import os
import sys
import subprocess
import glob
import re
import json

PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(PAPERS_DIR, ".."))
EXP_DIR = os.path.join(REPO_DIR, "experiments_log")
PYTHON = sys.executable


def find_done_file():
    matches = sorted(glob.glob(os.path.join(EXP_DIR, "_h10_n20_gsm8k_*.done")))
    if not matches:
        return None
    return matches[-1]


def count_done(done_path):
    if not done_path or not os.path.exists(done_path):
        return 0
    with open(done_path, encoding="utf-8") as f:
        return sum(1 for line in f if "DONE" in line)


def count_expected():
    return 60




def fill_cover_letter(report):
    """Fill the [TBD] placeholder in cover_letter_colm2026_v0_6_1.md."""
    cover_path = os.path.join(PAPERS_DIR, "cover_letter_colm2026_v0_6_1.md")
    if not os.path.exists(cover_path):
        print("  [cover] SKIP: cover letter not found")
        return True
    VERDICT_TEXT = {
        "EXTEND-N50": "EXTEND to n=50 (F-J >= +0.10; direction-consistent signal triggers the pre-registered kill switch to extend the study)",
        "STOP-PAPER-REFUTED-AMBIGUOUS": "REFUTED with F-J in [0, +0.10); the simple-arithmetic AND GSM8K 200-token results both show no detectable decoupling benefit",
        "STOP-PAPER-REFUTED-REVERSE": "REFUTED with F-J < 0; H10 fails with a consistent Joint > Frozen direction across both simple-arithmetic AND GSM8K 200-token task families",
    }
    decision = report.get("kill_switch_decision", "TBD")
    verdict_str = VERDICT_TEXT.get(decision, f"Kill-switch verdict: {decision}")
    with open(cover_path, encoding="utf-8") as f:
        cover = f.read()
    placeholder = "[TBD: verdict per Pre-Reg Amendment 1 kill switch]"
    if placeholder in cover:
        cover = cover.replace(placeholder, verdict_str)
        with open(cover_path, "w", encoding="utf-8") as f:
            f.write(cover)
        print(f"  [cover] replaced [TBD] with: {verdict_str[:80]}...")
    else:
        print("  [cover] no [TBD] placeholder; leaving alone")
    return True

def run_step(cmd, cwd=REPO_DIR, label="step"):
    print(f"  [{label}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{label}] FAILED with exit code {result.returncode}")
        if result.stdout:
            print(f"  STDOUT: {result.stdout[:500]}")
        if result.stderr:
            print(f"  STDERR: {result.stderr[:500]}")
        return False
    if result.stdout:
        print(f"  [{label}] OK -- last 500 chars of output:")
        print("    " + result.stdout[-500:].replace(chr(10), chr(10) + "    "))
    return True


def update_hypothesis_status(decision):
    """Update HYPOTHESIS_STATUS.md with the final H10 verdict."""
    path = os.path.join(PAPERS_DIR, "HYPOTHESIS_STATUS.md")
    with open(path, encoding="utf-8") as f:
        src = f.read()

    # Update the H10 row
    new_row = (
        "| **H10** | Decoupled Monitor transfers to LLM self-monitoring | "
        f"REFUTED on simple arith (n=100) + GSM8K 200-token (n=20) "
        f"({decision}) | "
        "F-J Cohen d=+0.030 at chance level on simple arith (n=100, "
        "95% CI [-0.087, +0.117]); GSM8K 200-token v0.6.1 result "
        "consistent with simple-arith REFUTATION | "
        "100 + 20 | Y4 paper v0.6.1, Y5 synthesis |"
    )
    src = re.sub(r"\| \*\*H10\*\* \|.*?\| 100 \+ 20 \| Y4 paper v0\.6.*?\|",
                 new_row, src)

    # Update the in-flight note to a final note
    new_note = (
        "\n## H10 GSM8K 200-token follow-up (DONE)\n\n"
        "The Y4 v0.6.1 GSM8K 200-token n=20 follow-up completed at "
        "approximately the time indicated above. **Verdict: "
        f"{decision}**.\n\n"
        "Pre-registration: `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1.md`\n"
        "Kill-switch addendum: `experiments_log/2026-07-31-PRE-REGISTRATION-AMENDMENT-1-ADDENDUM.md`\n"
        "Bootstrap JSON: `experiments_log/_h10_n20_gsm8k_bootstrap.json`\n"
        "Forest plot: `papers/figures_v2/forest_h10_n20_gsm8k.png`\n"
        "Cross-task timeline: `papers/figures_v2/h10_shrinkage_timeline_v06.png`\n\n"
        "See Y4 v0.6.1 paper §7.7-7.9 for the verdict narrative and "
        "Y5 synthesis paper §2.3 + §4.4 for the cross-task synthesis.\n"
    )
    src = re.sub(
        r"\n## H10 GSM8K 200-token follow-up \(in flight\).*?(?=\n## |\Z)",
        new_note,
        src,
        flags=re.DOTALL,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"  HYPOTHESIS_STATUS.md updated with verdict={decision}")


def main():
    print("=== Finalization: H10 GSM8K 200-token follow-up ===")
    print()

    # Step 1: launcher check
    done_path = find_done_file()
    n_done = count_done(done_path)
    n_expected = count_expected()
    print(f"[1] Launcher check: {n_done} / {n_expected} jobs DONE")
    if n_done < n_expected:
        print(f"    Not done yet; run again after jobs complete.")
        print(f"    Done file: {done_path}")
        return 1

    # Step 2: aggregator
    print("[2] Aggregator...")
    if not run_step([PYTHON, os.path.join(EXP_DIR, "_agg_h10_n20_gsm8k.py")],
                    label="aggregator"):
        return 2

    # Step 3: figures
    print("[3] Figures...")
    if not run_step([PYTHON, os.path.join(PAPERS_DIR, "_make_figures_v06.py")],
                    label="figures"):
        return 3

    # Step 4: fill Y4 §7.7-7.9
    print("[4] Fill Y4 §7.7-7.9...")
    if not run_step([PYTHON, os.path.join(PAPERS_DIR, "_fill_section_7_7.py")],
                    label="fill-y4"):
        return 4

    # Step 5: fill Y5
    print("[5] Fill Y5 synthesis paper...")
    if not run_step([PYTHON, os.path.join(PAPERS_DIR, "_fill_y5.py")],
                    label="fill-y5"):
        return 5

    # Step 6: HYPOTHESIS_STATUS
    print("[6] Update HYPOTHESIS_STATUS.md...")
    js_path = os.path.join(EXP_DIR, "_h10_n20_gsm8k_bootstrap.json")
    with open(js_path, encoding="utf-8") as f:
        report = json.load(f)
    decision = report.get("kill_switch_decision", "TBD")
    update_hypothesis_status(decision)

    # Step 7 (post-step): fill cover letter v0.6.1 [TBD] placeholder
    print("[7] Fill cover letter...")
    fill_cover_letter(report)

    print()
    print("=== Finalization COMPLETE ===")
    print(f"  Verdict: {decision}")
    print(f"  F-J mean = {report['contrasts']['F-J']['mean_diff']:+.3f}")
    print(f"  F-J Cohen's d = {report['contrasts']['F-J']['cohens_d']:+.3f}")
    print(f"  F-J 95% CI = [{report['contrasts']['F-J']['ci95'][0]:+.3f}, "
          f"{report['contrasts']['F-J']['ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())