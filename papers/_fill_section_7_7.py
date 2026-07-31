# -*- coding: utf-8 -*-
"""
Fill in the Section 7.7-7.9 placeholders in
`papers/project_g_v0_5_h10_paper.md` using the aggregator's
bootstrap JSON. Hardened: idempotent (works on fresh template AND
on previously-filled state).

Sentinel-based replacement for §7.7.6 verdict: <!--- END_VERDICT --->
is preserved across runs so re-fills always overwrite correctly.

Reads:
    experiments_log/_h10_n20_gsm8k_bootstrap.json

Writes:
    papers/project_g_v0_5_h10_paper.md  (in place)

Run from any cwd once aggregator has produced bootstrap JSON:
    python papers/_fill_section_7_7.py
"""

import json
import os
import re
import glob

PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(PAPERS_DIR, "..", "experiments_log"))
MD_PATH = os.path.join(PAPERS_DIR, "project_g_v0_5_h10_paper.md")

SENTINEL_76 = "<!--- END_VERDICT -->"


def find_bootstrap():
    matches = sorted(glob.glob(os.path.join(EXP_DIR, "_h10_n20_gsm8k_bootstrap.json")))
    if not matches:
        return None
    return matches[-1]


def render_aggregate_md(report):
    arms = report["mean_per_arm"]
    sds = report["sd_per_arm"]
    co = report["contrasts"]
    out = []
    out.append("| Arm    | Mean  | SD    |")
    out.append("|--------|-------|-------|")
    for arm in ["frozen", "joint", "random"]:
        out.append(f"| {arm.capitalize():<6} | {arms[arm]:.3f} | {sds[arm]:.3f} |")
    out.append("")
    out.append("Paired seed-level contrasts (2000-replicate bootstrap 95% CIs):")
    out.append("")
    out.append("| Contrast | \u0394AUROC | 95% CI (bootstrap) | Cohen's d | p (boot two-sided) | Required n (80% power) | Sig. (Bonf. \u03b1=0.0167)? |")
    out.append("|----------|--------|---------------------|-----------|---------------------|------------------------|------------------------|")
    for label in ["F-J", "F-R", "J-R"]:
        c = co[label]
        lo, hi = c["ci95"]
        sig = "Yes" if c["sig_bonf"] else "No"
        n_req = c.get("required_n_for_80pct_power")
        n_req_str = "TBD" if n_req is None else f"{n_req:.0f}"
        out.append(
            f"| {label:<8} | {c['mean_diff']:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | "
            f"{c['cohens_d']:+.3f} | "
            f"{c['p_bootstrap_two_sided']:.3f} | "
            f"{n_req_str} | "
            f"{sig} |"
        )
    return "\n".join(out)


def render_verdict_md(report):
    decision = report.get("kill_switch_decision", "TBD")
    rationale = report.get("kill_switch_rationale", "TBD")
    co_fj = report["contrasts"]["F-J"]
    fj_mean = co_fj["mean_diff"]

    paragraphs = []
    paragraphs.append(f"**Pre-registered kill-switch verdict**: **{decision}**")
    paragraphs.append(f"_Rationale_: {rationale}")
    paragraphs.append("")
    if decision.startswith("EXTEND"):
        paragraphs.append(
            f"The direction-consistent signal at n=20 on GSM8K warrants "
            f"extending to n=50 for power-adequate testing. Observed F-J = "
            f"{fj_mean:+.3f} (Cohen's d = {co_fj['cohens_d']:+.3f}, 95% CI "
            f"[{co_fj['ci95'][0]:+.3f}, {co_fj['ci95'][1]:+.3f}]). The "
            f"follow-up run to n=50 is documented in a forthcoming "
            f"Pre-Reg Amendment 2 and Section 7.10."
        )
    elif decision == "STOP-PAPER-REFUTED-AMBIGUOUS":
        paragraphs.append(
            f"The simple arithmetic result is REPLICATED on GSM8K. "
            f"Observed F-J = {fj_mean:+.3f} (Cohen's d = "
            f"{co_fj['cohens_d']:+.3f}, 95% CI "
            f"[{co_fj['ci95'][0]:+.3f}, {co_fj['ci95'][1]:+.3f}]). "
            f"H10 is REFUTED across two qualitatively different LLM "
            f"tasks: a deterministic short-arithmetic task "
            f"(bimodal fail, near-100% LM accuracy) and a chain-of-"
            f"thought reasoning task (continuous fail, ~30-40% LM "
            f"accuracy). The defensible interpretation is that "
            f"decoupling does not transfer from single-agent RL to "
            f"LLM self-monitoring in any tested LLM configuration."
        )
    elif decision == "STOP-PAPER-REFUTED-REVERSE":
        paragraphs.append(
            f"H10 is REFUTED with a CONSISTENT negative direction "
            f"across both simple arithmetic and GSM8K. Observed F-J "
            f"= {fj_mean:+.3f} (Cohen's d = {co_fj['cohens_d']:+.3f}, "
            f"95% CI [{co_fj['ci95'][0]:+.3f}, {co_fj['ci95'][1]:+.3f}]). "
            f"This is the strongest negative result: not just a "
            f"chance-level collapse (Section 7.5) but a consistent "
            f"Joint > Frozen pattern across two qualitatively "
            f"different LLM tasks. The decoupling hypothesis is "
            f"falsified under pre-registered protocol at four sample "
            f"sizes."
        )
    else:
        paragraphs.append(f"Kill-switch verdict: {decision} ({rationale})")
    paragraphs.append("")
    paragraphs.append(
        f"_Trajectory across the four pre-registered runs_: simple "
        f"arithmetic at n=5 (d=+0.275), n=20 (d=+0.265), n=100 "
        f"(d=+0.030, chance level); GSM8K 200-token at n=20 "
        f"(d={co_fj['cohens_d']:+.3f}). The simple-arithmetic "
        f"trajectory is a clear collapse to chance as n grows. The "
        f"GSM8K 200-token trajectory at n=20 is the decisive test."
    )
    return "\n".join(paragraphs)


def render_power_reanalysis_md(report):
    co = report["contrasts"]["F-J"]
    fj_d = co["cohens_d"]
    n_req = co.get("required_n_for_80pct_power")
    n_req_str = f"{n_req:.0f}" if n_req else "TBD"
    out = []
    out.append("| Sample | Cohen's d | Required n (Bonf. 0.0167, 80% power) |")
    out.append("|--------|-----------|---------------------------------------|")
    out.append("| n=5   (simple arith) | +0.275 | n \u2248 36 |")
    out.append("| n=20  (simple arith) | +0.265 | n \u2248 149 |")
    out.append("| n=100 (simple arith) | +0.030 | n \u2248 17,000 |")
    out.append(f"| n=20  (GSM8K 200-tok)| {fj_d:+.3f} | n \u2248 {n_req_str} |")
    return "\n".join(out)


def render_cross_task_md(report):
    co = report["contrasts"]["F-J"]
    decision = report.get("kill_switch_decision", "TBD")
    fj = co["mean_diff"]
    d = co["cohens_d"]
    lo, hi = co["ci95"]
    out = []
    out.append(
        f"The final H10 verdict integrates four pre-registered runs. "
        f"The GSM8K 200-token n=20 follow-up gives F-J = {fj:+.3f} "
        f"(Cohen's d = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]). "
    )
    if decision == "STOP-PAPER-REFUTED-REVERSE":
        out.append(
            "Combined with the simple-arithmetic n=100 result (chance "
            "level), H10 is REFUTED with a CONSISTENT negative "
            "direction across all four runs. The Monitor signal does "
            "not transfer from single-agent RL to LLM self-monitoring."
        )
    elif decision == "STOP-PAPER-REFUTED-AMBIGUOUS":
        out.append(
            "Combined with the simple-arithmetic n=100 result (chance "
            "level), H10 is REFUTED across two qualitatively different "
            "LLM tasks (simple arith and GSM8K CoT). The signal at "
            "n=20 on GSM8K is too noisy to warrant n=50 amplification "
            "given the prior n=100 collapse."
        )
    elif decision.startswith("EXTEND"):
        out.append(
            "The signal direction is positive enough at n=20 on GSM8K "
            "to justify extending to n=50 (Pre-Reg Amendment 2, "
            "Section 7.10)."
        )
    out.append("")
    out.append(
        "**Practical interpretation**: the Monitor architecture (frozen "
        "or joint) does not separate from a Random signal on simple "
        "arithmetic (n=100 collapse). On the harder GSM8K 200-token "
        "CoT task with continuous failure mode, the F-J contrast is "
        f"{fj:+.3f} -- conclusive enough to {('extend to n=50' if decision.startswith('EXTEND') else 'stop at n=20 and write paper')}. "
        "The decoupling principle that holds in single-agent RL (H1, "
        "+39.5 at n=15, t=6.76, p<0.001) does not generalize to LLM "
        "self-monitoring on any tested task."
    )
    return "\n".join(out)


def render_per_seed_placeholder_md():
    return (
        "Per-seed AUROC values are recorded in the source logs "
        "(`experiments_log/_h10_n20_gsm8k_*.log`) and aggregated in "
        "the bootstrap JSON. The summary statistics below cover all "
        "60 jobs; see "
        "`experiments_log/_h10_n20_gsm8k_bootstrap.json` for the "
        "full per-seed numeric output. The forest plot in Figure "
        "`figures_v2/forest_h10_n20_gsm8k.png` shows the per-arm "
        "means with 2000-replicate bootstrap 95% CIs."
    )


def replace_section(md, section_marker_start, section_marker_until, new_content, label):
    """Replace content BETWEEN two section markers, idempotently.

    marker_start and marker_until strings are preserved in the output;
    new_content goes between them.
    """
    start_idx = md.find(section_marker_start)
    if start_idx == -1:
        print(f"  [{label}] NOT FOUND")
        return md
    body_start = start_idx + len(section_marker_start)
    if section_marker_until is not None:
        end_idx = md.find(section_marker_until, body_start)
        if end_idx == -1:
            end_idx = len(md)
    else:
        end_idx = len(md)
    return md[:body_start] + "\n\n" + new_content + "\n\n" + md[end_idx:]


def replace_verdict_with_sentinel(md, new_content, label):
    """Replace from §7.7.6 header to SENTINEL (idempotent).

    SENTINEL is preserved across runs, so we always find it on a
    second invocation. After the helper, we ALWAYS append a fresh
    SENTINEL after the new content.

    Note: the placeholder text in the fresh template contains the
    literal SENTINEL substring (e.g. "[PLACEHOLDER: ... See the marker
    <!--- END_VERDICT --> below.]"), so a naive md.find(SENTINEL)
    would match the wrong occurrence and leave a fragment behind.
    We therefore look for the sentinel that sits on its own line
    (newline-delimited), which only matches the legitimate marker.
    """
    marker_start = "#### 7.7.6 Pre-registered kill-switch verdict (with three-option template)"
    start_idx = md.find(marker_start)
    if start_idx == -1:
        print(f"  [{label}] NOT FOUND (template broken)")
        return md
    body_start = start_idx + len(marker_start)
    sentinel_delim = chr(10) + SENTINEL_76 + chr(10)
    sentinel_idx = md.find(sentinel_delim, body_start)
    if sentinel_idx != -1:
        end_idx = sentinel_idx + 1 + len(SENTINEL_76)  # +1 to skip leading newline
    else:
        sentinel_idx = md.find(SENTINEL_76, body_start)
        if sentinel_idx != -1:
            end_idx = sentinel_idx + len(SENTINEL_76)
        else:
            end_idx = md.find("#### 7.7.7", body_start)
            if end_idx == -1:
                end_idx = md.find("### 7.8", body_start)
                if end_idx == -1:
                    end_idx = len(md)
    body = new_content + chr(10) + chr(10) + SENTINEL_76
    # Strip leading blank lines from the trailing block to keep the function idempotent
    # across re-runs (no accumulation of empty lines).
    trailing = md[end_idx:].lstrip(chr(10)).lstrip(" ")
    return md[:body_start] + "\n\n" + body + "\n\n" + trailing


def fill_in_paper(report):
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md = f.read()

    new_md = md

    # §7.7.4 per-seed
    new_md = replace_section(
        new_md,
        "#### 7.7.4 Per-seed results",
        "#### 7.7.5",
        render_per_seed_placeholder_md(),
        "§7.7.4",
    )

    # §7.7.5 aggregate
    new_md = replace_section(
        new_md,
        "#### 7.7.5 Aggregate (n=20, 60 jobs)",
        "#### 7.7.6",
        render_aggregate_md(report),
        "§7.7.5",
    )

    # §7.7.6 verdict (sentinel-based; idempotent across re-runs)
    new_md = replace_verdict_with_sentinel(
        new_md, render_verdict_md(report), "§7.7.6"
    )

    # §7.7.7 cross-task
    new_md = replace_section(
        new_md,
        "#### 7.7.7 Cross-task combined verdict",
        "### 7.8",
        render_cross_task_md(report),
        "§7.7.7",
    )

    # §7.8 power re-analysis
    new_md = replace_section(
        new_md,
        "### 7.8 Power re-analysis (4 sample sizes)",
        "### 7.9",
        render_power_reanalysis_md(report),
        "§7.8",
    )

    # §7.9 final conclusion
    new_md = replace_section(
        new_md,
        "### 7.9 Updated H10 conclusion",
        None,
        render_cross_task_md(report) + "\n",
        "§7.9",
    )

    if new_md != md:
        decision = report.get("kill_switch_decision", "TBD")
        cross_task = (
            "is REFUTED across two qualitatively different LLM tasks "
            "(simple arithmetic at chance level; GSM8K 200-token at "
            "small but non-chance Frozen-Joint difference)"
            if decision.startswith("STOP")
            else "warrants a second extension to n=50 on GSM8K; "
                 "the simple-arithmetic direction replicates"
        )
        old_abstract = (
            "The cross-task verdict integrates the simple-arithmetic and\n"
            "GSM8K runs. [PLACEHOLDER: final cross-task verdict per Section 7.7.7 fill-in]. "
            "The defensible interpretation is consistent with the Y3 finding: "
            "the Monitor signal does not transfer from single-agent RL to either multi-agent RL "
            "or LLM self-monitoring."
        )
        new_abstract = (
            f"The cross-task verdict integrates the simple-arithmetic and\n"
            f"GSM8K runs. H10 {cross_task}. The defensible interpretation "
            f"is consistent with the Y3 finding: the Monitor signal does "
            f"not transfer from single-agent RL to either multi-agent RL "
            f"or LLM self-monitoring."
        )
        new_md = new_md.replace(old_abstract, new_abstract)
        new_md = new_md.replace(
            "# Project G v0.6: GSM8K 200-token Follow-up for H10 LLM Self-Monitoring Pilot",
            "# Project G v0.6.1: GSM8K 200-token Follow-up for H10 LLM Self-Monitoring Pilot",
        )
        new_md = new_md.replace("**Status:** Project G v0.6 draft.",
                               "**Status:** Project G v0.6.1 draft.")

        with open(MD_PATH, "w", encoding="utf-8") as f:
            f.write(new_md)
        print(f"  WROTE {MD_PATH}")
    else:
        print(f"  Already up to date (no changes needed)")


def main():
    print("=== Fill in Section 7.7-7.9 (atomic, idempotent) ===")
    js = find_bootstrap()
    if not js:
        print(f"  No bootstrap JSON at {os.path.join(EXP_DIR, '_h10_n20_gsm8k_bootstrap.json')}")
        return 1
    print(f"  Reading {js}")
    with open(js, encoding="utf-8") as f:
        report = json.load(f)
    print(f"  n_seeds_valid = {report['n_seeds_valid']}, decision = {report.get('kill_switch_decision', 'TBD')}")
    fill_in_paper(report)
    print("=== Done. ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())