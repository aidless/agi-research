"""Front-door honesty checks.

Recomputes the figures STATUS.md treats as verified, and checks that
the README does not restore retracted headlines, a CI badge, or
author-machine paths. Stdlib only.

The README must state the number of checks in this file as
"front-door script: N checks".
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE = re.compile(r"[A-Za-z]:\\")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    na, nb = len(a), len(b)
    se = math.sqrt(std(a) ** 2 / na + std(b) ** 2 / nb)
    return mean(a) - mean(b), (mean(a) - mean(b)) / se


def problem_lines(readme: str) -> list[str]:
    match = re.search(r"## Problem\n\n(.+?)\n\n## ", readme, re.S)
    if not match:
        return []
    return [line for line in match.group(1).splitlines() if line.strip()]


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    readme = read("README.md")
    status = read("STATUS.md")
    failures = read("FAILURE_MODES.md")
    failures_lower = failures.lower()
    y13 = load("experiments_log/y13_n10_per_arm_summary.json")
    h10 = load("experiments_log/_h10_n100_bootstrap.json")
    gsm = load("experiments_log/_h10_n20_gsm8k_bootstrap.json")
    v8 = load("experiments_log/_v8_sanity_4seed.json")
    real = [float(x) for x in y13["y13_real_n15"]["per_seed"]]
    rand = [float(x) for x in y13["y13_random_n15"]["per_seed"]]
    ppo = [float(x) for x in y13["ppo_baseline"]["per_seed"]]
    delta_rr, t_rr = welch(real, rand)
    fj = h10["contrasts"]["F-J"]
    gsm_fj = gsm["contrasts"]["F-J"]
    path_files = [
        "README.md",
        "STATUS.md",
        "FAILURE_MODES.md",
        "AGENTS.md",
        "PROGRESS.md",
        "prompts/executor.md",
        "prompts/planner.md",
        "prompts/reviewer.md",
        "prompts/safety.md",
        "projects/project_a_self_improvement/code/README.md",
    ]

    checks: list[tuple[str, bool]] = []

    def add(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    lines = problem_lines(readme)
    add("problem is three lines", len(lines) == 3 and all(len(line) < 120 for line in lines))
    add(
        "front door has no drive-letter paths",
        all(DRIVE.search(read(rel)) is None for rel in path_files),
    )
    add("readme omits the unrecovered t statistic", "6.76" not in readme)
    add(
        "readme has no substrate or achieved-AGI phrasing",
        all(
            phrase not in readme
            for phrase in (
                "self-improving AGI",
                "AGI substrate",
                "AGI achieved",
                "achieved AGI",
            )
        ),
    )
    add("readme has no actions badge", "actions/workflows" not in readme and "badge.svg" not in readme)
    add("readme does not mark H10 validated", "VALIDATED" not in readme and "H10" not in readme.split("falsified")[0])
    add("status is work in progress", "WORK-IN-PROGRESS" in status and "not A yet" in status)
    add("status keeps H10 falsified", "H10" in status and "falsified" in status)
    add(
        "status keeps multi-agent monitor pathways falsified",
        "Multi-agent monitor pathways" in status and "falsified" in status,
    )
    add(
        "H10 n=100 JSON is a null with matching d",
        fj["n"] == 98
        and fj["ci95"][0] < 0 < fj["ci95"][1]
        and fj["p_bootstrap_two_sided"] > 0.05
        and abs(fj["cohens_d"] - fj["mean_diff"] / fj["sd_diffs"]) < 1e-12,
    )
    add(
        "H10 GSM8K JSON interval includes zero",
        gsm_fj["n"] == 19
        and gsm_fj["ci95"][0] < 0 < gsm_fj["ci95"][1]
        and gsm_fj["p_bootstrap_two_sided"] > 0.05,
    )
    add(
        "Y1.3 monitor versus random is not supported",
        y13["h1_verdict"]["supported"] is False
        and abs(delta_rr - 13.58) < 0.02
        and abs(t_rr - 0.781) < 0.01
        and abs(welch(real, ppo)[1] - 1.938) < 0.01,
    )
    add(
        "v8 sanity file is three seeds and not significant",
        v8["n"] == 3
        and len(v8["seeds"]) == 3
        and v8["p_two_sided_normal_approx"] > 0.05,
    )
    add("status quotes 98 valid seeds", "98 valid seeds" in status)
    add("failure modes say what is not proved", "does not prove" in failures_lower)
    add(
        "failure modes keep the null distinct from a reversal",
        "not a significant reversal" in failures_lower,
    )
    add(
        "status names claim, note, and verified",
        all(label in status for label in ("Verified", "Note", "Claim")),
    )
    add(
        "readme states this check count",
        f"front-door script: {len(checks) + 1} checks" in readme,
    )

    failed = [name for name, ok in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    for name in failed:
        print(f"FAIL: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
