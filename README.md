# Archimedes research notebook

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Data](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](LICENSE)

## Problem

This repository is a research notebook and experiment log, not a product.
It records tests of a failure-prediction monitor trained apart from the policy.
Multi-agent and LLM self-monitor hypotheses in this log stay falsified.

## Status

**STATUS: WORK-IN-PROGRESS — not A yet.** The ledger is [STATUS.md](STATUS.md). What this page does not prove is [FAILURE_MODES.md](FAILURE_MODES.md).

The tool repository is [ai-researcher](https://github.com/aidless/ai-researcher). This repository is the notebook.

## Checked artifacts

These JSON files were recomputed for the ledger. Open them before any draft in `papers/`:

- `experiments_log/_h10_n100_bootstrap.json`
- `experiments_log/_h10_n20_gsm8k_bootstrap.json`
- `experiments_log/_h10_n20_bootstrap.json`
- `experiments_log/y13_n10_per_arm_summary.json`
- `experiments_log/h3_500k_summary.json`
- `experiments_log/_v8_sanity_4seed.json`

## CI

On every push, `.github/workflows/ci.yml` runs:

- `experiments_log/_data_audit.py`, a string search over 12 documents. A missing string does not fail the job.
- `tests/test_front_door_honesty.py`, the front-door script: 18 checks. This step fails the job when a check fails.
- `pdflatex` on `thesis_draft_v2.0.tex`.
- A mutation check that a missing `papers/HYPOTHESIS_STATUS.md` is reported.
- gitleaks over the full git history.

The Y1, Y3, Y4, and Y5 PDF jobs run only on `workflow_dispatch`. The workflow comment records that they have not passed in this workflow. There is no training-code unit-test suite. `make check` runs the 18 checks and then the string audit.

## License

Code is MIT. Notes, logs, and pre-registration text are CC-BY-4.0. See `LICENSE`.
