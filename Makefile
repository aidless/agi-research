# Archimedes Project Makefile
# Common commands for paper building, figure generation, audits,
# and reproduce. Requires bash on Linux/macOS or Git-Bash on Windows.
# On Windows PowerShell, use the .ps1 launchers in experiments_log/
# directly.

PY ?= python
LATEX ?= pdflatex
PANDOC ?= pandoc

.PHONY: help
help:
	@echo "Archimedes Project Makefile"
	@echo ""
	@echo "Main targets:"
	@echo "  make pdfs       Build all 5 paper PDFs (Y1, Y3, Y4, Y5, thesis)"
	@echo "  make figures    Regenerate all PNGs in papers/figures_v2/"
	@echo "  make audit      Run data audit (cross-reference all numbers)"
	@echo "  make reproduce  Print reproduce commands"
	@echo "  make arxiv-dry-run"
	@echo "  make arxiv-submit (requires ARXIV_TOKEN env var)"
	@echo "  make clean      Remove LaTeX aux/log/out/toc files"
	@echo "  make test       Run audit (alias for now)"
	@echo "  make help       This message"

# ----- paper PDFs -----
.PHONY: pdfs
pdfs: y1.pdf y3.pdf y4.pdf y5.pdf thesis.pdf
	@echo "All 5 PDFs built."

y1.pdf: papers/y1_9hypothesis_framework.md
	cd papers && $(PANDOC) y1_9hypothesis_framework.md -o y1_9hypothesis_framework.tex \
		--from=markdown --to=latex --standalone --pdf-engine=xelatex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error y1_9hypothesis_framework.tex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error y1_9hypothesis_framework.tex

y3.pdf: papers/monitor_signal_vs_dlr_6pathway.md
	cd papers && $(PANDOC) monitor_signal_vs_dlr_6pathway.md -o monitor_signal_vs_dlr_6pathway.tex \
		--from=markdown --to=latex --standalone --pdf-engine=pdflatex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error monitor_signal_vs_dlr_6pathway.tex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error monitor_signal_vs_dlr_6pathway.tex

y4.pdf: papers/project_g_v0_5_h10_paper.md
	cd papers && $(PANDOC) project_g_v0_5_h10_paper.md -o project_g_v0_5_h10_paper.tex \
		--from=markdown --to=latex --standalone --pdf-engine=pdflatex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error project_g_v0_5_h10_paper.tex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error project_g_v0_5_h10_paper.tex

y5.pdf: papers/y5_monitor_transfer_synthesis.md
	cd papers && $(PANDOC) y5_monitor_transfer_synthesis.md -o y5_monitor_transfer_synthesis.tex \
		--from=markdown --to=latex --standalone --pdf-engine=pdflatex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error y5_monitor_transfer_synthesis.tex
	cd papers && $(LATEX) -interaction=nonstopmode -halt-on-error y5_monitor_transfer_synthesis.tex

thesis.pdf: thesis_draft_v2.0.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error thesis_draft_v2.0.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error thesis_draft_v2.0.tex

# ----- figure generation -----
.PHONY: figures
figures:
	$(PY) experiments_log/_mk_figures.py
	$(PY) experiments_log/_mk_more_figures.py
	$(PY) experiments_log/_mk_global_fig.py

# ----- audit -----
.PHONY: audit
audit:
	$(PY) experiments_log/_data_audit.py
	@echo "Audit complete."

# ----- reproduce (does NOT actually run the experiments, just prints commands) -----
.PHONY: reproduce
reproduce:
	@echo "To reproduce the Y3 paper, see papers/REPRODUCE.sh"
	@echo "To reproduce the Y4 H10 n=100 pilot:"
	@echo "  PowerShell: powershell -File experiments_log/_run_h10_n100.ps1"
	@echo "To reproduce the Y4 H10 n=20 pilot:"
	@echo "  PowerShell: powershell -File experiments_log/_run_h10_n20.ps1"

# ----- arxiv submission (requires ARXIV_TOKEN env var) -----
.PHONY: arxiv-dry-run arxiv-submit
arxiv-dry-run:
	cd papers/arxiv_submission && $(PY) arxiv_submit.py --dry-run

arxiv-submit:
	@if [ -z "$$ARXIV_TOKEN" ]; then \
		echo "ERROR: ARXIV_TOKEN env var not set. Get a token at https://arxiv.org/user"; \
		exit 1; \
	fi
	cd papers/arxiv_submission && $(PY) arxiv_submit.py

# ----- clean -----
.PHONY: clean
clean:
	find . -name "*.aux" -delete
	find . -name "*.log" -delete
	find . -name "*.out" -delete
	find . -name "*.toc" -delete

.PHONY: clean-all
clean-all: clean
	find . -name "*.pdf" ! -path "./thesis_draft_v2.0.pdf" ! -path "./papers/*.pdf" -delete
	@echo "Removed all generated PDFs (except in papers/ and thesis_draft_v2.0.pdf)."

# ----- test (data audit) -----
.PHONY: test
test: audit
	@echo "All tests pass."
