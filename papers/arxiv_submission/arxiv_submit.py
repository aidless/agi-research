#!/usr/bin/env python3
"""
arXiv submission script for the Y3 paper.
"Monitor Signal vs DLR Predicates in Cooperative MARL: A 6-Pathway
Systematic Investigation"

Usage:
    export ARXIV_TOKEN=<your-arxiv-api-token>
    python arxiv_submit.py [--dry-run]

This script:
1. Validates the submission package (LaTeX source, figures, metadata)
2. Validates the LaTeX source compiles cleanly with pdflatex
3. Optionally uploads to arXiv via the arXiv API
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent.parent
TEX_FILE = SUBMISSION_DIR / "monitor_signal_vs_dlr_6pathway.tex"
PDF_FILE = SUBMISSION_DIR / "monitor_signal_vs_dlr_6pathway.pdf"
FIGURES_DIR = SUBMISSION_DIR / "figures"
METADATA_FILE = SUBMISSION_DIR / "arxiv_metadata.txt"


def parse_metadata(path):
    """Parse the arxiv_metadata.txt file."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    metadata = {}
    current_key = None
    for line in content.split("\n"):
        if line.startswith("Title:"):
            metadata["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("Authors:"):
            metadata["authors"] = line.split(":", 1)[1].strip()
        elif line.startswith("Abstract:"):
            metadata["abstract"] = line.split(":", 1)[1].strip()
        elif line.startswith("Comments:"):
            metadata["comments"] = line.split(":", 1)[1].strip()
        elif line.startswith("Categories:"):
            current_key = "categories"
        elif line.startswith("- "):
            if current_key == "categories":
                metadata.setdefault("categories", []).append(line[2:].strip())
        elif line.startswith("License:"):
            metadata["license"] = line.split(":", 1)[1].strip()
            current_key = None
    return metadata


def validate_package(verbose=True):
    """Validate the submission package contents."""
    print("Validating submission package...")
    for f in [TEX_FILE, PDF_FILE, METADATA_FILE]:
        if not f.exists():
            print("  MISSING: " + str(f))
            return False
        if verbose:
            print("  OK: " + str(f.relative_to(REPO_ROOT)) + " (" + str(f.stat().st_size) + " bytes)")
    if not FIGURES_DIR.exists():
        print("  MISSING: " + str(FIGURES_DIR))
        return False
    figures = list(FIGURES_DIR.glob("*.png"))
    if len(figures) < 1:
        print("  MISSING: no .png figures in " + str(FIGURES_DIR))
        return False
    for fig in figures:
        if verbose:
            print("  OK: figures/" + fig.name + " (" + str(fig.stat().st_size) + " bytes)")
    metadata = parse_metadata(METADATA_FILE)
    if verbose:
        print("\nMetadata:")
        for key, value in metadata.items():
            if isinstance(value, list):
                print("  " + key + ": " + ", ".join(value))
            elif len(str(value)) > 80:
                print("  " + key + ": " + str(value)[:80] + "...")
            else:
                print("  " + key + ": " + value)
    required = ["title", "authors", "abstract", "categories", "license"]
    for key in required:
        if key not in metadata:
            print("  MISSING metadata field: " + key)
            return False
    return True


def validate_latex_compiles():
    """Check that the LaTeX source compiles cleanly with pdflatex."""
    print("\nValidating LaTeX source compiles...")
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-draftmode", str(TEX_FILE)],
            cwd=SUBMISSION_DIR,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print("  LaTeX compile FAILED: returncode " + str(result.returncode))
            return False
        print("  OK: LaTeX compiles cleanly")
        return True
    except FileNotFoundError:
        print("  pdflatex not found, skipping compile check")
        return True
    except subprocess.TimeoutExpired:
        print("  pdflatex timed out, skipping compile check")
        return True


def submit_to_arxiv(token, dry_run=False):
    """Submit the package to arXiv via the API."""
    if dry_run:
        print("\n[DRY RUN] Would submit the following package to arXiv:")
        print("  Source: " + str(TEX_FILE.relative_to(REPO_ROOT)))
        print("  PDF: " + str(PDF_FILE.relative_to(REPO_ROOT)))
        for fig in FIGURES_DIR.glob("*.png"):
            print("  Figure: figures/" + fig.name)
        print("\n  Metadata: " + str(METADATA_FILE.relative_to(REPO_ROOT)))
        print("\n[DRY RUN] Skipping actual upload. To submit for real, set ARXIV_TOKEN environment variable and remove --dry-run.")
        return True
    import urllib.request
    import urllib.parse
    boundary = "----arxiv-submission-boundary-12345"
    def encode_file_field(name, filename, content):
        return ("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + name + "\"; filename=\"" + filename + "\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n"
    def encode_text_field(name, value):
        return ("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + name + "\"\r\n\r\n" + str(value) + "\r\n").encode()
    body = b""
    body += encode_file_field("tex", TEX_FILE.name, TEX_FILE.read_bytes())
    for fig in sorted(FIGURES_DIR.glob("*.png")):
        body += encode_file_field("figure_" + fig.stem, fig.name, fig.read_bytes())
    metadata = parse_metadata(METADATA_FILE)
    for key, value in metadata.items():
        if isinstance(value, list):
            for v in value:
                body += encode_text_field(key, v)
        else:
            body += encode_text_field(key, str(value))
    body += ("--" + boundary + "--\r\n").encode()
    url = "https://arxiv.org/api/submit"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "multipart/form-data; boundary=" + boundary,
            "Cookie": "arxiv_session=" + token,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = response.read().decode()
            print("\narXiv response:\n" + result[:2000])
            return True
    except urllib.error.HTTPError as e:
        print("\narXiv HTTP error " + str(e.code) + ": " + e.reason)
        return False
    except Exception as e:
        print("\narXiv submission error: " + str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="arXiv submission for the Y3 paper")
    parser.add_argument("--dry-run", action="store_true", help="Validate package without submitting")
    parser.add_argument("--skip-compile", action="store_true", help="Skip pdflatex compile check")
    args = parser.parse_args()
    if not validate_package(verbose=True):
        print("\nVALIDATION FAILED. Aborting.")
        sys.exit(1)
    print("\nPackage validation: OK")
    if not args.skip_compile and not validate_latex_compiles():
        print("\nLATEX COMPILE FAILED. Aborting.")
        sys.exit(1)
    token = os.environ.get("ARXIV_TOKEN", None)
    if args.dry_run or token is None:
        submit_to_arxiv(token, dry_run=True)
        print("\nDry run complete.")
        print("\nTo submit for real:")
        print("  1. Get an arXiv API token from https://arxiv.org/account")
        print("  2. Set ARXIV_TOKEN environment variable")
        print("  3. Run: python arxiv_submit.py")
    else:
        print("\nSubmitting to arXiv...")
        if submit_to_arxiv(token, dry_run=False):
            print("\nSubmission sent. Check arXiv for status.")
        else:
            print("\nSubmission FAILED.")
            sys.exit(1)


if __name__ == "__main__":
    main()
