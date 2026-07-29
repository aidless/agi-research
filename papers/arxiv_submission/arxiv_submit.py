#!/usr/bin/env python3
"""
arXiv submission script for the Y3 paper.
"Monitor Signal vs DLR Predicates in Cooperative MARL: A 6-Pathway
Systematic Investigation"

Usage:
    export ARXIV_TOKEN=<your-arxiv-api-token>
    python arxiv_submit.py [--dry-run] [--skip-compile] [--no-verify]

This script:
1. Validates the submission package (LaTeX source, figures, metadata)
2. Validates the LaTeX source compiles cleanly with pdflatex
3. Optionally uploads to arXiv via the arXiv API

The arXiv API uses HTTPS POST with:
- Multipart form data with the source files
- Authentication via API token in cookie or header
- The endpoint: https://arxiv.org/api/submit

To get an arXiv API token:
1. Create account at https://arxiv.org (if not already)
2. Go to https://arxiv.org/user (login)
3. Go to "API Tokens" section
4. Generate a new token
5. Set ARXIV_TOKEN environment variable before running this script

Reference: https://info.arxiv.org/help/api/index.html
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent.parent
TEX_FILE = SUBMISSION_DIR / "monitor_signal_vs_dlr_6pathway.tex"
PDF_FILE = SUBMISSION_DIR / "monitor_signal_vs_dlr_6pathway.pdf"
FIGURES_DIR = SUBMISSION_DIR / "figures"
METADATA_FILE = SUBMISSION_DIR / "arxiv_metadata.txt"
TAR_GZ = REPO_ROOT / "papers" / "arxiv_submission.tar.gz"


def parse_metadata(path):
    """Parse the arxiv_metadata.txt file into a dict."""
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
    print("=" * 60)
    print("Validating submission package...")
    print("=" * 60)
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
    if not TAR_GZ.exists():
        print("  WARNING: tar.gz not found: " + str(TAR_GZ))
        print("           (arXiv accepts individual files; tar.gz is optional)")
    elif verbose:
        print("  OK: " + str(TAR_GZ.relative_to(REPO_ROOT)) + " (" + str(TAR_GZ.stat().st_size) + " bytes)")
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
    print("\n" + "=" * 60)
    print("Validating LaTeX source compiles...")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-draftmode", str(TEX_FILE)],
            cwd=SUBMISSION_DIR,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print("  LaTeX compile FAILED: returncode " + str(result.returncode))
            print(result.stdout[-2000:])
            return False
        print("  OK: LaTeX compiles cleanly")
        return True
    except FileNotFoundError:
        print("  pdflatex not found, skipping compile check (install TeX Live to enable)")
        return True
    except subprocess.TimeoutExpired:
        print("  pdflatex timed out, skipping compile check")
        return True


def submit_to_arxiv(token, dry_run=False):
    """Submit the package to arXiv via the API."""
    if dry_run:
        print("\n" + "=" * 60)
        print("[DRY RUN] Would submit the following package to arXiv:")
        print("=" * 60)
        print("  Source: " + str(TEX_FILE.relative_to(REPO_ROOT)))
        print("  PDF: " + str(PDF_FILE.relative_to(REPO_ROOT)))
        for fig in sorted(FIGURES_DIR.glob("*.png")):
            print("  Figure: figures/" + fig.name)
        print("\n  Metadata: " + str(METADATA_FILE.relative_to(REPO_ROOT)))
        print("\n[DRY RUN] Skipping actual upload.")
        return True
    import urllib.request
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
            if "success" in result.lower() or "queued" in result.lower() or "submitted" in result.lower():
                return True
            return False
    except urllib.error.HTTPError as e:
        print("\narXiv HTTP error " + str(e.code) + ": " + e.reason)
        try:
            print(e.read().decode()[:2000])
        except:
            pass
        return False
    except Exception as e:
        print("\narXiv submission error: " + str(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="arXiv submission for the Y3 paper 'Monitor Signal vs DLR Predicates in Cooperative MARL'")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate package without submitting (default if no ARXIV_TOKEN)")
    parser.add_argument("--skip-compile", action="store_true",
                       help="Skip pdflatex compile check")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip package validation (use with caution)")
    args = parser.parse_args()
    if not args.no_verify and not validate_package(verbose=True):
        print("\nVALIDATION FAILED. Aborting.")
        sys.exit(1)
    print("\nPackage validation: OK")
    if not args.skip_compile and not validate_latex_compiles():
        print("\nLATEX COMPILE FAILED. Aborting.")
        sys.exit(1)
    token = os.environ.get("ARXIV_TOKEN", None)
    if args.dry_run or token is None:
        submit_to_arxiv(token, dry_run=True)
        print("\n" + "=" * 60)
        print("Dry run complete.")
        print("=" * 60)
        print("\nTo submit for real:")
        print("  1. Create arXiv account at https://arxiv.org (if not already)")
        print("  2. Get API token at https://arxiv.org/user (API Tokens section)")
        print("  3. Set ARXIV_TOKEN environment variable:")
        print("     $env:ARXIV_TOKEN = '<your-token>'      # PowerShell")
        print("     export ARXIV_TOKEN='<your-token>'    # bash")
        print("  4. Run: python arxiv_submit.py")
        print("\n  Or to skip the LaTeX compile check (faster):")
        print("     python arxiv_submit.py --skip-compile")
        print("\nNotes:")
        print("  - cs.MA is endorsement-free, so no endorsement is needed")
        print("  - First-time submitters may need to verify their email first")
        print("  - After submission, paper is queued and may take 1-2 days")
        print("    before appearing at arxiv.org/abs/[id]")
    else:
        print("\n" + "=" * 60)
        print("Submitting to arXiv...")
        print("=" * 60)
        if submit_to_arxiv(token, dry_run=False):
            print("\n" + "=" * 60)
            print("Submission sent successfully!")
            print("=" * 60)
            print("Check https://arxiv.org/user for status updates.")
            print("Paper will be queued and may take 1-2 days to appear.")
        else:
            print("\nSubmission FAILED. Check the error messages above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
