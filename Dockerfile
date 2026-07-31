# Archimedes Project: Dockerfile for the H10 and v8 dlr_only
# experiments. Build and run with:
#   docker build -t archimedes .
#   docker run -it --rm -v $(pwd):/work -w /work archimedes bash
#
# On Windows:
#   docker build -t archimedes .
#   docker run -it --rm -v ${PWD}:/work -w /work archimedes bash
#
# The image is based on python:3.10-slim, includes TeX Live for
# paper PDF building, and exposes the full repo for reproducing
# any experiment via Make.

FROM python:3.10-slim

# System dependencies for TeX Live (paper PDFs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-xetex \
    texlive-luatex \
    pandoc \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install Python dependencies (cache as a separate layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# Default to a help message
CMD ["make", "help"]
