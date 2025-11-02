#!/usr/bin/env bash
set -euo pipefail
echo "Creating conda env 'cholera-demo' from environment.yml (using conda-forge)…"
conda env create -f environment.yml || conda env update -f environment.yml --prune
echo "Done. Activate with: conda activate cholera-demo"
