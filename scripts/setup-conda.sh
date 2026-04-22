#!/usr/bin/env bash
set -euo pipefail
echo "Creating conda env 'cholera-demo' from conda-environment.yml (using conda-forge)..."
conda env create -f conda-environment.yml || conda env update -f conda-environment.yml --prune
echo "Done. Activate with: conda activate cholera-demo"
