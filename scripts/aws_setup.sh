#!/usr/bin/env bash
# AWS EC2 setup script for cf_benchmark
# Run this after SSH-ing into a fresh Ubuntu 22.04/24.04 instance.
#
# Usage:
#   chmod +x scripts/aws_setup.sh
#   ./scripts/aws_setup.sh
#
# After setup, run experiments with:
#   source .venv/bin/activate
#   nohup python -m scripts.run_parallel --scenario lean --workers 12 --timeout 7200 \
#         --exclude heloc,gs 2>&1 | tee experiment_run.log &

set -euo pipefail

echo "=== System update ==="
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip curl

echo "=== Clone repo ==="
if [ ! -d "cf_benchmark" ]; then
    git clone https://github.com/creaturerigger/cf_benchmark.git
fi
cd cf_benchmark

echo "=== Create venv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Install dependencies ==="
pip install --upgrade pip
pip install -e ".[dev]" 2>/dev/null || pip install -e .

echo "=== Install nicex (--no-deps to bypass stale pandas constraint) ==="
pip install nicex==0.2.3 --no-deps

echo "=== Install growingspheres (no PyPI — copy from GitHub) ==="
if [ ! -d "../growingspheres" ]; then
    git clone https://github.com/thibaultlaugel/growingspheres.git ../growingspheres
fi
cp -r ../growingspheres/growingspheres .venv/lib/python3.*/site-packages/

echo "=== Install DiCE-X (local editable) ==="
if [ ! -d "../DiCE-X" ]; then
    git clone https://github.com/creaturerigger/DiCE-X.git ../DiCE-X
fi
pip install -e ../DiCE-X

echo "=== Install lore-sa (not on PyPI — install from GitHub) ==="
if [ ! -d "../LORE_sa" ]; then
    git clone https://github.com/kdd-lab/LORE_sa.git ../LORE_sa
fi
pip install -e ../LORE_sa

echo "=== Install lore_sa transitive deps (not in pyproject.toml) ==="
pip install deap==1.4.3
pip install liac-arff
pip install scikit-multilearn

echo "=== Verify imports ==="
python -c "
import growingspheres
import deap
import arff
import skmultilearn
import lore_sa
import nicex
import dice_ml
from src.orchestration.prefect_flow import run_pipeline
print('All imports OK')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Dry run — see what will execute:"
echo "  python -m scripts.run_parallel --scenario lean --dry-run"
echo ""
echo "  # Full Lean run (background, with log):"
echo "  nohup python -u -m scripts.run_parallel \\"
echo "      --scenario lean --workers \$(( \$(nproc) - 1 )) \\"
echo "      --timeout 7200 --exclude heloc,gs \\"
echo "      2>&1 | tee experiment_\$(date +%Y%m%d_%H%M).log &"
echo ""
echo "  # Monitor:"
echo "  tail -f experiment_*.log"
echo "  grep -c 'OK\|FAIL\|TIMEOUT' experiment_*.log"
echo ""
