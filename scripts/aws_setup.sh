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

echo "=== Verify import ==="
python -c "from src.orchestration.prefect_flow import run_pipeline; print('Import OK')"

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
