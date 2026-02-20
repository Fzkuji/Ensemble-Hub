#!/bin/bash
# Simple Latency Measurement for Tandem (Reviewer yUBt W5)
#
# Usage:
#   ./run_latency.sh                                   # Use pre-measured times (no GPU needed)
#   ./run_latency.sh --benchmark --intern-gpus 4       # Re-measure 7B generation
#   ./run_latency.sh --benchmark --intern-gpus 4,5,6,7 --mentor-gpus 0,1,2,3  # Full re-measure
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python eval_latency.py "$@"
