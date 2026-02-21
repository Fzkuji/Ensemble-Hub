#!/bin/bash
# Wall-clock Latency Measurement for Tandem (Reviewer yUBt W5)
#
# Benchmarks LLM (32B) and SLM (7B) generation times separately,
# then computes Tandem cascade latency using paper's Table 1 data.
#
# Usage:
#   ./run_latency.sh --benchmark --mentor-gpus 0,1,2,3 --intern-gpus 4,5,6,7
#   ./run_latency.sh --benchmark-intern --intern-gpus 0,1,2,3
#   ./run_latency.sh --benchmark-mentor --mentor-gpus 0,1,2,3
#   ./run_latency.sh   # use previously saved results
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python eval_latency.py "$@"
