#!/bin/bash
# Split collected data by hendrycks_math subsets
# Usage: bash run_split.sh <data_dir> <output_dir>
#
# Example:
#   bash run_split.sh /path/to/collected/hendrycks_math_all_DeepSeek-R1-Distill-Qwen-32B /path/to/output/hendrycks_math_split

set -e

# Default paths (modify as needed)
DATA_DIR="${1:-/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_all_DeepSeek-R1-Distill-Qwen-32B}"
OUTPUT_DIR="${2:-/mnt/data/zichuanfu/Ensemble-Hub/data/acte_experiments/collected/hendrycks_math_split}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Splitting data by hendrycks_math subsets"
echo "=========================================="
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# List available files
echo "Available files in data directory:"
ls -la "$DATA_DIR"
echo ""

# Run the split script
python "$SCRIPT_DIR/split_by_subset.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Done! Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Show output structure
echo ""
echo "Output structure:"
find "$OUTPUT_DIR" -type d | head -20
echo ""
echo "Sample files:"
find "$OUTPUT_DIR" -name "*.json" | head -10
