#!/bin/bash
set -e  # Exit on error
set -o pipefail

BUILD_DIR="cmake-build-debug"
LOG_DIR="logs/"
EXECUTABLE="${BUILD_DIR}/project_repo"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_log_${TIMESTAMP}.txt"

echo "=== Building Project ==="

mkdir -p "$BUILD_DIR"
mkdir -p "$LOG_DIR"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug

cmake --build "$BUILD_DIR" -- -j$(nproc)

echo "=== Running Program ==="
echo "Logging output to $LOG_FILE"
echo

"$EXECUTABLE" |& tee "$LOG_DIR/$LOG_FILE"

echo
echo "=== Done. Log saved to $LOG_FILE ==="
