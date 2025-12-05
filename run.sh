#!/bin/bash
set -e # exit when error 
set -o pipefail # exit when error in pipeline

export OMP_NUM_THREADS=8 # too many threads will slow it down

BUILD_DIR="cmake-build-debug"
LOG_DIR="logs/"
EXECUTABLE="${BUILD_DIR}/neural_network"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_log_${TIMESTAMP}.txt"

echo "Cleaning Old Build files"
rm -rf "$BUILD_DIR"  
rm -rf "$LOG_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$LOG_DIR"


echo "Building Project"

mkdir -p "$BUILD_DIR"
mkdir -p "$LOG_DIR"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" -- -j$(nproc)

echo " Running Program "
echo "Logging output to $LOG_FILE"
echo

"$EXECUTABLE" |& tee "$LOG_DIR/$LOG_FILE"

echo
echo "Done. Log saved to $LOG_FILE"
