#!/bin/bash
# Run GPU batch PSO fitting on all 500 CSVs at once
set -e

PROJECT_DIR="/users/5/ssharmac/lc_fitting_comparison"
DATA_DIR="${PROJECT_DIR}/data/photometry"
OUTPUT_CSV="${PROJECT_DIR}/src/lc_fitting_comparison/comparison_results/villar_pso_gpu.csv"
SIF="${PROJECT_DIR}/rustgp.sif"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo "Running gpu-batch on all CSVs in ${DATA_DIR}..."

apptainer exec --nv \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    ${SIF} \
    /app/gpu-batch "${DATA_DIR}" \
        --output "${OUTPUT_CSV}"

echo "Results written to ${OUTPUT_CSV}"
