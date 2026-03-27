#!/bin/bash
# Run CPU and GPU scaling benchmarks via Apptainer
set -e

PROJECT_DIR="/users/5/ssharmac/lc_fitting_comparison"
DATA_DIR="${PROJECT_DIR}/data/photometry"
SIF="${PROJECT_DIR}/villar-pso/rustgp.sif"

if [ ! -f "${SIF}" ]; then
    echo "Container not found: ${SIF}"
    echo "Build it first: cd ${PROJECT_DIR} && apptainer build --fakeroot rustgp.sif villar-pso/rustgp.def"
    exit 1
fi

echo "============================================"
echo "  CPU Scaling Benchmark (villar-pso)"
echo "============================================"
apptainer exec \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    ${SIF} \
    /app/cpu-scaling-bench "${DATA_DIR}"

echo ""
echo "============================================"
echo "  GPU Scaling Benchmark (villar-pso)"
echo "============================================"
apptainer exec --nv \
    --bind ${PROJECT_DIR}:${PROJECT_DIR} \
    ${SIF} \
    /app/gpu-scaling-bench "${DATA_DIR}"
