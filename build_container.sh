#!/bin/bash
# Build the Apptainer container locally (no SLURM needed)
set -e

cd /users/5/ssharmac/lc_fitting_comparison

echo "Building Apptainer image..."
apptainer build --fakeroot rustgp.sif rustgp.def

echo "Verifying..."
apptainer exec rustgp.sif python3 -c "import lightcurve_fitting as lf; print('lightcurve_fitting OK')"

echo "Done. Container: rustgp.sif"
