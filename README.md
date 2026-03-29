# villar-pso

Joint two-band Villar light-curve fitter using Particle Swarm Optimisation (PSO).

## Using as a dependency

Add to your project's `Cargo.toml`:

### Without GPU (CPU only)

```toml
[dependencies]
villar-pso = { git = "https://github.com/frenbox/villar-pso.git" }
```

No feature flags needed. This gives you the core library (`fit_lightcurve`,
`preprocess`, etc.) and Rayon-based CPU parallelism.

### With GPU (CUDA)

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["cuda"] }
```

Requires `nvcc` and CUDA toolkit on the build machine. This enables the `gpu`
module (`GpuContext`, `GpuBatchData`, `batch_pso_multi_seed`, etc.).

### Python bindings (PyO3)

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["python"] }
```

Build with [maturin](https://www.maturin.rs/):

```bash
maturin develop --release --features python         # CPU only
maturin develop --release --features "python,cuda"   # CPU + GPU
```

## Benchmarks

Two benchmark binaries are included:

| Binary | What it tests |
|--------|--------------|
| `cpu-dispatch-bench` | Compares sequential dispatch (`par_iter`, 1 file at a time) vs batch dispatch (`par_chunks(500)`) across 2, 4, 6, 8, 10 Rayon threads |
| `gpu-scaling-bench` | Scales from 1 to N GPUs, one Rayon thread per GPU, processing all sources in chunks of 500 |

### Running benchmarks locally

**CPU benchmark (no GPU needed):**

```bash
cargo build --release --bin cpu-dispatch-bench
./target/release/cpu-dispatch-bench /path/to/photometry/
```

**GPU benchmark (requires CUDA):**

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda --bin gpu-scaling-bench
./target/release/gpu-scaling-bench /path/to/photometry/
```

### Running benchmarks via Apptainer

For machines where CUDA is available only through containers.

**1. Build the container** (from the `villar-pso/` directory):

```bash
apptainer build --fakeroot rustgp.sif rustgp.def
```

**2. Run CPU benchmark:**

```bash
apptainer exec \
    --bind /path/to/project:/path/to/project \
    rustgp.sif \
    /app/cpu-dispatch-bench /path/to/photometry/
```

**3. Run GPU benchmark:**

```bash
apptainer exec --nv \
    --bind /path/to/project:/path/to/project \
    rustgp.sif \
    /app/gpu-scaling-bench /path/to/photometry/
```

The `--nv` flag exposes host NVIDIA drivers to the container.

**4. Limit GPU count** (optional):

```bash
apptainer exec --nv \
    --bind /path/to/project:/path/to/project \
    rustgp.sif \
    /app/gpu-scaling-bench /path/to/photometry/ --gpus 4
```

## Project structure

```
villar-pso/
├── Cargo.toml
├── build.rs              # CUDA kernel compilation (when cuda feature enabled)
├── rustgp.def            # Apptainer container definition
├── cuda/
│   └── villar_joint.cu   # GPU cost function kernel
└── src/
    ├── lib.rs            # Core library: preprocessing, PSO, fitting, PyO3 bindings
    ├── gpu.rs            # GPU batch PSO, multi-GPU support
    └── bin/
        ├── cpu_dispatch_bench.rs
        └── gpu_scaling_bench.rs
```
