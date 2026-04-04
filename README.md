# villar-pso

Joint two-band Villar light-curve fitter using Particle Swarm Optimisation (PSO).
Supports CPU-only execution, CUDA acceleration on NVIDIA/Linux hosts, and
Metal acceleration on macOS hosts.

## Using as a dependency

Add to your project's `Cargo.toml`:

### CPU only

```toml
[dependencies]
villar-pso = { git = "https://github.com/frenbox/villar-pso.git" }
```

No feature flags needed. This gives you the core library (`fit_lightcurve`,
`preprocess`, etc.) and Rayon-based CPU parallelism.

### CUDA on Linux / NVIDIA

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["cuda"] }
```

Requires `nvcc`, the CUDA toolkit, and NVIDIA drivers on the build machine.
This enables the GPU backend on Linux and other CUDA-capable systems.

### Metal on macOS

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["metal"] }
```

Requires macOS and Apple Metal support. This enables the GPU backend on Apple
platforms.

### Using GPU features in code

The public accelerator API lives under `villar_pso::gpu` and is selected at
build time:

* `GpuContext::new(...)` uses the default backend for the enabled build.
* `GpuContext::new_with_backend(...)` lets you pick CUDA or Metal explicitly.
* When both GPU features are enabled, CUDA is preferred as the default backend.

### Python bindings (PyO3)

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["python"] }
```

Build with [maturin](https://www.maturin.rs/):

```bash
maturin develop --release --features python         # CPU only
maturin develop --release --features "python,cuda"   # CPU + CUDA (Linux/NVIDIA)
maturin develop --release --features "python,metal"  # CPU + Metal (macOS)
```

## Benchmarks

Two benchmark binaries are included:

| Binary | What it tests |
| ------ | ------------- |
| `cpu-dispatch-bench` | Compares sequential dispatch (`par_iter`, 1 file at a time) vs batch dispatch (`par_chunks(500)`) across 2, 4, 6, 8, 10 Rayon threads |
| `gpu-scaling-bench` | Scales CUDA execution from 1 to N GPUs, one Rayon thread per GPU, processing all sources in chunks of 500 |
| `metal-scaling-bench` | Runs the Metal backend benchmark on macOS (single-device scaffold for now) |

### Running benchmarks locally

**CPU benchmark (no GPU needed):**

```bash
cargo build --release --bin cpu-dispatch-bench
./target/release/cpu-dispatch-bench /path/to/photometry/
```

**CUDA benchmark (Linux / NVIDIA):**

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda --bin gpu-scaling-bench
./target/release/gpu-scaling-bench /path/to/photometry/
```

**Metal benchmark (macOS):**

```bash
cargo build --release --features metal --bin metal-scaling-bench
./target/release/metal-scaling-bench /path/to/photometry/
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

Apptainer is currently documented for CUDA/NVIDIA only. Metal support is
macOS-specific and is not containerized here.

## Project structure

```text
villar-pso/
├── Cargo.toml
├── build.rs              # CUDA kernel compilation and arch detection
├── rustgp.def            # Apptainer container definition
├── cuda/
│   └── villar_joint.cu   # CUDA cost function + helper kernels
├── metal/
│   └── villar_joint.metal # Metal cost function kernel
└── src/
    ├── lib.rs            # Core library: preprocessing, PSO, fitting, PyO3 bindings
    ├── gpu/
    │   ├── mod.rs        # GPU backend dispatch layer
    │   ├── cuda.rs       # CUDA backend host orchestration
    │   ├── metal.rs      # Metal backend host orchestration
    │   └── host_shared.rs# Shared host-side PSO helpers
    └── bin/
        ├── cpu_dispatch_bench.rs
        ├── gpu_scaling_bench.rs
        └── metal_scaling_bench.rs
```
