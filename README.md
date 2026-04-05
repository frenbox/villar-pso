# villar-pso

Joint two-band Villar light-curve fitter using Particle Swarm Optimisation (PSO).

Backends:

* CPU (default)
* CUDA (`feature = "cuda"`, non-macOS targets)
* Metal (`feature = "metal"`, macOS only)

Backend feature policy:

* `cuda` and `metal` are mutually exclusive.
* `cuda` is rejected for macOS targets.
* `metal` is rejected for non-macOS targets.

## Installation

### CPU-only

```toml
[dependencies]
villar-pso = { git = "https://github.com/frenbox/villar-pso.git" }
```

### CUDA (NVIDIA)

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["cuda"] }
```

Requirements:

* NVIDIA GPU and drivers
* CUDA toolkit
* `nvcc` available on PATH (build fails with a clear error if not found)

### Metal (macOS)

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["metal"] }
```

Requirements:

* macOS
* Apple Metal support

## Core API

### CPU fitting

* `fit_lightcurve(path)`
* `fit_photometry(points)`

Config-aware variants:

* `fit_lightcurve_with_config(path, &PsoConfig)`
* `fit_photometry_with_config(points, &PsoConfig)`

### GPU fitting

GPU API is under `villar_pso::gpu`:

* `GpuContext::new(device)` chooses default backend for the enabled build.
* `GpuContext::new_with_backend(device, GpuBackend::Cuda|Metal)` chooses explicitly.
* `pack_batch(...)` uploads batched sources.
* `batch_pso_multi_seed(...)` runs multi-seed fitting.

Backend selection notes:

* Exactly one GPU backend feature may be enabled.
* `GpuContext::new(...)` selects the enabled backend for the build.

## Multi-Seed Strategy

`PsoConfig` includes `multi_seed_strategy`:

* `MultiSeedStrategy::EarlyStop` (default)
: Runs seeds in order and stops once improvement stabilizes.
* `MultiSeedStrategy::TryAll`
: Runs all seeds in `MULTI_SEEDS` and keeps the best result.

Applies consistently across CPU, CUDA, and Metal.

Example:

```rust
use villar_pso::{fit_lightcurve_with_config, MultiSeedStrategy, PsoConfig};

let config = PsoConfig {
    multi_seed_strategy: MultiSeedStrategy::TryAll,
    ..PsoConfig::default()
};

let fit = fit_lightcurve_with_config("/path/to/source.csv", &config)?;
println!("reduced chi2 = {}", fit.reduced_chi2);
# Ok::<(), String>(())
```

## Numerical Precision by Backend

Precision differs by backend:

* CPU: `f64`
* CUDA: `double`/`f64`
* Metal: `float`/`f32`

Metal uses `float` because general fp64 (`double`) support is not available in Metal shader targets.
The host side converts between `f64` and `f32` at the Metal boundary.

Practical implication: backend outputs are expected to be close, but not bit-identical.

## Python Bindings (PyO3)

Enable with:

```toml
villar-pso = { git = "https://github.com/frenbox/villar-pso.git", features = ["python"] }
```

Build via [maturin](https://www.maturin.rs/):

```bash
maturin develop --release --features python
maturin develop --release --features "python,cuda"
maturin develop --release --features "python,metal"
```

## Tests

Common test commands:

```bash
cargo test
cargo test --features metal
cargo test --features cuda
```

Notes:

* CUDA tests are feature-gated and require CUDA toolchain/runtime.
* Metal tests are feature-gated; runtime path is macOS-specific.
* The suite includes parity checks (CPU vs GPU), strategy checks (EarlyStop vs TryAll), smoke tests, and kernel-layout checks.

## Benchmarks

Three benchmark binaries are included:

| Binary | Purpose |
| ------ | ------- |
| `cpu-dispatch-bench` | Compares sequential vs chunked CPU dispatch over Rayon thread counts |
| `gpu-scaling-bench` | Measures CUDA scaling from 1..N GPUs |
| `metal-scaling-bench` | Runs Metal backend scaling scaffold on macOS |

Run locally:

```bash
cargo build --release --bin cpu-dispatch-bench
./target/release/cpu-dispatch-bench /path/to/photometry/

CUDA_HOME=/usr/local/cuda cargo build --release --features cuda --bin gpu-scaling-bench
./target/release/gpu-scaling-bench /path/to/photometry/

cargo build --release --features metal --bin metal-scaling-bench
./target/release/metal-scaling-bench /path/to/photometry/
```

### Apptainer (CUDA)

```bash
apptainer build --fakeroot rustgp.sif rustgp.def

apptainer exec --nv \
  --bind /path/to/project:/path/to/project \
  rustgp.sif \
  /app/gpu-scaling-bench /path/to/photometry/
```

Optional GPU limit:

```bash
apptainer exec --nv \
  --bind /path/to/project:/path/to/project \
  rustgp.sif \
  /app/gpu-scaling-bench /path/to/photometry/ --gpus 4
```

## Project Structure

```text
villar-pso/
├── Cargo.toml
├── build.rs
├── rustgp.def
├── cuda/
│   └── villar_joint.cu
├── metal/
│   └── villar_joint.metal
├── tests/
│   ├── synthetic_lightcurves.rs
│   ├── metal_kernel_layout.rs
│   └── cuda_kernel_layout.rs
└── src/
    ├── lib.rs
    ├── gpu/
    │   ├── mod.rs
    │   ├── cuda.rs
    │   ├── metal.rs
    │   └── host_shared.rs
    └── bin/
        ├── cpu_dispatch_bench.rs
        ├── gpu_scaling_bench.rs
        └── metal_scaling_bench.rs
```
