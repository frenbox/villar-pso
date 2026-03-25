//! GPU-accelerated batch PSO for the joint two-band Villar model.
//!
//! The cost function (model eval + band-balanced likelihood + priors) runs on
//! the GPU. The swarm update loop runs on the CPU.

use std::ffi::c_int;
use std::mem::size_of;
use std::ptr;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{
    offsets_to_absolute, preprocess, reduced_chi2, to_physical, build_param_map,
    BandIndices, PreprocessedData, PriorArrays, PsoConfig,
    N_PARAMS,
};

// ---------------------------------------------------------------------------
// CUDA FFI
// ---------------------------------------------------------------------------

type CudaResult = c_int;

extern "C" {
    fn cudaSetDevice(device: c_int) -> CudaResult;
    fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> CudaResult;
    fn cudaFree(devPtr: *mut u8) -> CudaResult;
    fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: c_int) -> CudaResult;
    fn cudaDeviceSynchronize() -> CudaResult;
    fn cudaGetLastError() -> CudaResult;
    fn cudaGetErrorString(error: CudaResult) -> *const i8;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

extern "C" {
    fn launch_batch_pso_cost_villar_joint(
        all_times: *const f64,
        all_flux: *const f64,
        all_flux_err_sq: *const f64,
        all_band: *const c_int,
        source_offsets: *const c_int,
        n_r_per_source: *const c_int,
        n_g_per_source: *const c_int,
        positions: *const f64,
        costs: *mut f64,
        prior_means: *const f64,
        prior_stds: *const f64,
        logged_mask: *const c_int,
        n_sources: c_int,
        n_particles: c_int,
        grid: c_int,
        block: c_int,
    );

}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

fn cuda_check(code: CudaResult) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                "unknown CUDA error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Err(format!("CUDA error {}: {}", code, msg))
    }
}

struct DevBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    size: usize,
}

impl DevBuf {
    fn alloc(size: usize) -> Result<Self, String> {
        let mut ptr: *mut u8 = ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, size) })?;
        Ok(Self { ptr, size })
    }

    fn upload<T>(data: &[T]) -> Result<Self, String> {
        let bytes = data.len() * size_of::<T>();
        let buf = Self::alloc(bytes)?;
        cuda_check(unsafe {
            cudaMemcpy(buf.ptr, data.as_ptr() as *const u8, bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        })?;
        Ok(buf)
    }

    fn download_into<T>(&self, host: &mut [T]) -> Result<(), String> {
        let bytes = host.len() * size_of::<T>();
        cuda_check(unsafe {
            cudaMemcpy(host.as_mut_ptr() as *mut u8, self.ptr, bytes, CUDA_MEMCPY_DEVICE_TO_HOST)
        })
    }

    fn upload_from<T>(&self, data: &[T]) -> Result<(), String> {
        let bytes = data.len() * size_of::<T>();
        cuda_check(unsafe {
            cudaMemcpy(self.ptr, data.as_ptr() as *const u8, bytes, CUDA_MEMCPY_HOST_TO_DEVICE)
        })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr); }
    }
}

// ---------------------------------------------------------------------------
// Batch data
// ---------------------------------------------------------------------------

/// Preprocessed data for one source, ready for GPU packing.
pub struct SourceData {
    pub name: String,
    pub data: PreprocessedData,
}

impl AsRef<SourceData> for SourceData {
    fn as_ref(&self) -> &SourceData { self }
}

/// Packed observation data resident on the GPU.
pub struct GpuBatchData {
    d_times: DevBuf,
    d_flux: DevBuf,
    d_flux_err_sq: DevBuf,
    d_band: DevBuf,
    d_offsets: DevBuf,
    d_n_r: DevBuf,
    d_n_g: DevBuf,
    pub n_sources: usize,
    /// Per-source orig_size (for CPU-side reduced_chi2 after PSO).
    pub h_orig_sizes: Vec<usize>,
}

impl GpuBatchData {
    /// Pack and upload preprocessed sources to the GPU.
    pub fn new<S: AsRef<SourceData>>(sources: &[S]) -> Result<Self, String> {
        let n_sources = sources.len();
        let mut all_times = Vec::new();
        let mut all_flux = Vec::new();
        let mut all_flux_err_sq = Vec::new();
        let mut all_band: Vec<c_int> = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_sources + 1);
        let mut n_r_vec: Vec<c_int> = Vec::with_capacity(n_sources);
        let mut n_g_vec: Vec<c_int> = Vec::with_capacity(n_sources);
        let mut h_orig_sizes = Vec::with_capacity(n_sources);

        offsets.push(0);
        for src in sources {
            let d = &src.as_ref().data;
            let bi = BandIndices::new(&d.obs, d.orig_size);
            for ob in d.obs.iter().take(d.orig_size) {
                all_times.push(ob.phase);
                all_flux.push(ob.flux);
                all_flux_err_sq.push(ob.flux_err * ob.flux_err);
                all_band.push(ob.band as c_int);
            }
            offsets.push(all_times.len() as c_int);
            n_r_vec.push(bi.r_indices.len() as c_int);
            n_g_vec.push(bi.g_indices.len() as c_int);
            h_orig_sizes.push(d.orig_size);
        }

        Ok(Self {
            d_times: DevBuf::upload(&all_times)?,
            d_flux: DevBuf::upload(&all_flux)?,
            d_flux_err_sq: DevBuf::upload(&all_flux_err_sq)?,
            d_band: DevBuf::upload(&all_band)?,
            d_offsets: DevBuf::upload(&offsets)?,
            d_n_r: DevBuf::upload(&n_r_vec)?,
            d_n_g: DevBuf::upload(&n_g_vec)?,
            n_sources,
            h_orig_sizes,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU context + batch PSO
// ---------------------------------------------------------------------------

pub struct GpuContext {
    _device: i32,
}

/// Result for one source from batch PSO.
#[derive(Debug, Clone)]
pub struct BatchPsoResult {
    /// Best raw parameters (14 values: r-band + g-band offsets).
    pub raw_params: [f64; N_PARAMS],
    /// Physical parameters (14 values, exponentiated where needed).
    pub phys_params: [f64; N_PARAMS],
    /// Reduced chi-squared.
    pub reduced_chi2: f64,
    /// PSO cost (includes priors + band balancing).
    pub cost: f64,
}

impl GpuContext {
    pub fn new(device: i32) -> Result<Self, String> {
        cuda_check(unsafe { cudaSetDevice(device) })?;
        Ok(Self { _device: device })
    }

    /// Run batch PSO for all sources simultaneously on the GPU.
    ///
    /// The entire PSO loop (cost eval, velocity/position update, best reduction)
    /// runs on GPU. Only a small `source_done` array is downloaded each iteration
    /// for convergence checking.
    pub fn batch_pso<S: AsRef<SourceData>>(
        &self,
        data: &GpuBatchData,
        sources: &[S],
        config: &PsoConfig,
        seed: u64,
    ) -> Result<Vec<BatchPsoResult>, String> {
        let n_sources = data.n_sources;
        let dim = N_PARAMS;
        let n_particles = config.n_particles;
        let total_particles = n_sources * n_particles;

        let priors = PriorArrays::new();

        // Upload prior arrays for GPU cost kernel
        let prior_means: Vec<f64> = priors.means.to_vec();
        let prior_stds: Vec<f64> = priors.stds.to_vec();
        let logged_mask: Vec<c_int> = priors.logged.iter().map(|&b| b as c_int).collect();
        let d_prior_means = DevBuf::upload(&prior_means)?;
        let d_prior_stds = DevBuf::upload(&prior_stds)?;
        let d_logged_mask = DevBuf::upload(&logged_mask)?;

        // PSO bounds and velocity clamp (CPU-side only)
        let lower: [f64; N_PARAMS] = priors.mins;
        let upper: [f64; N_PARAMS] = priors.maxs;
        let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

        let w_start = config.w;
        let w_end = 0.4;
        let c1 = config.c1;
        let c2 = config.c2;
        let inv_max_iters = 1.0 / config.max_iters as f64;

        // Initialize particles on CPU
        let mut h_positions = vec![0.0f64; total_particles * dim];
        let mut h_velocities = vec![0.0f64; total_particles * dim];

        let n_seeded = n_particles * 3 / 10;
        for s in 0..n_sources {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(s as u64));
            for p in 0..n_particles {
                let idx = s * n_particles + p;
                let base = idx * dim;
                if p < n_seeded {
                    for d in 0..dim {
                        let u1: f64 = rng.random::<f64>().max(1e-10);
                        let u2: f64 = rng.random::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt()
                            * (2.0 * std::f64::consts::PI * u2).cos();
                        h_positions[base + d] =
                            (priors.means[d] + z * priors.stds[d]).clamp(lower[d], upper[d]);
                    }
                } else {
                    for d in 0..dim {
                        h_positions[base + d] =
                            lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                    }
                }
                for d in 0..dim {
                    h_velocities[base + d] =
                        v_max[d] * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
                }
            }
        }

        // Allocate all GPU buffers (persist for entire PSO loop)
        // GPU buffers: only positions (uploaded each iter) and costs (downloaded each iter)
        let d_positions = DevBuf::alloc(total_particles * dim * size_of::<f64>())?;
        let d_costs = DevBuf::alloc(total_particles * size_of::<f64>())?;

        let block: c_int = 256;
        let particle_grid: c_int = ((total_particles as c_int) + block - 1) / block;

        // Hybrid approach: GPU evaluates cost, CPU handles PSO dynamics.
        // This gives us the CPU's proven restart/exploration strategy
        // while offloading the expensive cost function to GPU.

        let mut h_pbest_pos = h_positions.clone();
        let mut h_pbest_cost = vec![f64::INFINITY; total_particles];
        let mut h_gbest_pos = vec![0.0f64; n_sources * dim];
        let mut h_gbest_cost = vec![f64::INFINITY; n_sources];
        let mut h_costs = vec![0.0f64; total_particles];

        // Per-source stall tracking
        let mut prev_gbest = vec![f64::INFINITY; n_sources];
        let mut stall_count = vec![0usize; n_sources];
        let mut source_done = vec![false; n_sources];

        let restart_threshold = 40usize;
        let restart_fraction = 0.3;

        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(n_sources as u64 + 1));

        for iter in 0..config.max_iters {
            let w = w_start - (w_start - w_end) * (iter as f64) * inv_max_iters;

            // 1. Upload positions, evaluate costs on GPU
            d_positions.upload_from(&h_positions)?;
            unsafe {
                launch_batch_pso_cost_villar_joint(
                    data.d_times.ptr as _, data.d_flux.ptr as _,
                    data.d_flux_err_sq.ptr as _, data.d_band.ptr as _,
                    data.d_offsets.ptr as _, data.d_n_r.ptr as _, data.d_n_g.ptr as _,
                    d_positions.ptr as _, d_costs.ptr as _,
                    d_prior_means.ptr as _, d_prior_stds.ptr as _, d_logged_mask.ptr as _,
                    n_sources as c_int, n_particles as c_int,
                    particle_grid, block,
                );
                cuda_check(cudaGetLastError())?;
                cuda_check(cudaDeviceSynchronize())?;
            }
            d_costs.download_into(&mut h_costs)?;

            // 2. CPU: update personal bests, global bests, velocities, positions
            let mut all_done = true;
            for s in 0..n_sources {
                if source_done[s] { continue; }

                // Update pbest and gbest
                for p in 0..n_particles {
                    let idx = s * n_particles + p;
                    let cost = h_costs[idx];
                    if cost < h_pbest_cost[idx] {
                        h_pbest_cost[idx] = cost;
                        let base = idx * dim;
                        h_pbest_pos[base..base + dim]
                            .copy_from_slice(&h_positions[base..base + dim]);
                        if cost < h_gbest_cost[s] {
                            h_gbest_cost[s] = cost;
                            let gb = s * dim;
                            h_gbest_pos[gb..gb + dim]
                                .copy_from_slice(&h_positions[base..base + dim]);
                        }
                    }
                }

                // Stall detection
                let improved = prev_gbest[s] - h_gbest_cost[s]
                    > 1e-4 * prev_gbest[s].abs().max(1e-10);
                if improved {
                    stall_count[s] = 0;
                    prev_gbest[s] = h_gbest_cost[s];
                } else {
                    stall_count[s] += 1;

                    // Partial restart: reinitialize worst particles
                    if stall_count[s] % restart_threshold == 0
                        && stall_count[s] < config.stall_iters
                    {
                        let n_restart = (n_particles as f64 * restart_fraction) as usize;
                        let mut indices: Vec<usize> = (0..n_particles).collect();
                        indices.sort_by(|&a, &b| {
                            let ia = s * n_particles + a;
                            let ib = s * n_particles + b;
                            h_pbest_cost[ib].partial_cmp(&h_pbest_cost[ia])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for (ri, &p) in indices.iter().take(n_restart).enumerate() {
                            let idx = s * n_particles + p;
                            let base = idx * dim;
                            for d in 0..dim {
                                if ri < n_restart / 2 {
                                    let u1: f64 = rng.random::<f64>().max(1e-10);
                                    let u2: f64 = rng.random::<f64>();
                                    let z = (-2.0 * u1.ln()).sqrt()
                                        * (2.0 * std::f64::consts::PI * u2).cos();
                                    h_positions[base + d] = (priors.means[d] + z * priors.stds[d])
                                        .clamp(lower[d], upper[d]);
                                } else {
                                    h_positions[base + d] = lower[d]
                                        + rng.random::<f64>() * (upper[d] - lower[d]);
                                }
                                h_velocities[base + d] = (upper[d] - lower[d])
                                    * 0.1 * (2.0 * rng.random::<f64>() - 1.0);
                            }
                            h_pbest_cost[idx] = f64::INFINITY;
                        }
                    }

                    if stall_count[s] >= config.stall_iters {
                        source_done[s] = true;
                    }
                }

                if !source_done[s] {
                    all_done = false;

                    // Update velocities and positions
                    for p in 0..n_particles {
                        let idx = s * n_particles + p;
                        let base = idx * dim;
                        let gb = s * dim;
                        for d in 0..dim {
                            let r1: f64 = rng.random();
                            let r2: f64 = rng.random();
                            let mut v = w * h_velocities[base + d]
                                + c1 * r1 * (h_pbest_pos[base + d] - h_positions[base + d])
                                + c2 * r2 * (h_gbest_pos[gb + d] - h_positions[base + d]);
                            v = v.clamp(-v_max[d], v_max[d]);
                            let new_pos = h_positions[base + d] + v;
                            if new_pos <= lower[d] {
                                h_positions[base + d] = lower[d];
                                h_velocities[base + d] = 0.0;
                            } else if new_pos >= upper[d] {
                                h_positions[base + d] = upper[d];
                                h_velocities[base + d] = 0.0;
                            } else {
                                h_positions[base + d] = new_pos;
                                h_velocities[base + d] = v;
                            }
                        }
                    }
                }
            }
            if all_done { break; }
        }

        // Collect results from CPU-side gbest arrays
        let results: Vec<BatchPsoResult> = (0..n_sources)
            .map(|s| {
                let gb = s * dim;
                let mut raw = [0.0f64; N_PARAMS];
                raw.copy_from_slice(&h_gbest_pos[gb..gb + dim]);

                let abs_raw = offsets_to_absolute(&raw);
                let phys = to_physical(&raw, &priors);

                let d = &sources[s].as_ref().data;
                let param_map = build_param_map(&d.obs);
                let rchi2 = reduced_chi2(&raw, &d.obs, &param_map, d.orig_size, &priors);

                BatchPsoResult {
                    raw_params: abs_raw,
                    phys_params: phys,
                    reduced_chi2: rchi2,
                    cost: h_gbest_cost[s],
                }
            })
            .collect();

        Ok(results)
    }

    /// Run batch PSO with multiple seeds (matching CPU's 3-seed strategy).
    /// Returns the best result per source across all seeds.
    pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
        &self,
        data: &GpuBatchData,
        sources: &[S],
        config: &PsoConfig,
    ) -> Result<Vec<BatchPsoResult>, String> {
        let seeds: [u64; 3] = [42, 137, 271];
        let mut best: Option<Vec<BatchPsoResult>> = None;

        for &seed in &seeds {
            let results = self.batch_pso(data, sources, config, seed)?;
            best = Some(match best {
                None => results,
                Some(prev) => prev.into_iter().zip(results).map(|(a, b)| {
                    if b.cost < a.cost { b } else { a }
                }).collect(),
            });
        }
        Ok(best.unwrap())
    }
}

// ---------------------------------------------------------------------------
// Convenience: load a directory of CSVs and run batch GPU PSO
// ---------------------------------------------------------------------------

/// Load all CSVs from a directory, preprocess them, and return source data.
/// Sources that fail preprocessing are skipped (with a warning on stderr).
pub fn load_sources(data_dir: &str) -> Vec<SourceData> {
    let mut entries: Vec<_> = std::fs::read_dir(data_dir)
        .expect("Cannot read data directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "csv")
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut sources = Vec::new();
    for entry in entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let csv_path = path.to_string_lossy().to_string();
        match preprocess(&csv_path) {
            Ok(data) => sources.push(SourceData { name, data }),
            Err(e) => eprintln!("SKIP {}: {}", name, e),
        }
    }
    sources
}
