//! Metal-accelerated batch PSO for the joint two-band Villar model.
//!
//! This is the macOS counterpart to [`crate::gpu`]. It evaluates the cost
//! function on Apple Silicon GPUs (M1–M5) via Metal and keeps the PSO swarm
//! dynamics on the CPU — the same hybrid layout the CUDA path uses.
//!
//! Apple GPUs don't support fp64, so device-side computation runs in fp32 with
//! Kahan-corrected reductions inside the shader (see `metal/villar_joint.metal`).
//! The host API is f64-in / f64-out so this module is a drop-in replacement
//! for [`crate::gpu`] from the caller's perspective.
//!
//! Public surface deliberately matches `crate::gpu`:
//!   * [`SourceData`], [`GpuBatchData`], [`GpuContext`]
//!   * [`GpuContext::batch_pso`], [`GpuContext::batch_pso_multi_seed`]
//!   * [`load_sources`]

use std::ffi::c_int;
use std::mem::size_of;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLResourceOptions,
    MTLSize,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{
    build_param_map, preprocess, reduced_chi2, to_physical, BandIndices, FitResult, Obs,
    PreprocessedData, PriorArrays, PsoConfig, VillarParams, N_PARAMS,
};

const KERNEL_SRC: &str = include_str!("../metal/villar_joint.metal");
const KERNEL_FN: &str = "batch_pso_cost_villar_joint";
const THREADGROUP: u64 = 256;

/// Inline scalar-args struct passed to the kernel via `set_bytes`.
/// Layout must match `struct CostParams` in the .metal file.
#[repr(C)]
#[derive(Copy, Clone)]
struct CostParams {
    n_sources: c_int,
    n_particles: c_int,
}

// ---------------------------------------------------------------------------
// SourceData (mirrors gpu::SourceData)
// ---------------------------------------------------------------------------

/// Preprocessed data for one source, ready for GPU packing.
pub struct SourceData {
    pub name: String,
    pub data: PreprocessedData,
}

impl AsRef<SourceData> for SourceData {
    fn as_ref(&self) -> &SourceData {
        self
    }
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Allocate a shared-storage Metal buffer and copy `data` (interpreted as raw
/// bytes) into it. `StorageModeShared` is the right choice on Apple Silicon —
/// the buffer is visible to both CPU and GPU through unified memory.
fn upload_bytes<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = (data.len() * size_of::<T>()) as u64;
    if bytes == 0 {
        // Metal rejects zero-byte buffers; allocate one element to keep the
        // binding valid even when no data is present.
        return device.new_buffer(size_of::<T>() as u64, MTLResourceOptions::StorageModeShared);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        bytes,
        MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_shared(device: &Device, bytes: usize) -> Buffer {
    let n = bytes.max(1) as u64;
    device.new_buffer(n, MTLResourceOptions::StorageModeShared)
}

/// Copy `src` into `buf`'s shared-memory backing store. Caller guarantees the
/// buffer is at least `src.len() * size_of::<T>()` bytes.
unsafe fn write_buffer<T: Copy>(buf: &Buffer, src: &[T]) {
    let dst = buf.contents() as *mut T;
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
}

/// Borrow `buf`'s shared-memory contents as a `&[T]` of `len` elements.
unsafe fn read_buffer<T: Copy>(buf: &Buffer, len: usize) -> &[T] {
    let ptr = buf.contents() as *const T;
    std::slice::from_raw_parts(ptr, len)
}

// ---------------------------------------------------------------------------
// Batch data
// ---------------------------------------------------------------------------

/// Packed observation data resident on the GPU.
///
/// Mirrors [`crate::gpu::GpuBatchData`]. Note the constructor signature
/// differs slightly from the CUDA version: it takes the [`GpuContext`]
/// because Metal needs the device handle to allocate buffers (CUDA uses an
/// implicit current device).
pub struct GpuBatchData {
    d_times: Buffer,
    d_flux: Buffer,
    d_flux_err_sq: Buffer,
    d_band: Buffer,
    d_offsets: Buffer,
    d_n_r: Buffer,
    d_n_g: Buffer,
    pub n_sources: usize,
    /// Per-source orig_size (for CPU-side reduced_chi2 after PSO).
    pub h_orig_sizes: Vec<usize>,
}

impl GpuBatchData {
    /// Pack and upload preprocessed sources to the GPU.
    pub fn new<S: AsRef<SourceData>>(ctx: &GpuContext, sources: &[S]) -> Result<Self, String> {
        let n_sources = sources.len();
        let mut all_times: Vec<f32> = Vec::new();
        let mut all_flux: Vec<f32> = Vec::new();
        let mut all_flux_err_sq: Vec<f32> = Vec::new();
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
                all_times.push(ob.phase as f32);
                all_flux.push(ob.flux as f32);
                all_flux_err_sq.push((ob.flux_err * ob.flux_err) as f32);
                all_band.push(ob.band.idx() as c_int);
            }
            offsets.push(all_times.len() as c_int);
            n_r_vec.push(bi.r_indices.len() as c_int);
            n_g_vec.push(bi.g_indices.len() as c_int);
            h_orig_sizes.push(d.orig_size);
        }

        Ok(Self {
            d_times: upload_bytes(&ctx.device, &all_times),
            d_flux: upload_bytes(&ctx.device, &all_flux),
            d_flux_err_sq: upload_bytes(&ctx.device, &all_flux_err_sq),
            d_band: upload_bytes(&ctx.device, &all_band),
            d_offsets: upload_bytes(&ctx.device, &offsets),
            d_n_r: upload_bytes(&ctx.device, &n_r_vec),
            d_n_g: upload_bytes(&ctx.device, &n_g_vec),
            n_sources,
            h_orig_sizes,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU context
// ---------------------------------------------------------------------------

/// Holds the Metal device, command queue, and compiled compute pipeline.
///
/// Constructing this performs shader compilation (~hundred milliseconds on
/// first call), so prefer to build one and reuse it across many fits.
pub struct GpuContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    #[allow(dead_code)]
    library: Library,
}

// Safety: the underlying Objective-C objects (MTLDevice, MTLCommandQueue,
// MTLComputePipelineState, MTLLibrary) are documented thread-safe by Apple.
// Buffers are accessed only while a CommandBuffer is in flight or when no
// kernel is running, so there's no host/device data race.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Construct a new context bound to the system's default Metal device.
    /// `device` is accepted for API parity with [`crate::gpu::GpuContext`] but
    /// is currently ignored — Apple Silicon Macs have a single integrated GPU.
    pub fn new(_device: i32) -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found (this Mac has no GPU?)".to_string())?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(KERNEL_SRC, &opts)
            .map_err(|e| format!("Metal shader compile failed: {}", e))?;
        let function = library
            .get_function(KERNEL_FN, None)
            .map_err(|e| format!("Kernel function `{}` not found: {}", KERNEL_FN, e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline state creation failed: {}", e))?;

        Ok(Self {
            device,
            queue,
            pipeline,
            library,
        })
    }

    /// No-op on Metal (single GPU on Apple Silicon). Present for API parity.
    pub fn set_device(&self) -> Result<(), String> {
        Ok(())
    }

    /// Always 0 on Apple Silicon — exists for API parity with the CUDA path.
    pub fn device_id(&self) -> i32 {
        0
    }

    /// Run batch PSO for all sources simultaneously.
    ///
    /// Cost evaluation runs on the GPU; pbest / gbest / restart logic runs on
    /// the CPU. This is the same split the CUDA path uses — see [`crate::gpu`].
    pub fn batch_pso<S: AsRef<SourceData>>(
        &self,
        data: &GpuBatchData,
        sources: &[S],
        config: &PsoConfig,
        seed: u64,
    ) -> Result<Vec<FitResult>, String> {
        let n_sources = data.n_sources;
        let dim = N_PARAMS;
        let n_particles = config.n_particles;
        let total_particles = n_sources * n_particles;

        let priors = PriorArrays::new();

        // Static per-run uploads: priors + logged mask (constant across iters).
        let prior_means_f32: Vec<f32> = priors.means.iter().map(|&x| x as f32).collect();
        let prior_stds_f32: Vec<f32> = priors.stds.iter().map(|&x| x as f32).collect();
        let logged_mask: Vec<c_int> = priors.logged.iter().map(|&b| b as c_int).collect();
        let d_prior_means = upload_bytes(&self.device, &prior_means_f32);
        let d_prior_stds = upload_bytes(&self.device, &prior_stds_f32);
        let d_logged_mask = upload_bytes(&self.device, &logged_mask);

        // PSO bounds and velocity clamp (CPU-side only).
        let lower: [f64; N_PARAMS] = priors.mins;
        let upper: [f64; N_PARAMS] = priors.maxs;
        let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

        let w_start = config.w;
        let w_end = 0.4;
        let c1 = config.c1;
        let c2 = config.c2;
        let inv_max_iters = 1.0 / config.max_iters as f64;

        // Initialize particles on CPU (matches gpu.rs exactly).
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

        // Persistent device buffers and host scratch space (f32 for the device,
        // converted from / to the f64 host arrays each iteration).
        let d_positions =
            alloc_shared(&self.device, total_particles * dim * size_of::<f32>());
        let d_costs = alloc_shared(&self.device, total_particles * size_of::<f32>());

        let mut scratch_pos_f32 = vec![0.0f32; total_particles * dim];
        let mut scratch_costs_f32 = vec![0.0f32; total_particles];

        let n_threadgroups = (total_particles as u64 + THREADGROUP - 1) / THREADGROUP;
        let threadgroup_count = MTLSize::new(n_threadgroups, 1, 1);
        let threads_per_group = MTLSize::new(THREADGROUP, 1, 1);

        let mut h_pbest_pos = h_positions.clone();
        let mut h_pbest_cost = vec![f64::INFINITY; total_particles];
        let mut h_gbest_pos = vec![0.0f64; n_sources * dim];
        let mut h_gbest_cost = vec![f64::INFINITY; n_sources];
        let mut h_costs = vec![0.0f64; total_particles];

        let mut prev_gbest = vec![f64::INFINITY; n_sources];
        let mut stall_count = vec![0usize; n_sources];
        let mut source_done = vec![false; n_sources];

        let restart_threshold = 40usize;
        let restart_fraction = 0.3;

        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(n_sources as u64 + 1));

        for iter in 0..config.max_iters {
            let w = w_start - (w_start - w_end) * (iter as f64) * inv_max_iters;

            // 1. Convert positions f64 → f32 and copy into the device buffer.
            for (dst, &src) in scratch_pos_f32.iter_mut().zip(h_positions.iter()) {
                *dst = src as f32;
            }
            unsafe { write_buffer(&d_positions, &scratch_pos_f32) };

            // 2. Dispatch the cost kernel.
            let cost_params = CostParams {
                n_sources: n_sources as c_int,
                n_particles: n_particles as c_int,
            };
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&data.d_times), 0);
            enc.set_buffer(1, Some(&data.d_flux), 0);
            enc.set_buffer(2, Some(&data.d_flux_err_sq), 0);
            enc.set_buffer(3, Some(&data.d_band), 0);
            enc.set_buffer(4, Some(&data.d_offsets), 0);
            enc.set_buffer(5, Some(&data.d_n_r), 0);
            enc.set_buffer(6, Some(&data.d_n_g), 0);
            enc.set_buffer(7, Some(&d_positions), 0);
            enc.set_buffer(8, Some(&d_costs), 0);
            enc.set_buffer(9, Some(&d_prior_means), 0);
            enc.set_buffer(10, Some(&d_prior_stds), 0);
            enc.set_buffer(11, Some(&d_logged_mask), 0);
            enc.set_bytes(
                12,
                size_of::<CostParams>() as u64,
                &cost_params as *const _ as *const _,
            );
            enc.dispatch_thread_groups(threadgroup_count, threads_per_group);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // 3. Read costs back, convert f32 → f64.
            unsafe {
                let src = read_buffer::<f32>(&d_costs, total_particles);
                scratch_costs_f32.copy_from_slice(src);
            }
            for (dst, &src) in h_costs.iter_mut().zip(scratch_costs_f32.iter()) {
                *dst = src as f64;
            }

            // 4. CPU PSO dynamics — identical to gpu.rs.
            let mut all_done = true;
            for s in 0..n_sources {
                if source_done[s] {
                    continue;
                }

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

                let improved = prev_gbest[s] - h_gbest_cost[s]
                    > 1e-4 * prev_gbest[s].abs().max(1e-10);
                if improved {
                    stall_count[s] = 0;
                    prev_gbest[s] = h_gbest_cost[s];
                } else {
                    stall_count[s] += 1;

                    if stall_count[s] % restart_threshold == 0
                        && stall_count[s] < config.stall_iters
                    {
                        let n_restart = (n_particles as f64 * restart_fraction) as usize;
                        let mut indices: Vec<usize> = (0..n_particles).collect();
                        indices.sort_by(|&a, &b| {
                            let ia = s * n_particles + a;
                            let ib = s * n_particles + b;
                            h_pbest_cost[ib]
                                .partial_cmp(&h_pbest_cost[ia])
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
                                    h_positions[base + d] =
                                        (priors.means[d] + z * priors.stds[d])
                                            .clamp(lower[d], upper[d]);
                                } else {
                                    h_positions[base + d] = lower[d]
                                        + rng.random::<f64>() * (upper[d] - lower[d]);
                                }
                                h_velocities[base + d] = (upper[d] - lower[d])
                                    * 0.1
                                    * (2.0 * rng.random::<f64>() - 1.0);
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
            if all_done {
                break;
            }
        }

        let results: Vec<FitResult> = (0..n_sources)
            .map(|s| {
                let gb = s * dim;
                let mut raw = [0.0f64; N_PARAMS];
                raw.copy_from_slice(&h_gbest_pos[gb..gb + dim]);

                let phys = to_physical(&raw, &priors);

                let d = &sources[s].as_ref().data;
                let param_map = build_param_map(&d.obs);
                let rchi2 = reduced_chi2(&raw, &d.obs, &param_map, d.orig_size, &priors);

                let params = VillarParams::from_phys(&phys);
                let real_obs: Vec<Obs> = d
                    .obs
                    .iter()
                    .filter(|o| o.phase < 999.0 && o.flux_err < 999.0)
                    .cloned()
                    .collect();

                FitResult {
                    params,
                    params_unnorm: params.unnormalized(d.peak_flux),
                    peak_flux: d.peak_flux,
                    reduced_chi2: rchi2,
                    orig_size: d.orig_size,
                    obs: real_obs,
                }
            })
            .collect();

        Ok(results)
    }

    /// Run batch PSO with three seeds and return the best fit per source
    /// (matching the CPU and CUDA paths).
    pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
        &self,
        data: &GpuBatchData,
        sources: &[S],
        config: &PsoConfig,
    ) -> Result<Vec<FitResult>, String> {
        let seeds: [u64; 3] = [42, 137, 271];
        let mut best: Option<Vec<FitResult>> = None;

        for &seed in &seeds {
            let results = self.batch_pso(data, sources, config, seed)?;
            best = Some(match best {
                None => results,
                Some(prev) => prev
                    .into_iter()
                    .zip(results)
                    .map(|(a, b)| if b.reduced_chi2 < a.reduced_chi2 { b } else { a })
                    .collect(),
            });
        }
        Ok(best.unwrap())
    }
}

// ---------------------------------------------------------------------------
// Convenience: load a directory of CSVs (mirrors gpu::load_sources)
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
