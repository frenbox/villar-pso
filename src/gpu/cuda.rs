//! CUDA backend for batch PSO.

use std::ffi::c_int;
use std::mem::size_of;
use std::ptr;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{
    build_param_map, reduced_chi2, to_physical, BandIndices, FitResult, Obs, PriorArrays,
    PsoConfig, VillarParams, N_PARAMS,
};

use super::SourceData;

type CudaResult = c_int;

extern "C" {
    fn cudaSetDevice(device: c_int) -> CudaResult;
    fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> CudaResult;
    fn cudaFree(devPtr: *mut u8) -> CudaResult;
    fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: c_int) -> CudaResult;
    fn cudaDeviceSynchronize() -> CudaResult;
    fn cudaGetLastError() -> CudaResult;
    fn cudaGetErrorString(error: CudaResult) -> *const i8;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
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

fn cuda_check(code: CudaResult) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                "unknown CUDA error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
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

unsafe impl Send for DevBuf {}

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
            cudaMemcpy(
                buf.ptr,
                data.as_ptr() as *const u8,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        })?;
        Ok(buf)
    }

    fn download_into<T>(&self, host: &mut [T]) -> Result<(), String> {
        let bytes = host.len() * size_of::<T>();
        cuda_check(unsafe {
            cudaMemcpy(
                host.as_mut_ptr() as *mut u8,
                self.ptr,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        })
    }

    fn upload_from<T>(&self, data: &[T]) -> Result<(), String> {
        let bytes = data.len() * size_of::<T>();
        cuda_check(unsafe {
            cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const u8,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

/// Packed observation data resident on CUDA.
pub struct CudaBatchData {
    d_times: DevBuf,
    d_flux: DevBuf,
    d_flux_err_sq: DevBuf,
    d_band: DevBuf,
    d_offsets: DevBuf,
    d_n_r: DevBuf,
    d_n_g: DevBuf,
    n_sources: usize,
}

impl CudaBatchData {
    pub fn new<S: AsRef<SourceData>>(sources: &[S]) -> Result<Self, String> {
        let n_sources = sources.len();
        let mut all_times = Vec::new();
        let mut all_flux = Vec::new();
        let mut all_flux_err_sq = Vec::new();
        let mut all_band: Vec<c_int> = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_sources + 1);
        let mut n_r_vec: Vec<c_int> = Vec::with_capacity(n_sources);
        let mut n_g_vec: Vec<c_int> = Vec::with_capacity(n_sources);

        offsets.push(0);
        for src in sources {
            let d = &src.as_ref().data;
            let bi = BandIndices::new(&d.obs, d.orig_size);
            for ob in d.obs.iter().take(d.orig_size) {
                all_times.push(ob.phase);
                all_flux.push(ob.flux);
                all_flux_err_sq.push(ob.flux_err * ob.flux_err);
                all_band.push(ob.band.idx() as c_int);
            }
            offsets.push(all_times.len() as c_int);
            n_r_vec.push(bi.r_indices.len() as c_int);
            n_g_vec.push(bi.g_indices.len() as c_int);
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
        })
    }
}

fn init_particles(
    n_sources: usize,
    n_particles: usize,
    dim: usize,
    seed: u64,
    priors: &PriorArrays,
    lower: &[f64; N_PARAMS],
    upper: &[f64; N_PARAMS],
    v_max: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let total_particles = n_sources * n_particles;
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
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    h_positions[base + d] =
                        (priors.means[d] + z * priors.stds[d]).clamp(lower[d], upper[d]);
                }
            } else {
                for d in 0..dim {
                    h_positions[base + d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                }
            }
            for d in 0..dim {
                h_velocities[base + d] = v_max[d] * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
            }
        }
    }

    (h_positions, h_velocities)
}

fn collect_fit_results<S: AsRef<SourceData>>(
    sources: &[S],
    h_gbest_pos: &[f64],
    dim: usize,
    priors: &PriorArrays,
) -> Vec<FitResult> {
    (0..sources.len())
        .map(|s| {
            let gb = s * dim;
            let mut raw = [0.0f64; N_PARAMS];
            raw.copy_from_slice(&h_gbest_pos[gb..gb + dim]);

            let phys = to_physical(&raw, priors);
            let d = &sources[s].as_ref().data;
            let param_map = build_param_map(&d.obs);
            let rchi2 = reduced_chi2(&raw, &d.obs, &param_map, d.orig_size, priors);

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
        .collect()
}

pub struct CudaContext {
    device: i32,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device: i32) -> Result<Self, String> {
        cuda_check(unsafe { cudaSetDevice(device) })?;
        Ok(Self { device })
    }

    pub fn set_device(&self) -> Result<(), String> {
        cuda_check(unsafe { cudaSetDevice(self.device) })
    }

    pub fn device_id(&self) -> i32 {
        self.device
    }

    fn evaluate_costs_on_gpu(
        &self,
        data: &CudaBatchData,
        d_positions: &DevBuf,
        d_costs: &DevBuf,
        h_positions: &[f64],
        h_costs: &mut [f64],
        d_prior_means: &DevBuf,
        d_prior_stds: &DevBuf,
        d_logged_mask: &DevBuf,
        n_sources: usize,
        n_particles: usize,
        particle_grid: c_int,
        block: c_int,
    ) -> Result<(), String> {
        d_positions.upload_from(h_positions)?;
        unsafe {
            launch_batch_pso_cost_villar_joint(
                data.d_times.ptr as _,
                data.d_flux.ptr as _,
                data.d_flux_err_sq.ptr as _,
                data.d_band.ptr as _,
                data.d_offsets.ptr as _,
                data.d_n_r.ptr as _,
                data.d_n_g.ptr as _,
                d_positions.ptr as _,
                d_costs.ptr as _,
                d_prior_means.ptr as _,
                d_prior_stds.ptr as _,
                d_logged_mask.ptr as _,
                n_sources as c_int,
                n_particles as c_int,
                particle_grid,
                block,
            );
            cuda_check(cudaGetLastError())?;
            cuda_check(cudaDeviceSynchronize())?;
        }
        d_costs.download_into(h_costs)
    }

    pub fn batch_pso<S: AsRef<SourceData>>(
        &self,
        data: &CudaBatchData,
        sources: &[S],
        config: &PsoConfig,
        seed: u64,
    ) -> Result<Vec<FitResult>, String> {
        cuda_check(unsafe { cudaSetDevice(self.device) })?;

        let n_sources = data.n_sources;
        let dim = N_PARAMS;
        let n_particles = config.n_particles;
        let total_particles = n_sources * n_particles;

        let priors = PriorArrays::new();

        let prior_means: Vec<f64> = priors.means.to_vec();
        let prior_stds: Vec<f64> = priors.stds.to_vec();
        let logged_mask: Vec<c_int> = priors.logged.iter().map(|&b| b as c_int).collect();
        let d_prior_means = DevBuf::upload(&prior_means)?;
        let d_prior_stds = DevBuf::upload(&prior_stds)?;
        let d_logged_mask = DevBuf::upload(&logged_mask)?;

        let lower: [f64; N_PARAMS] = priors.mins;
        let upper: [f64; N_PARAMS] = priors.maxs;
        let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

        let w_start = config.w;
        let w_end = 0.4;
        let c1 = config.c1;
        let c2 = config.c2;
        let inv_max_iters = 1.0 / config.max_iters as f64;

        let (mut h_positions, mut h_velocities) = init_particles(
            n_sources,
            n_particles,
            dim,
            seed,
            &priors,
            &lower,
            &upper,
            &v_max,
        );

        let d_positions = DevBuf::alloc(total_particles * dim * size_of::<f64>())?;
        let d_costs = DevBuf::alloc(total_particles * size_of::<f64>())?;

        let block: c_int = 256;
        let particle_grid: c_int = ((total_particles as c_int) + block - 1) / block;

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

            self.evaluate_costs_on_gpu(
                data,
                &d_positions,
                &d_costs,
                &h_positions,
                &mut h_costs,
                &d_prior_means,
                &d_prior_stds,
                &d_logged_mask,
                n_sources,
                n_particles,
                particle_grid,
                block,
            )?;

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

                let improved =
                    prev_gbest[s] - h_gbest_cost[s] > 1e-4 * prev_gbest[s].abs().max(1e-10);
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
                                    h_positions[base + d] = (priors.means[d] + z * priors.stds[d])
                                        .clamp(lower[d], upper[d]);
                                } else {
                                    h_positions[base + d] =
                                        lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                                }
                                h_velocities[base + d] =
                                    (upper[d] - lower[d]) * 0.1 * (2.0 * rng.random::<f64>() - 1.0);
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

        Ok(collect_fit_results(sources, &h_gbest_pos, dim, &priors))
    }

    pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
        &self,
        data: &CudaBatchData,
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
                    .map(|(a, b)| {
                        if b.reduced_chi2 < a.reduced_chi2 {
                            b
                        } else {
                            a
                        }
                    })
                    .collect(),
            });
        }
        Ok(best.unwrap())
    }
}

pub fn detect_gpu_count() -> usize {
    let mut count: i32 = 0;
    let err = unsafe { cudaGetDeviceCount(&mut count) };
    if err != 0 || count < 1 {
        1
    } else {
        count as usize
    }
}
