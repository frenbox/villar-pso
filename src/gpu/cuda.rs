//! CUDA backend for batch PSO.

use std::ffi::c_int;
use std::mem::size_of;
use std::ptr;

use crate::{FitResult, PriorArrays, PsoConfig, MULTI_SEEDS, N_PARAMS};

use super::host_shared::{
    collect_fit_results, merge_best_results, pack_host_batch, run_host_pso_loop, HostPsoContext,
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
        let packed = pack_host_batch(sources);

        Ok(Self {
            d_times: DevBuf::upload(&packed.all_times)?,
            d_flux: DevBuf::upload(&packed.all_flux)?,
            d_flux_err_sq: DevBuf::upload(&packed.all_flux_err_sq)?,
            d_band: DevBuf::upload(&packed.all_band)?,
            d_offsets: DevBuf::upload(&packed.source_offsets)?,
            d_n_r: DevBuf::upload(&packed.n_r_per_source)?,
            d_n_g: DevBuf::upload(&packed.n_g_per_source)?,
            n_sources,
        })
    }
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

        let d_positions = DevBuf::alloc(total_particles * dim * size_of::<f64>())?;
        let d_costs = DevBuf::alloc(total_particles * size_of::<f64>())?;

        let block: c_int = 256;
        let particle_grid: c_int = ((total_particles as c_int) + block - 1) / block;

        let outcome = run_host_pso_loop(
            HostPsoContext {
                n_sources,
                n_particles,
                dim,
                seed,
                config,
                priors: &priors,
                lower: &lower,
                upper: &upper,
                v_max: &v_max,
            },
            |h_positions, h_costs| {
                self.evaluate_costs_on_gpu(
                    data,
                    &d_positions,
                    &d_costs,
                    h_positions,
                    h_costs,
                    &d_prior_means,
                    &d_prior_stds,
                    &d_logged_mask,
                    n_sources,
                    n_particles,
                    particle_grid,
                    block,
                )
            },
        )?;

        Ok(collect_fit_results(
            sources,
            &outcome.gbest_pos,
            dim,
            &priors,
        ))
    }

    pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
        &self,
        data: &CudaBatchData,
        sources: &[S],
        config: &PsoConfig,
    ) -> Result<Vec<FitResult>, String> {
        let mut best: Option<Vec<FitResult>> = None;

        for &seed in &MULTI_SEEDS {
            let results = self.batch_pso(data, sources, config, seed)?;
            best = merge_best_results(best, results);
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
