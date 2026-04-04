use crate::{FitResult, PsoConfig};

#[cfg(target_os = "macos")]
use crate::{PriorArrays, MULTI_SEEDS, N_PARAMS};

#[cfg(target_os = "macos")]
use super::host_shared::{
    collect_fit_results, merge_best_results, pack_host_batch, run_host_pso_loop, HostPsoContext,
};
use super::SourceData;

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    };
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::ptr;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct KernelParams {
        n_sources: u32,
        n_particles: u32,
    }

    fn as_u64_bytes<T>(len: usize) -> u64 {
        (len * size_of::<T>()) as u64
    }

    fn upload_slice<T>(device: &Device, data: &[T]) -> Buffer {
        unsafe {
            device.new_buffer_with_data(
                data.as_ptr() as *const c_void,
                as_u64_bytes::<T>(data.len()),
                MTLResourceOptions::StorageModeShared,
            )
        }
    }

    fn alloc_buffer<T>(device: &Device, len: usize) -> Buffer {
        device.new_buffer(
            as_u64_bytes::<T>(len),
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub struct MetalBatchData {
        all_times: Buffer,
        all_flux: Buffer,
        all_flux_err_sq: Buffer,
        all_band: Buffer,
        source_offsets: Buffer,
        n_r_per_source: Buffer,
        n_g_per_source: Buffer,
        n_sources: usize,
    }

    pub struct MetalContext {
        device_id: i32,
        device: Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    impl MetalContext {
        pub fn new(device: i32) -> Result<Self, String> {
            let devices = Device::all();
            if devices.is_empty() {
                return Err("No Metal devices available".to_string());
            }
            let idx = (device as usize).min(devices.len() - 1);
            let dev = devices[idx].to_owned();

            let source = include_str!("../../metal/villar_joint.metal");
            let options = CompileOptions::new();
            let lib = dev
                .new_library_with_source(source, &options)
                .map_err(|e| format!("Metal shader compile failed: {}", e))?;
            let func = lib
                .get_function("batch_pso_cost_villar_joint_metal", None)
                .map_err(|e| format!("Metal function lookup failed: {}", e))?;
            let pipeline = dev
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Metal pipeline creation failed: {}", e))?;
            let queue = dev.new_command_queue();

            Ok(Self {
                device_id: idx as i32,
                device: dev,
                queue,
                pipeline,
            })
        }

        pub fn set_device(&self) -> Result<(), String> {
            Ok(())
        }

        pub fn device_id(&self) -> i32 {
            self.device_id
        }

        pub fn pack_batch<S: AsRef<SourceData>>(
            &self,
            sources: &[S],
        ) -> Result<MetalBatchData, String> {
            let n_sources = sources.len();
            let packed = pack_host_batch(sources);

            Ok(MetalBatchData {
                all_times: upload_slice(&self.device, &packed.all_times),
                all_flux: upload_slice(&self.device, &packed.all_flux),
                all_flux_err_sq: upload_slice(&self.device, &packed.all_flux_err_sq),
                all_band: upload_slice(&self.device, &packed.all_band),
                source_offsets: upload_slice(&self.device, &packed.source_offsets),
                n_r_per_source: upload_slice(&self.device, &packed.n_r_per_source),
                n_g_per_source: upload_slice(&self.device, &packed.n_g_per_source),
                n_sources,
            })
        }

        fn evaluate_costs_on_metal(
            &self,
            data: &MetalBatchData,
            positions_buf: &Buffer,
            costs_buf: &Buffer,
            params_buf: &Buffer,
            prior_means: &Buffer,
            prior_stds: &Buffer,
            logged_mask: &Buffer,
            h_positions: &[f64],
            h_costs: &mut [f64],
            total_particles: usize,
        ) -> Result<(), String> {
            unsafe {
                ptr::copy_nonoverlapping(
                    h_positions.as_ptr(),
                    positions_buf.contents() as *mut f64,
                    h_positions.len(),
                );
            }

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&data.all_times), 0);
            encoder.set_buffer(1, Some(&data.all_flux), 0);
            encoder.set_buffer(2, Some(&data.all_flux_err_sq), 0);
            encoder.set_buffer(3, Some(&data.all_band), 0);
            encoder.set_buffer(4, Some(&data.source_offsets), 0);
            encoder.set_buffer(5, Some(&data.n_r_per_source), 0);
            encoder.set_buffer(6, Some(&data.n_g_per_source), 0);
            encoder.set_buffer(7, Some(positions_buf), 0);
            encoder.set_buffer(8, Some(costs_buf), 0);
            encoder.set_buffer(9, Some(prior_means), 0);
            encoder.set_buffer(10, Some(prior_stds), 0);
            encoder.set_buffer(11, Some(logged_mask), 0);
            encoder.set_buffer(12, Some(params_buf), 0);

            let threads_per_group = self.pipeline.thread_execution_width().max(1) as u64;
            let group_count =
                ((total_particles as u64) + threads_per_group - 1) / threads_per_group;
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(group_count, 1, 1),
                metal::MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            unsafe {
                ptr::copy_nonoverlapping(
                    costs_buf.contents() as *const f64,
                    h_costs.as_mut_ptr(),
                    h_costs.len(),
                );
            }

            Ok(())
        }

        pub fn batch_pso<S: AsRef<SourceData>>(
            &self,
            data: &MetalBatchData,
            sources: &[S],
            config: &PsoConfig,
            seed: u64,
        ) -> Result<Vec<FitResult>, String> {
            let n_sources = data.n_sources;
            let dim = N_PARAMS;
            let n_particles = config.n_particles;
            let total_particles = n_sources * n_particles;

            let priors = PriorArrays::new();
            let lower: [f64; N_PARAMS] = priors.mins;
            let upper: [f64; N_PARAMS] = priors.maxs;
            let v_max: Vec<f64> = (0..dim).map(|d| 0.5 * (upper[d] - lower[d])).collect();

            let positions_buf = alloc_buffer::<f64>(&self.device, total_particles * dim);
            let costs_buf = alloc_buffer::<f64>(&self.device, total_particles);
            let prior_means_buf = upload_slice(&self.device, &priors.means);
            let prior_stds_buf = upload_slice(&self.device, &priors.stds);
            let logged_mask: Vec<i32> = priors
                .logged
                .iter()
                .map(|&b| if b { 1 } else { 0 })
                .collect();
            let logged_mask_buf = upload_slice(&self.device, &logged_mask);
            let params = KernelParams {
                n_sources: n_sources as u32,
                n_particles: n_particles as u32,
            };
            let params_buf = upload_slice(&self.device, &[params]);

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
                    self.evaluate_costs_on_metal(
                        data,
                        &positions_buf,
                        &costs_buf,
                        &params_buf,
                        &prior_means_buf,
                        &prior_stds_buf,
                        &logged_mask_buf,
                        h_positions,
                        h_costs,
                        total_particles,
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
            data: &MetalBatchData,
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
        Device::all().len().max(1)
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    pub struct MetalBatchData;

    pub struct MetalContext {
        device: i32,
    }

    impl MetalContext {
        pub fn new(device: i32) -> Result<Self, String> {
            Ok(Self { device })
        }

        pub fn set_device(&self) -> Result<(), String> {
            Ok(())
        }

        pub fn device_id(&self) -> i32 {
            self.device
        }

        pub fn pack_batch<S: AsRef<SourceData>>(
            &self,
            _sources: &[S],
        ) -> Result<MetalBatchData, String> {
            Err("Metal backend requires macOS; not implemented on this platform".to_string())
        }

        pub fn batch_pso<S: AsRef<SourceData>>(
            &self,
            _data: &MetalBatchData,
            _sources: &[S],
            _config: &PsoConfig,
            _seed: u64,
        ) -> Result<Vec<FitResult>, String> {
            Err("Metal backend requires macOS; not implemented on this platform".to_string())
        }

        pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
            &self,
            _data: &MetalBatchData,
            _sources: &[S],
            _config: &PsoConfig,
        ) -> Result<Vec<FitResult>, String> {
            Err("Metal backend requires macOS; not implemented on this platform".to_string())
        }
    }

    pub fn detect_gpu_count() -> usize {
        1
    }
}

pub use imp::{detect_gpu_count, MetalBatchData, MetalContext};
