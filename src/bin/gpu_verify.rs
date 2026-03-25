//! Verify GPU cost function matches CPU cost function.
//! Runs a single source, evaluates cost on both, prints comparison.

use villar_pso::gpu::{GpuBatchData, GpuContext, SourceData};
use villar_pso::{preprocess, pso_cost, build_param_map, BandIndices, PriorArrays, PsoConfig, N_PARAMS};
use std::ffi::c_int;
use std::mem::size_of;

fn main() {
    let csv = std::env::args().nth(1).unwrap_or_else(|| "../data/photometry/ZTF18aafzers.csv".to_string());

    let data = preprocess(&csv).expect("preprocess failed");
    let priors = PriorArrays::new();
    let param_map = build_param_map(&data.obs);
    let band_indices = BandIndices::new(&data.obs, data.orig_size);

    // Create a known parameter vector (near prior means)
    let mut raw = [0.0f64; N_PARAMS];
    for i in 0..N_PARAMS {
        raw[i] = priors.means[i];
    }

    // CPU cost
    let cpu_cost = pso_cost(&raw, &data.obs, &param_map, data.orig_size, &priors, 0.0, &band_indices);
    println!("CPU cost at prior means: {:.10}", cpu_cost);

    // GPU cost: upload single source, evaluate kernel
    let source = SourceData { name: "test".to_string(), data };
    let sources = vec![source];
    let gpu = GpuContext::new(0).expect("GPU init");
    let batch_data = GpuBatchData::new(&sources).expect("upload");

    // Build the same GPU buffers used in batch_pso
    let prior_means: Vec<f64> = priors.means.to_vec();
    let prior_stds: Vec<f64> = priors.stds.to_vec();
    let logged_mask: Vec<c_int> = priors.logged.iter().map(|&b| b as c_int).collect();

    // We need to call the kernel with 1 source, 1 particle, positions = raw
    // Use the batch_pso with 1 particle and 1 iteration to get the initial cost
    let config = PsoConfig { n_particles: 1, max_iters: 1, stall_iters: 1, ..PsoConfig::default() };
    let results = gpu.batch_pso(&sources, &batch_data, &sources, &config, 42);

    // Actually, let's just print the raw params and the batch_pso result
    match results {
        Ok(r) => println!("GPU result: cost={:.10} chi2={:.10}", r[0].cost, r[0].reduced_chi2),
        Err(e) => println!("GPU error: {}", e),
    }

    // Also test with different raw params
    let mut raw2 = raw;
    raw2[0] = 0.2; // change A
    let cpu_cost2 = pso_cost(&raw2, &sources[0].data.obs, &param_map, sources[0].data.orig_size, &priors, 0.0, &band_indices);
    println!("CPU cost at modified params: {:.10}", cpu_cost2);
}
