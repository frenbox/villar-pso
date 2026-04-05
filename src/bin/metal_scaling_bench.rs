//! Metal scaling benchmark scaffold.
//!
//! Mirrors the CUDA scaling bench structure but uses the explicit Metal backend.
//! On macOS, this exercises the Metal kernel path.
//! On non-macOS hosts, the backend reports unsupported-platform errors.
//!
//! Usage: metal-scaling-bench [data_dir]

#[path = "common/fit_stats.rs"]
mod fit_stats;

use std::time::Instant;

use rayon::prelude::*;
use villar_pso::gpu::{detect_gpu_count_for_backend, load_sources, GpuBackend, GpuContext};
use villar_pso::PsoConfig;

use fit_stats::compute_fit_stats;

const CHUNK_SIZE: usize = 500;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../data/photometry");

    let backend = GpuBackend::Metal;
    let n_gpus_available = detect_gpu_count_for_backend(backend).max(1);
    eprintln!("Metal devices available (reported): {}", n_gpus_available);

    let t_load = Instant::now();
    let sources = load_sources(data_dir);
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let n_sources = sources.len();
    eprintln!("Loaded {} sources in {:.1}ms", n_sources, load_ms);

    if n_sources == 0 {
        eprintln!("No valid sources found in {}", data_dir);
        std::process::exit(1);
    }

    let n_chunks = (n_sources + CHUNK_SIZE - 1) / CHUNK_SIZE;
    eprintln!("Chunk size: {}, total chunks: {}", CHUNK_SIZE, n_chunks);

    let config = PsoConfig::default();
    let n_workers = 1usize;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build()
        .expect("Failed to build thread pool");

    let t_start = Instant::now();
    let result: Result<Vec<Vec<f64>>, String> = pool.install(|| {
        sources
            .par_chunks(CHUNK_SIZE)
            .map(|chunk| {
                let gpu = GpuContext::new_with_backend(0, backend)?;
                gpu.set_device()?;
                let batch = gpu.pack_batch(chunk)?;
                let fits = gpu.batch_pso_multi_seed(&batch, chunk, &config)?;
                Ok::<Vec<f64>, String>(fits.iter().map(|r| r.reduced_chi2).collect())
            })
            .collect()
    });

    match result {
        Ok(chunk_results) => {
            let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            let stats = compute_fit_stats(
                chunk_results.into_iter().flatten().collect(),
                n_sources,
                total_ms,
            );

            eprintln!(
                "metal benchmark complete: total_ms={:.1} ms/source={:.2} n_ok={} n_fail={} chi2_mean={:.3} chi2_med={:.3} chi2_std={:.3}",
                total_ms,
                stats.per_source_ms,
                stats.n_ok,
                stats.n_fail,
                stats.chi2_mean,
                stats.chi2_med,
                stats.chi2_std,
            );
        }
        Err(err) => {
            eprintln!("Metal benchmark backend status: {}", err);
            if err.contains("not implemented yet") || err.contains("requires macOS") {
                eprintln!("Metal backend is only available on macOS hosts.");
                std::process::exit(2);
            }
            std::process::exit(1);
        }
    }
}
