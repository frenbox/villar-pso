//! Metal scaling benchmark scaffold.
//!
//! Mirrors the CUDA scaling bench structure but uses the explicit Metal backend.
//! The backend is currently a stub and will report "not implemented yet".
//!
//! Usage: metal-scaling-bench [data_dir]

use std::time::Instant;

use rayon::prelude::*;
use villar_pso::gpu::{detect_gpu_count_for_backend, load_sources, GpuBackend, GpuContext};
use villar_pso::PsoConfig;

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
            let mut chi2_vals: Vec<f64> = chunk_results.into_iter().flatten().collect();
            let n_ok = chi2_vals.len();
            let n_fail = n_sources - n_ok;
            let per_source = total_ms / n_ok.max(1) as f64;

            let (chi2_mean, chi2_med, chi2_std) = if n_ok > 0 {
                let mean = chi2_vals.iter().sum::<f64>() / n_ok as f64;
                let var = chi2_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_ok as f64;
                chi2_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = if n_ok % 2 == 0 {
                    (chi2_vals[n_ok / 2 - 1] + chi2_vals[n_ok / 2]) / 2.0
                } else {
                    chi2_vals[n_ok / 2]
                };
                (mean, med, var.sqrt())
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };

            eprintln!(
                "metal benchmark complete: total_ms={:.1} ms/source={:.2} n_ok={} n_fail={} chi2_mean={:.3} chi2_med={:.3} chi2_std={:.3}",
                total_ms, per_source, n_ok, n_fail, chi2_mean, chi2_med, chi2_std,
            );
        }
        Err(err) => {
            eprintln!("Metal benchmark backend status: {}", err);
            if err.contains("not implemented yet") {
                eprintln!("Metal kernel path is scaffolded but not implemented yet.");
                std::process::exit(2);
            }
            std::process::exit(1);
        }
    }
}
