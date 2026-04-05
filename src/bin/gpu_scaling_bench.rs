//! GPU scaling benchmark: all 10,000 photometry fits with varying numbers
//! of GPUs. Each Rayon worker is bound to its own GPU and processes chunks
//! of 500 sources independently — no mutex needed.
//!
//! Usage: gpu-scaling-bench [data_dir] [--gpus N]
//!
//! If --gpus is not given, all visible GPUs are used.

#[path = "common/fit_stats.rs"]
mod fit_stats;

use rayon::prelude::*;
use std::time::Instant;
use villar_pso::gpu::{detect_gpu_count, load_sources, GpuContext};
use villar_pso::PsoConfig;

use fit_stats::compute_fit_stats;

const CHUNK_SIZE: usize = 500;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../data/photometry");

    // Parse --gpus flag
    let max_gpus: Option<usize> = args
        .windows(2)
        .find(|w| w[0] == "--gpus")
        .and_then(|w| w[1].parse().ok());

    // Detect available GPUs
    let n_gpus_available = detect_gpu_count();
    let n_gpus_max = max_gpus.unwrap_or(n_gpus_available).min(n_gpus_available);
    eprintln!(
        "GPUs available: {}, max to use: {}",
        n_gpus_available, n_gpus_max
    );

    // Preprocess all sources up-front
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

    // Warmup: fit first chunk on GPU 0
    {
        let gpu0 = GpuContext::new(0).expect("Failed to init GPU 0");
        let warm = &sources[..CHUNK_SIZE.min(n_sources)];
        let batch = gpu0.pack_batch(warm).expect("GPU upload failed");
        let _ = gpu0.batch_pso_multi_seed(&batch, warm, &config);
    }

    eprintln!(
        "\n{:>7}  {:>9}  {:>10}  {:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "n_gpus", "threads", "chunk_sz", "total_srcs", "total_ms", "ms/source", "speedup",
        "n_ok", "n_fail", "chi2_mean", "chi2_med", "chi2_std"
    );
    eprintln!("{}", "-".repeat(140));

    let mut baseline_ms_per_source = 0.0f64;

    let gpu_counts = [1, 2, 3, 4, 5, 6, 7, 8];

    for &n_gpus in gpu_counts.iter().filter(|&&g| g <= n_gpus_max) {
        // Create one GpuContext per GPU
        let gpus: Vec<GpuContext> = (0..n_gpus as i32)
            .map(|d| GpuContext::new(d).expect("Failed to init GPU"))
            .collect();

        // Rayon pool with one thread per GPU
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_gpus)
            .build()
            .expect("Failed to build thread pool");

        let t_start = Instant::now();

        let chunk_results: Vec<Vec<f64>> = pool.install(|| {
            sources
                .par_chunks(CHUNK_SIZE)
                .enumerate()
                .map(|(i, chunk)| {
                    // Round-robin assign chunks to GPUs
                    let gpu = &gpus[i % n_gpus];

                    // Bind this thread to the assigned GPU before allocating
                    gpu.set_device().expect("cudaSetDevice failed");

                    let batch = gpu.pack_batch(chunk).expect("GPU upload failed");
                    let results = gpu
                        .batch_pso_multi_seed(&batch, chunk, &config)
                        .expect("GPU batch PSO failed");
                    results.iter().map(|r| r.reduced_chi2).collect()
                })
                .collect()
        });
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let stats = compute_fit_stats(
            chunk_results.into_iter().flatten().collect(),
            n_sources,
            total_ms,
        );

        if n_gpus == 1 {
            baseline_ms_per_source = stats.per_source_ms;
        }
        let speedup = baseline_ms_per_source / stats.per_source_ms;

        eprintln!(
            "{:>7}  {:>9}  {:>10}  {:>12}  {:>10.1}  {:>10.2}  {:>10.2}x  {:>8}  {:>8}  {:>10.3}  {:>10.3}  {:>10.3}",
            n_gpus, n_gpus, CHUNK_SIZE, n_sources, total_ms, stats.per_source_ms, speedup,
            stats.n_ok, stats.n_fail, stats.chi2_mean, stats.chi2_med, stats.chi2_std,
        );
    }
}
