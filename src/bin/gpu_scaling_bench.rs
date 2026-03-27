//! GPU scaling benchmark: all 10,000 photometry fits with varying numbers
//! of GPUs. Each Rayon worker is bound to its own GPU and processes chunks
//! of 500 sources independently — no mutex needed.
//!
//! Usage: gpu-scaling-bench [data_dir] [--gpus N]
//!
//! If --gpus is not given, all visible GPUs are used.

use std::time::Instant;
use rayon::prelude::*;
use villar_pso::gpu::{GpuBatchData, GpuContext, load_sources};
use villar_pso::PsoConfig;

const CHUNK_SIZE: usize = 500;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("../data/photometry");

    // Parse --gpus flag
    let max_gpus: Option<usize> = args.windows(2)
        .find(|w| w[0] == "--gpus")
        .and_then(|w| w[1].parse().ok());

    // Detect available GPUs
    let n_gpus_available = detect_gpu_count();
    let n_gpus_max = max_gpus.unwrap_or(n_gpus_available).min(n_gpus_available);
    eprintln!("GPUs available: {}, max to use: {}", n_gpus_available, n_gpus_max);

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
        let batch = GpuBatchData::new(warm).expect("GPU upload failed");
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

                    let batch = GpuBatchData::new(chunk).expect("GPU upload failed");
                    let results = gpu
                        .batch_pso_multi_seed(&batch, chunk, &config)
                        .expect("GPU batch PSO failed");
                    results.iter().map(|r| r.reduced_chi2).collect()
                })
                .collect()
        });
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        // Flatten and compute statistics
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

        if n_gpus == 1 {
            baseline_ms_per_source = per_source;
        }
        let speedup = baseline_ms_per_source / per_source;

        eprintln!(
            "{:>7}  {:>9}  {:>10}  {:>12}  {:>10.1}  {:>10.2}  {:>10.2}x  {:>8}  {:>8}  {:>10.3}  {:>10.3}  {:>10.3}",
            n_gpus, n_gpus, CHUNK_SIZE, n_sources, total_ms, per_source, speedup, n_ok, n_fail,
            chi2_mean, chi2_med, chi2_std,
        );
    }
}

/// Query the number of CUDA-visible GPUs.
fn detect_gpu_count() -> usize {
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
    }
    let mut count: i32 = 0;
    let err = unsafe { cudaGetDeviceCount(&mut count) };
    if err != 0 || count < 1 {
        eprintln!("Warning: cudaGetDeviceCount failed or returned 0, defaulting to 1");
        1
    } else {
        count as usize
    }
}
