//! Single-batch GPU benchmark.
//!
//! Loads every CSV in `data_dir`, packs it into one `GpuBatchData`, and runs a
//! single `batch_pso` call on GPU 0 — measuring wall time and peak GPU memory.
//!
//! Usage: gpu-batch1000-bench [data_dir] [--multi-seed] [--n-particles N] [--max-iters N]
//!
//! Default data_dir is `photometry_long` (relative to the crate root).

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use villar_pso::gpu::{load_sources, GpuBatchData, GpuContext};
use villar_pso::PsoConfig;

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

fn mem_info() -> (usize, usize) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        cudaMemGetInfo(&mut free, &mut total);
    }
    (free, total)
}

fn mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn parse_usize_flag(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--") && !a.parse::<usize>().is_ok())
        .map(|s| s.as_str())
        .unwrap_or("photometry_long");
    let multi_seed = args.iter().any(|a| a == "--multi-seed");

    let mut config = PsoConfig::default();
    if let Some(n) = parse_usize_flag(&args, "--n-particles") {
        config.n_particles = n;
    }
    if let Some(n) = parse_usize_flag(&args, "--max-iters") {
        config.max_iters = n;
    }
    if let Some(n) = parse_usize_flag(&args, "--stall-iters") {
        config.stall_iters = n;
    }

    eprintln!("Loading sources from {}...", data_dir);
    let t = Instant::now();
    let sources = load_sources(data_dir);
    eprintln!(
        "Loaded {} sources in {:.2}s",
        sources.len(),
        t.elapsed().as_secs_f64()
    );

    if sources.is_empty() {
        eprintln!("No sources found in {}", data_dir);
        std::process::exit(1);
    }

    let total_obs: usize = sources.iter().map(|s| s.data.orig_size).sum();
    let max_obs = sources.iter().map(|s| s.data.orig_size).max().unwrap_or(0);
    let min_obs = sources.iter().map(|s| s.data.orig_size).min().unwrap_or(0);
    eprintln!(
        "Observations: total={}, min={}, max={}, mean={:.1}",
        total_obs,
        min_obs,
        max_obs,
        total_obs as f64 / sources.len() as f64
    );

    let gpu = GpuContext::new(0).expect("Failed to init GPU 0");

    let (free_baseline, total_mem) = mem_info();
    eprintln!("\nGPU 0 total memory: {:.1} MB", mb(total_mem));
    eprintln!(
        "Baseline free:      {:.1} MB (in use by others: {:.1} MB)",
        mb(free_baseline),
        mb(total_mem - free_baseline)
    );

    let t_up = Instant::now();
    let batch = GpuBatchData::new(&sources).expect("GPU upload failed");
    let upload_ms = t_up.elapsed().as_secs_f64() * 1000.0;
    let (free_after_upload, _) = mem_info();
    let upload_bytes = free_baseline.saturating_sub(free_after_upload);
    eprintln!(
        "\nGpuBatchData::new uploaded in {:.1}ms, allocated {:.1} MB",
        upload_ms,
        mb(upload_bytes)
    );

    let stop = Arc::new(AtomicBool::new(false));
    let min_free = Arc::new(AtomicUsize::new(usize::MAX));
    let poll = {
        let stop = Arc::clone(&stop);
        let min_free = Arc::clone(&min_free);
        thread::spawn(move || {
            unsafe {
                cudaSetDevice(0);
            }
            while !stop.load(Ordering::Relaxed) {
                let (free, _) = mem_info();
                let mut cur = min_free.load(Ordering::Relaxed);
                while free < cur {
                    match min_free.compare_exchange_weak(
                        cur,
                        free,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(c) => cur = c,
                    }
                }
                thread::sleep(Duration::from_millis(2));
            }
        })
    };

    eprintln!(
        "\nPSO config: n_particles={}, max_iters={}, stall_iters={}, mode={}",
        config.n_particles,
        config.max_iters,
        config.stall_iters,
        if multi_seed { "multi-seed (3)" } else { "single-seed" }
    );
    eprintln!("Running batch PSO on {} sources...", sources.len());

    let t_pso = Instant::now();
    let results = if multi_seed {
        gpu.batch_pso_multi_seed(&batch, &sources, &config)
            .expect("PSO failed")
    } else {
        gpu.batch_pso(&batch, &sources, &config, 42)
            .expect("PSO failed")
    };
    let pso_s = t_pso.elapsed().as_secs_f64();

    stop.store(true, Ordering::Relaxed);
    poll.join().ok();

    let min_free_v = min_free.load(Ordering::Relaxed);
    let peak_used_total = total_mem.saturating_sub(min_free_v);
    let peak_used_incremental = free_baseline.saturating_sub(min_free_v);

    let mut chi2: Vec<f64> = results.iter().map(|r| r.reduced_chi2).collect();
    chi2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = chi2[chi2.len() / 2];
    let mean = chi2.iter().sum::<f64>() / chi2.len() as f64;

    eprintln!("\n=========== Batch summary ===========");
    eprintln!("Sources processed:    {}", results.len());
    eprintln!("Wall time:            {:.2}s", pso_s);
    eprintln!(
        "ms/source:            {:.2}",
        pso_s * 1000.0 / results.len() as f64
    );
    eprintln!(
        "Obs upload bytes:     {:.1} MB",
        mb(upload_bytes)
    );
    eprintln!(
        "Peak GPU used (proc): {:.1} MB",
        mb(peak_used_incremental)
    );
    eprintln!(
        "Peak GPU used (tot):  {:.1} MB / {:.1} MB",
        mb(peak_used_total),
        mb(total_mem)
    );
    eprintln!(
        "Reduced chi2:         mean={:.3}  median={:.3}",
        mean, med
    );
}
