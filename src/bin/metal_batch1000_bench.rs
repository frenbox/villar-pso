//! Single-batch Apple Metal GPU benchmark.
//!
//! macOS counterpart to `gpu_batch1000_bench.rs`. Loads every CSV in
//! `data_dir`, packs it into one `GpuBatchData`, and runs a single
//! `batch_pso` call on the system's default Metal device — measuring wall
//! time and peak GPU allocations.
//!
//! Usage: metal-batch1000-bench [data_dir] [--multi-seed]
//!                              [--n-particles N] [--max-iters N]
//!                              [--stall-iters N]
//!
//! Default data_dir is `photometry_long` (relative to the crate root).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use metal::Device;
use villar_pso::gpu_metal::{load_sources, GpuBatchData, GpuContext};
use villar_pso::PsoConfig;

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

    // Construct a Metal context (compiles the shader, builds pipeline state).
    // The first call typically takes ~100ms.
    let t_ctx = Instant::now();
    let gpu = GpuContext::new(0).expect("Failed to init Metal device");
    eprintln!(
        "GpuContext::new (shader compile + pipeline) took {:.1}ms",
        t_ctx.elapsed().as_secs_f64() * 1000.0
    );

    // Independent device handle for memory polling. MTLDevice methods are
    // thread-safe; this avoids contending with the kernel-dispatch path.
    let probe = Device::system_default().expect("No Metal device");
    let recommended_max = probe.recommended_max_working_set_size() as usize;
    let baseline_alloc = probe.current_allocated_size() as usize;
    eprintln!(
        "\nMetal device:                {}",
        probe.name()
    );
    eprintln!(
        "Recommended max working set: {:.1} MB",
        mb(recommended_max)
    );
    eprintln!(
        "Baseline allocated:          {:.1} MB",
        mb(baseline_alloc)
    );

    let t_up = Instant::now();
    let batch = GpuBatchData::new(&gpu, &sources).expect("GPU upload failed");
    let upload_ms = t_up.elapsed().as_secs_f64() * 1000.0;
    let after_upload_alloc = probe.current_allocated_size() as usize;
    let upload_bytes = after_upload_alloc.saturating_sub(baseline_alloc);
    eprintln!(
        "\nGpuBatchData::new uploaded in {:.1}ms, allocated {:.1} MB",
        upload_ms,
        mb(upload_bytes)
    );

    // Background poller: track peak `current_allocated_size` while the PSO
    // loop runs. Apple Silicon uses unified memory so this reflects the
    // process's GPU-resident footprint, not a discrete VRAM pool.
    let stop = Arc::new(AtomicBool::new(false));
    let max_alloc = Arc::new(AtomicUsize::new(after_upload_alloc));
    let poll = {
        let stop = Arc::clone(&stop);
        let max_alloc = Arc::clone(&max_alloc);
        thread::spawn(move || {
            let probe = Device::system_default().expect("No Metal device");
            while !stop.load(Ordering::Relaxed) {
                let cur_alloc = probe.current_allocated_size() as usize;
                let mut cur_max = max_alloc.load(Ordering::Relaxed);
                while cur_alloc > cur_max {
                    match max_alloc.compare_exchange_weak(
                        cur_max,
                        cur_alloc,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(c) => cur_max = c,
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

    let peak_alloc = max_alloc.load(Ordering::Relaxed);
    let peak_used_incremental = peak_alloc.saturating_sub(baseline_alloc);

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
        "Peak GPU alloc (tot): {:.1} MB / {:.1} MB recommended",
        mb(peak_alloc),
        mb(recommended_max)
    );
    eprintln!(
        "Reduced chi2:         mean={:.3}  median={:.3}",
        mean, med
    );
}
