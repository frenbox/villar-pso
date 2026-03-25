//! Benchmark CPU parallel batch fitting with Rayon, sweeping 1–10 threads.
//!
//! Usage: cpu-parallel-bench [data_dir]

use std::time::Instant;
use rayon::prelude::*;
use villar_pso::fit_lightcurve;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("../data/photometry");

    let mut csvs: Vec<String> = std::fs::read_dir(data_dir)
        .expect("Cannot read data directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    csvs.sort();

    let n_files = csvs.len();
    eprintln!("Found {} CSV files in {}\n", n_files, data_dir);

    // Warmup: run one fit to avoid cold-start noise
    if let Some(first) = csvs.first() {
        let _ = fit_lightcurve(first);
    }

    eprintln!("{:>7}  {:>10}  {:>10}  {:>10}", "threads", "total_ms", "ms/source", "speedup");
    eprintln!("{}", "-".repeat(47));

    let mut baseline_ms = 0.0f64;

    for n_threads in 1..=10 {
        // Configure Rayon thread pool for this run
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("Failed to build thread pool");

        let t_start = Instant::now();
        let results: Vec<_> = pool.install(|| {
            csvs.par_iter()
                .map(|csv| {
                    let name = std::path::Path::new(csv)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                        .to_string();
                    match fit_lightcurve(csv) {
                        Ok(res) => Some((name, res.reduced_chi2)),
                        Err(_) => None,
                    }
                })
                .collect()
        });
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let n_ok = results.iter().filter(|r| r.is_some()).count();
        let per_source = total_ms / n_ok.max(1) as f64;

        if n_threads == 1 {
            baseline_ms = total_ms;
        }
        let speedup = baseline_ms / total_ms;

        eprintln!(
            "{:>7}  {:>10.1}  {:>10.2}  {:>10.2}x",
            n_threads, total_ms, per_source, speedup,
        );
    }
}
