//! CPU scaling benchmark: all 10,000 photometry fits with varying Rayon
//! worker counts. Each worker processes ~500 files (10000 / n_workers).
//!
//! Usage: cpu-scaling-bench [data_dir]

use std::time::Instant;
use rayon::prelude::*;
use villar_pso::fit_lightcurve_quiet;

const FITS_PER_WORKER: usize = 500;

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
        let _ = fit_lightcurve_quiet(first);
    }

    eprintln!(
        "{:>9}  {:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "workers", "files/wkr", "total_ms", "ms/source", "speedup",
        "n_ok", "n_fail", "chi2_mean", "chi2_med", "chi2_std"
    );
    eprintln!("{}", "-".repeat(115));

    let mut baseline_ms_per_source = 0.0f64;

    // All configs process all 10,000 files; vary worker count
    // so each worker handles ~500 files at the max worker count
    let worker_counts = [4, 6, 8, 10, 12];

    for &n_workers in &worker_counts {

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .expect("Failed to build thread pool");

        let t_start = Instant::now();
        // Each worker dequeues a chunk of 500 files at a time
        let results: Vec<_> = pool.install(|| {
            csvs.par_chunks(FITS_PER_WORKER)
                .flat_map_iter(|chunk| {
                    chunk.iter().map(|csv| match fit_lightcurve_quiet(csv) {
                        Ok(res) => Some(res.reduced_chi2),
                        Err(_) => None,
                    })
                })
                .collect()
        });
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let mut chi2_vals: Vec<f64> = results.iter().filter_map(|r| *r).collect();
        let n_ok = chi2_vals.len();
        let n_fail = n_files - n_ok;
        let per_source = total_ms / n_ok.max(1) as f64;

        // Aggregate chi2 statistics
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

        if n_workers == 1 {
            baseline_ms_per_source = per_source;
        }
        let speedup = baseline_ms_per_source / per_source;

        eprintln!(
            "{:>9}  {:>12}  {:>10.1}  {:>10.2}  {:>10.2}x  {:>8}  {:>8}  {:>10.3}  {:>10.3}  {:>10.3}",
            n_workers, n_files / n_workers, total_ms, per_source, speedup, n_ok, n_fail,
            chi2_mean, chi2_med, chi2_std,
        );
    }
}
