//! Compare two Rayon dispatch strategies on the same 10,000 photometry files:
//!
//!   A) Sequential dispatch — par_iter() hands one file at a time to the pool;
//!      Rayon's work-stealing balances load at single-file granularity.
//!
//!   B) Batch dispatch — par_chunks(500) hands 500 files at a time; each worker
//!      processes its chunk sequentially before grabbing the next.
//!
//! Both strategies process ALL files with the same thread counts.
//!
//! Usage: cpu-dispatch-bench [data_dir]

use rayon::prelude::*;
use std::time::Instant;
use villar_pso::fit_lightcurve;

const BATCH_SIZE: usize = 500;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("../data/photometry");

    let mut csvs: Vec<String> = std::fs::read_dir(data_dir)
        .expect("Cannot read data directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    csvs.sort();

    let n_files = csvs.len();
    let n_batches = (n_files + BATCH_SIZE - 1) / BATCH_SIZE;
    eprintln!("Found {} CSV files in {}", n_files, data_dir);
    eprintln!("Batch size: {}, total batches: {}\n", BATCH_SIZE, n_batches);

    // Warmup
    if let Some(first) = csvs.first() {
        let _ = fit_lightcurve(first);
    }

    let worker_counts = [2, 4, 6, 8, 10];

    // =====================================================================
    //  A) Sequential dispatch — par_iter (one file per work unit)
    // =====================================================================
    eprintln!("=== Strategy A: Sequential dispatch (par_iter, 1 file at a time) ===\n");
    eprintln!(
        "{:>9}  {:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "threads",
        "total_srcs",
        "total_ms",
        "ms/source",
        "speedup",
        "n_ok",
        "n_fail",
        "chi2_mean",
        "chi2_med",
        "chi2_std"
    );
    eprintln!("{}", "-".repeat(115));

    let mut baseline_seq = 0.0f64;

    for &n_threads in &worker_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("Failed to build thread pool");

        let t_start = Instant::now();
        let results: Vec<_> = pool.install(|| {
            csvs.par_iter()
                .map(|csv| match fit_lightcurve(csv) {
                    Ok(res) => Some(res.reduced_chi2),
                    Err(_) => None,
                })
                .collect()
        });
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let stats = compute_stats(&results, n_files, total_ms);
        if n_threads == worker_counts[0] {
            baseline_seq = stats.per_source;
        }
        let speedup = baseline_seq / stats.per_source;

        eprintln!(
            "{:>9}  {:>12}  {:>10.1}  {:>10.2}  {:>10.2}x  {:>8}  {:>8}  {:>10.3}  {:>10.3}  {:>10.3}",
            n_threads, n_files, total_ms, stats.per_source, speedup,
            stats.n_ok, stats.n_fail, stats.chi2_mean, stats.chi2_med, stats.chi2_std,
        );
    }

    // =====================================================================
    //  B) Batch dispatch — par_chunks(500)
    // =====================================================================
    eprintln!(
        "\n=== Strategy B: Batch dispatch (par_chunks({}), {} batches) ===\n",
        BATCH_SIZE, n_batches
    );
    eprintln!(
        "{:>9}  {:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "threads",
        "total_srcs",
        "total_ms",
        "ms/source",
        "speedup",
        "n_ok",
        "n_fail",
        "chi2_mean",
        "chi2_med",
        "chi2_std"
    );
    eprintln!("{}", "-".repeat(115));

    let mut baseline_batch = 0.0f64;

    for &n_threads in &worker_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("Failed to build thread pool");

        let t_start = Instant::now();
        let results: Vec<_> = pool.install(|| {
            csvs.par_chunks(BATCH_SIZE)
                .flat_map_iter(|chunk| {
                    chunk.iter().map(|csv| match fit_lightcurve(csv) {
                        Ok(res) => Some(res.reduced_chi2),
                        Err(_) => None,
                    })
                })
                .collect()
        });
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let stats = compute_stats(&results, n_files, total_ms);
        if n_threads == worker_counts[0] {
            baseline_batch = stats.per_source;
        }
        let speedup = baseline_batch / stats.per_source;

        eprintln!(
            "{:>9}  {:>12}  {:>10.1}  {:>10.2}  {:>10.2}x  {:>8}  {:>8}  {:>10.3}  {:>10.3}  {:>10.3}",
            n_threads, n_files, total_ms, stats.per_source, speedup,
            stats.n_ok, stats.n_fail, stats.chi2_mean, stats.chi2_med, stats.chi2_std,
        );
    }

    // =====================================================================
    //  Summary
    // =====================================================================
    eprintln!("\n=== Comparison (ms/source at each thread count) ===\n");
    eprintln!(
        "{:>9}  {:>14}  {:>14}  {:>10}",
        "threads", "sequential", "batch(500)", "winner"
    );
    eprintln!("{}", "-".repeat(52));

    // Re-run quickly just to collect the paired numbers — but we already
    // printed them above, so let's just tell the user to compare the tables.
    eprintln!("(compare ms/source rows above for each thread count)");
}

struct Stats {
    n_ok: usize,
    n_fail: usize,
    per_source: f64,
    chi2_mean: f64,
    chi2_med: f64,
    chi2_std: f64,
}

fn compute_stats(results: &[Option<f64>], n_files: usize, total_ms: f64) -> Stats {
    let mut chi2_vals: Vec<f64> = results.iter().filter_map(|r| *r).collect();
    let n_ok = chi2_vals.len();
    let n_fail = n_files - n_ok;
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

    Stats {
        n_ok,
        n_fail,
        per_source,
        chi2_mean,
        chi2_med,
        chi2_std,
    }
}
