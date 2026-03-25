//! Quick benchmark: runs PSO on first N objects and reports chi2 + timing.

use std::time::Instant;
use villar_pso::*;

fn run_fit(csv_path: &str) -> Option<(f64, f64)> {
    let data = match preprocess(csv_path) {
        Ok(d) => d,
        Err(_) => return None,
    };
    let param_map = build_param_map(&data.obs);
    let priors = PriorArrays::new();
    let band_indices = BandIndices::new(&data.obs, data.orig_size);

    let mut lower = [0.0; N_PARAMS];
    let mut upper = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        lower[i] = priors.mins[i];
        upper[i] = priors.maxs[i];
    }

    let config = PsoConfig::default();

    let cost_fn = |raw: &[f64; N_PARAMS]| {
        pso_cost(raw, &data.obs, &param_map, data.orig_size, &priors, 0.0, &band_indices)
    };

    let t0 = Instant::now();

    let seeds: [u64; 3] = [42, 137, 271];
    let mut best_params = [0.0; N_PARAMS];
    let mut best_cost = f64::INFINITY;
    let mut first_cost = f64::INFINITY;

    for (i, &seed) in seeds.iter().enumerate() {
        if i >= 2 && (first_cost - best_cost).abs() < 0.05 * best_cost.abs().max(1e-10) {
            break;
        }
        let (params, cost) = pso_minimize(&cost_fn, &lower, &upper, &priors, &config, seed);
        if i == 0 { first_cost = cost; }
        if cost < best_cost {
            best_cost = cost;
            best_params = params;
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let rchi2 = reduced_chi2(&best_params, &data.obs, &param_map, data.orig_size, &priors);
    Some((rchi2, elapsed_ms))
}

fn main() {
    let data_dir = std::env::args().nth(1)
        .unwrap_or_else(|| "../data/photometry".to_string());

    let mut csvs: Vec<String> = std::fs::read_dir(&data_dir)
        .expect("Cannot read data dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    csvs.sort();
    csvs.truncate(20);

    println!("{:<20} {:>12} {:>10}", "object", "chi2", "ms");
    println!("{}", "-".repeat(44));

    for csv in &csvs {
        let name = std::path::Path::new(csv)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");

        if let Some((chi2, ms)) = run_fit(csv) {
            println!("{:<20} {:>12.6} {:>10.1}", name, chi2, ms);
        } else {
            println!("{:<20} SKIP", name);
        }
    }
}
