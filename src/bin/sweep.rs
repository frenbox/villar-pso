//! PSO parameter sweep for a single object.

use std::time::Instant;
use villar_pso::*;

fn run_single(csv_path: &str, n_particles: usize, max_iters: usize, stall_iters: usize) -> Option<(f64, f64)> {
    let data = match preprocess(csv_path) {
        Ok(d) => d,
        Err(_) => return None,
    };
    let param_map = build_param_map(&data.obs);
    let priors = PriorArrays::new();

    let mut lower = [0.0; N_PARAMS];
    let mut upper = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        lower[i] = priors.mins[i];
        upper[i] = priors.maxs[i];
    }

    let config = PsoConfig {
        n_particles,
        max_iters,
        stall_iters,
        ..PsoConfig::default()
    };

    let band_indices = BandIndices::new(&data.obs, data.orig_size);

    let mut best_params = [0.0; N_PARAMS];
    let mut best_cost = f64::INFINITY;
    let mut first_cost = f64::INFINITY;

    let t0 = Instant::now();
    for (i, &seed) in [42u64, 137, 271].iter().enumerate() {
        if i == 2 && (first_cost - best_cost).abs() < 0.1 * best_cost.abs().max(1e-10) {
            break;
        }
        let cost_fn = |raw: &[f64; N_PARAMS]| {
            pso_cost(raw, &data.obs, &param_map, data.orig_size, &priors, 0.0, &band_indices)
        };
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
    let csv_path = "../data/photometry/ZTF18aafzers.csv";

    let particle_counts = [50, 100, 150, 200, 300, 500];
    let iter_counts = [200, 400, 600, 1000, 1500, 2000];

    println!("{:>10} {:>10} {:>10} {:>10}", "particles", "max_iter", "chi2", "time_ms");
    println!("{}", "-".repeat(44));

    for &np in &particle_counts {
        for &mi in &iter_counts {
            let si = (mi / 10).max(20);
            if let Some((rchi2, ms)) = run_single(csv_path, np, mi, si) {
                println!("{:>10} {:>10} {:>10.4} {:>10.1}", np, mi, rchi2, ms);
            }
        }
    }
}
