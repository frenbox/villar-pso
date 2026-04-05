use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use villar_pso::{
    build_param_map, preprocess, pso_cost, pso_minimize, reduced_chi2, villar_flux_at, Band,
    BandIndices, MultiSeedStrategy, Obs, PhotometryMag, PriorArrays, PsoConfig, MULTI_SEEDS,
    N_PARAMS, VILLAR_BANDS,
};

#[cfg(feature = "cuda")]
use villar_pso::gpu::{GpuContext, SourceData};

fn flux_to_mag(flux: f64) -> f64 {
    23.9 - 2.5 * flux.log10()
}

fn normal_sample(rng: &mut SmallRng, stddev: f64) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    z * stddev
}

fn sample_times(
    rng: &mut SmallRng,
    start: f64,
    end: f64,
    step: f64,
    anchors: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut grid = Vec::new();
    let mut t = start;
    while t <= end + 1e-9 {
        grid.push(t);
        t += step;
    }

    let mut selected = anchors.to_vec();
    selected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    selected.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut candidates: Vec<f64> = grid
        .into_iter()
        .filter(|t| !selected.iter().any(|a| (a - t).abs() < 1e-9))
        .collect();
    candidates.shuffle(rng);

    for t in candidates
        .into_iter()
        .take(n.saturating_sub(selected.len()))
    {
        selected.push(t);
    }

    selected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    selected
}

fn write_csv(path: &PathBuf, points: &[PhotometryMag]) {
    let mut csv = String::from("mjd,magpsf,sigmapsf,fid\n");
    for p in points {
        let fid = match p.band {
            Band::G => 1,
            Band::R => 2,
            _ => panic!("test data only uses r/g bands"),
        };
        writeln!(csv, "{:.6},{:.6},{:.4},{}", p.time, p.mag, p.mag_err, fid).unwrap();
    }
    fs::write(path, csv).unwrap();
}

fn dense_villar_phys() -> [f64; N_PARAMS] {
    [
        1.00, 0.012, 22.0, 0.0, 4.0, 28.0, 0.03, 0.92, 0.010, 20.0, -1.0, 3.5, 24.0, 0.03,
    ]
}

fn generate_good_points_with_seed(seed: u64) -> Vec<PhotometryMag> {
    let mut rng_times = SmallRng::seed_from_u64(7);
    let mut rng_noise = SmallRng::seed_from_u64(seed);
    let phys = dense_villar_phys();
    let mut points = Vec::new();

    for &band in &VILLAR_BANDS {
        let anchors = if band == Band::R {
            vec![0.0, 6.0, 18.0]
        } else {
            vec![1.0, 7.0, 17.0]
        };
        let times = sample_times(&mut rng_times, -18.0, 82.0, 4.0, &anchors, 14);
        let pm = if band == Band::R {
            [0, 1, 2, 3, 4, 5, 6]
        } else {
            [7, 8, 9, 10, 11, 12, 13]
        };

        for t in times {
            let obs = Obs {
                phase: t,
                flux: 0.0,
                flux_err: 0.0,
                band,
            };
            let (true_flux, _) = villar_flux_at(&phys, &obs, &pm, 0.0);
            let noisy_flux =
                (true_flux + normal_sample(&mut rng_noise, true_flux * 0.03)).max(1e-6);
            let mag = flux_to_mag(noisy_flux);
            let mag_err = 0.03 * 2.5 / std::f64::consts::LN_10;
            points.push(PhotometryMag {
                time: 60_000.0 + t,
                mag: mag as f32,
                mag_err: mag_err as f32,
                band,
            });
        }
    }

    points
}

fn generate_good_points() -> Vec<PhotometryMag> {
    generate_good_points_with_seed(7)
}

fn generate_bad_points_with_seed(seed: u64) -> Vec<PhotometryMag> {
    let mut rng_times = SmallRng::seed_from_u64(11);
    let mut rng_noise = SmallRng::seed_from_u64(seed);
    let mut points = Vec::new();

    let period_r = 120.0;
    let period_g = 118.0;
    let amp_r = 0.18;
    let amp_g = 0.16;
    let base_r = 1.0;
    let base_g = 0.95;

    for &band in &VILLAR_BANDS {
        let anchors = if band == Band::R {
            vec![60.0]
        } else {
            vec![62.0]
        };
        let times = sample_times(&mut rng_times, 0.0, 360.0, 6.0, &anchors, 20);
        let (period, amp, base, shift) = if band == Band::R {
            (period_r, amp_r, base_r, 60.0)
        } else {
            (period_g, amp_g, base_g, 62.0)
        };

        for t in times {
            let flux =
                base * (1.0 + amp * (2.0 * std::f64::consts::PI * (t - shift) / period).sin());
            let noisy_flux = (flux + normal_sample(&mut rng_noise, base * 0.03)).max(1e-6);
            let mag = flux_to_mag(noisy_flux);
            let mag_err = 0.03 * 2.5 / std::f64::consts::LN_10;
            points.push(PhotometryMag {
                time: 60_000.0 + t,
                mag: mag as f32,
                mag_err: mag_err as f32,
                band,
            });
        }
    }

    points
}

fn generate_bad_points() -> Vec<PhotometryMag> {
    generate_bad_points_with_seed(11)
}

fn standard_test_config_with_strategy(strategy: MultiSeedStrategy) -> PsoConfig {
    PsoConfig {
        n_particles: 56,
        max_iters: 240,
        stall_iters: 45,
        multi_seed_strategy: strategy,
        ..PsoConfig::default()
    }
}

fn standard_test_config() -> PsoConfig {
    standard_test_config_with_strategy(MultiSeedStrategy::EarlyStop)
}

fn fit_cpu_sparse_csv(path: &str) -> Result<f64, String> {
    let config = standard_test_config();
    fit_cpu_sparse_csv_with_config(path, &config)
}

fn fit_cpu_sparse_csv_with_config(path: &str, config: &PsoConfig) -> Result<f64, String> {
    let data = preprocess(path)?;
    let param_map = build_param_map(&data.obs);
    let priors = PriorArrays::new();

    let mut lower = [0.0; N_PARAMS];
    let mut upper = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        lower[i] = priors.mins[i];
        upper[i] = priors.maxs[i];
    }

    let band_indices = BandIndices::new(&data.obs, data.orig_size);
    let mut best_params = [0.0; N_PARAMS];
    let mut best_cost = f64::INFINITY;
    let mut first_cost = f64::INFINITY;

    let cost_fn = |raw: &[f64; N_PARAMS]| {
        pso_cost(
            raw,
            &data.obs,
            &param_map,
            data.orig_size,
            &priors,
            0.0,
            &band_indices,
        )
    };

    for (i, &seed) in MULTI_SEEDS.iter().enumerate() {
        if i >= 2 {
            if matches!(config.multi_seed_strategy, MultiSeedStrategy::EarlyStop)
                && (first_cost - best_cost).abs() < 0.05 * best_cost.abs().max(1e-10)
            {
                break;
            }
        }

        let (params, cost) = pso_minimize(&cost_fn, &lower, &upper, &priors, config, seed);
        if i == 0 {
            first_cost = cost;
        }
        if cost < best_cost {
            best_cost = cost;
            best_params = params;
        }
    }

    Ok(reduced_chi2(
        &best_params,
        &data.obs,
        &param_map,
        data.orig_size,
        &priors,
    ))
}

fn temp_csv_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "villar_pso_{name}_{}_{}.csv",
        std::process::id(),
        rand::random::<u64>()
    ));
    path
}

fn test_strategy_configs() -> (PsoConfig, PsoConfig) {
    let early = standard_test_config_with_strategy(MultiSeedStrategy::EarlyStop);
    let try_all = standard_test_config_with_strategy(MultiSeedStrategy::TryAll);
    (early, try_all)
}

fn write_good_bad_fixture_csv(tag: &str) -> (PathBuf, PathBuf) {
    let good_path = temp_csv_path(&format!("good_{tag}"));
    let bad_path = temp_csv_path(&format!("bad_{tag}"));
    write_csv(&good_path, &generate_good_points());
    write_csv(&bad_path, &generate_bad_points());
    (good_path, bad_path)
}

fn cleanup_good_bad_fixture(good_path: &PathBuf, bad_path: &PathBuf) {
    let _ = fs::remove_file(good_path);
    let _ = fs::remove_file(bad_path);
}

fn assert_quality_good_beats_bad(scope: &str, good: f64, bad: f64) {
    assert!(
        good < 3.0,
        "{scope}: expected a SN-like curve to fit well, got {good:.3}"
    );
    assert!(
        bad > good * 2.5,
        "{scope}: expected a sinusoid to fit worse than SN-like curve: good={good:.3}, bad={bad:.3}"
    );
}

fn assert_strategy_try_all_not_worse(
    scope: &str,
    early_good: f64,
    try_all_good: f64,
    early_bad: f64,
    try_all_bad: f64,
) {
    assert!(
        try_all_good <= early_good + 1e-10,
        "{scope} TryAll should be no worse than EarlyStop on good curve: early={early_good:.6}, try_all={try_all_good:.6}"
    );
    assert!(
        try_all_bad <= early_bad + 1e-10,
        "{scope} TryAll should be no worse than EarlyStop on bad curve: early={early_bad:.6}, try_all={try_all_bad:.6}"
    );
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
const GPU_CPU_MAX_ABS_DELTA: f64 = 0.1;

#[cfg(feature = "cuda")]
fn format_optional_backend_value(name: &str, value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{name}={v:.6}"),
        None => format!("{name}=unavailable"),
    }
}

#[test]
fn synthetic_cpu_quality_good_beats_bad() {
    let (good_path, bad_path) = write_good_bad_fixture_csv("cpu_quality");

    let good = fit_cpu_sparse_csv(good_path.to_str().unwrap()).expect("good curve should fit");
    let bad = fit_cpu_sparse_csv(bad_path.to_str().unwrap()).expect("bad curve should fit");

    cleanup_good_bad_fixture(&good_path, &bad_path);
    assert_quality_good_beats_bad("CPU", good, bad);
}

#[test]
fn synthetic_preprocessing_produces_usable_points() {
    let path = temp_csv_path("preprocess");
    write_csv(&path, &generate_good_points());

    let data = preprocess(path.to_str().unwrap()).expect("preprocessing should work");
    let _ = fs::remove_file(&path);

    assert!(data.orig_size >= 20, "expected sparse but usable data");
    assert!(data.obs.iter().any(|o| o.band == Band::R));
    assert!(data.obs.iter().any(|o| o.band == Band::G));
}

#[test]
fn synthetic_cpu_strategy_try_all_not_worse_than_early_stop() {
    let (early_cfg, try_all_cfg) = test_strategy_configs();

    let (good_path, bad_path) = write_good_bad_fixture_csv("cpu_strategy_cmp");

    let early_good = fit_cpu_sparse_csv_with_config(good_path.to_str().unwrap(), &early_cfg)
        .expect("CPU EarlyStop fit should succeed");
    let try_all_good = fit_cpu_sparse_csv_with_config(good_path.to_str().unwrap(), &try_all_cfg)
        .expect("CPU TryAll fit should succeed");
    let early_bad = fit_cpu_sparse_csv_with_config(bad_path.to_str().unwrap(), &early_cfg)
        .expect("CPU EarlyStop fit should succeed");
    let try_all_bad = fit_cpu_sparse_csv_with_config(bad_path.to_str().unwrap(), &try_all_cfg)
        .expect("CPU TryAll fit should succeed");

    cleanup_good_bad_fixture(&good_path, &bad_path);
    assert_strategy_try_all_not_worse("CPU", early_good, try_all_good, early_bad, try_all_bad);
}

#[cfg(feature = "cuda")]
fn fit_cuda_sparse_csv(path: &str) -> Result<f64, String> {
    let config = standard_test_config();
    fit_cuda_sparse_csv_with_config(path, &config)
}

#[cfg(feature = "cuda")]
fn fit_cuda_sparse_csv_with_config(path: &str, config: &PsoConfig) -> Result<f64, String> {
    let data = preprocess(path)?;
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("synthetic")
        .to_string();

    let source = SourceData { name, data };
    let source_refs = vec![&source];
    let gpu = GpuContext::new(0)?;
    let batch = gpu.pack_batch(&source_refs)?;

    let result = gpu.batch_pso_multi_seed(&batch, &source_refs, config)?;
    Ok(result[0].reduced_chi2)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn fit_metal_sparse_csv(path: &str) -> Result<f64, String> {
    let config = standard_test_config();
    fit_metal_sparse_csv_with_config(path, &config)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn fit_metal_sparse_csv_with_config(path: &str, config: &PsoConfig) -> Result<f64, String> {
    use villar_pso::gpu::{GpuBackend, GpuContext, SourceData};

    let data = preprocess(path)?;
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("synthetic")
        .to_string();

    let source = SourceData { name, data };
    let source_refs = vec![&source];
    let gpu = GpuContext::new_with_backend(0, GpuBackend::Metal)?;
    let batch = gpu.pack_batch(&source_refs)?;

    let result = gpu.batch_pso_multi_seed(&batch, &source_refs, config)?;
    Ok(result[0].reduced_chi2)
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_cuda_quality_good_beats_bad() {
    let (good_path, bad_path) = write_good_bad_fixture_csv("cuda_quality");

    let good = fit_cuda_sparse_csv(good_path.to_str().unwrap())
        .expect("good curve should fit on GPU CUDA");
    let bad =
        fit_cuda_sparse_csv(bad_path.to_str().unwrap()).expect("bad curve should fit on GPU CUDA");

    cleanup_good_bad_fixture(&good_path, &bad_path);
    assert_quality_good_beats_bad("CUDA", good, bad);
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_cuda_strategy_try_all_not_worse_than_early_stop() {
    let (early_cfg, try_all_cfg) = test_strategy_configs();

    let (good_path, bad_path) = write_good_bad_fixture_csv("cuda_strategy_cmp");

    let early_good = fit_cuda_sparse_csv_with_config(good_path.to_str().unwrap(), &early_cfg)
        .expect("CUDA EarlyStop fit should succeed");
    let try_all_good = fit_cuda_sparse_csv_with_config(good_path.to_str().unwrap(), &try_all_cfg)
        .expect("CUDA TryAll fit should succeed");
    let early_bad = fit_cuda_sparse_csv_with_config(bad_path.to_str().unwrap(), &early_cfg)
        .expect("CUDA EarlyStop fit should succeed");
    let try_all_bad = fit_cuda_sparse_csv_with_config(bad_path.to_str().unwrap(), &try_all_cfg)
        .expect("CUDA TryAll fit should succeed");

    cleanup_good_bad_fixture(&good_path, &bad_path);
    assert_strategy_try_all_not_worse("CUDA", early_good, try_all_good, early_bad, try_all_bad);
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_cpu_cuda_point_parity_within_tolerance() {
    let (good_path, bad_path) = write_good_bad_fixture_csv("cpu_cuda_cmp");

    let cpu_good = fit_cpu_sparse_csv(good_path.to_str().unwrap()).expect("CPU fit should succeed");
    let cpu_bad = fit_cpu_sparse_csv(bad_path.to_str().unwrap()).expect("CPU fit should succeed");
    let gpu_good =
        fit_cuda_sparse_csv(good_path.to_str().unwrap()).expect("CUDA fit should succeed");
    let gpu_bad = fit_cuda_sparse_csv(bad_path.to_str().unwrap()).expect("CUDA fit should succeed");

    #[cfg(all(feature = "metal", target_os = "macos"))]
    let metal_good = fit_metal_sparse_csv(good_path.to_str().unwrap()).ok();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let metal_good: Option<f64> = None;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    let metal_bad = fit_metal_sparse_csv(bad_path.to_str().unwrap()).ok();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let metal_bad: Option<f64> = None;

    cleanup_good_bad_fixture(&good_path, &bad_path);

    let good_delta = (cpu_good - gpu_good).abs();
    if good_delta > GPU_CPU_MAX_ABS_DELTA {
        panic!(
            "good curve CPU/CUDA delta too large: abs={good_delta:.6} > {:.3}; cpu={cpu_good:.6}, cuda={gpu_good:.6}, {}",
            GPU_CPU_MAX_ABS_DELTA,
            format_optional_backend_value("metal", metal_good)
        );
    }

    let bad_delta = (cpu_bad - gpu_bad).abs();
    if bad_delta > GPU_CPU_MAX_ABS_DELTA {
        panic!(
            "bad curve CPU/CUDA delta too large: abs={bad_delta:.6} > {:.3}; cpu={cpu_bad:.6}, cuda={gpu_bad:.6}, {}",
            GPU_CPU_MAX_ABS_DELTA,
            format_optional_backend_value("metal", metal_bad)
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_cpu_cuda_distribution_parity_within_tolerance() {
    let strategies = [MultiSeedStrategy::EarlyStop, MultiSeedStrategy::TryAll];

    for strategy in strategies {
        let config = standard_test_config_with_strategy(strategy);

        let mut deltas = Vec::new();
        for i in 0..3 {
            let good_path = temp_csv_path(&format!("good_cuda_dist_{}", i));
            let bad_path = temp_csv_path(&format!("bad_cuda_dist_{}", i));

            write_csv(
                &good_path,
                &generate_good_points_with_seed(7 + i as u64 * 101),
            );
            write_csv(
                &bad_path,
                &generate_bad_points_with_seed(11 + i as u64 * 103),
            );

            let cpu_good = fit_cpu_sparse_csv_with_config(good_path.to_str().unwrap(), &config)
                .expect("CPU fit should succeed");
            let cpu_bad = fit_cpu_sparse_csv_with_config(bad_path.to_str().unwrap(), &config)
                .expect("CPU fit should succeed");
            let cuda_good = fit_cuda_sparse_csv_with_config(good_path.to_str().unwrap(), &config)
                .expect("CUDA fit should succeed");
            let cuda_bad = fit_cuda_sparse_csv_with_config(bad_path.to_str().unwrap(), &config)
                .expect("CUDA fit should succeed");

            deltas.push((cpu_good - cuda_good).abs());
            deltas.push((cpu_bad - cuda_bad).abs());

            cleanup_good_bad_fixture(&good_path, &bad_path);
        }

        let max_delta = deltas.iter().copied().fold(0.0f64, f64::max);
        let med_delta = median(&mut deltas);
        assert!(
            med_delta <= GPU_CPU_MAX_ABS_DELTA,
            "CPU/CUDA median delta too large for strategy {:?}: median={med_delta:.6}, max={max_delta:.6}",
            strategy
        );
        assert!(
            max_delta <= 0.25,
            "CPU/CUDA worst-case delta too large for strategy {:?}: max={max_delta:.6}",
            strategy
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn synthetic_metal_quality_good_beats_bad() {
    let (good_path, bad_path) = write_good_bad_fixture_csv("metal_quality");

    let good = fit_metal_sparse_csv(good_path.to_str().unwrap())
        .expect("good curve should fit on GPU Metal");
    let bad = fit_metal_sparse_csv(bad_path.to_str().unwrap())
        .expect("bad curve should fit on GPU Metal");

    cleanup_good_bad_fixture(&good_path, &bad_path);
    assert_quality_good_beats_bad("Metal", good, bad);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn synthetic_metal_strategy_try_all_not_worse_than_early_stop() {
    let (early_cfg, try_all_cfg) = test_strategy_configs();

    let (good_path, bad_path) = write_good_bad_fixture_csv("metal_strategy_cmp");

    let early_good = fit_metal_sparse_csv_with_config(good_path.to_str().unwrap(), &early_cfg)
        .expect("Metal EarlyStop fit should succeed");
    let try_all_good = fit_metal_sparse_csv_with_config(good_path.to_str().unwrap(), &try_all_cfg)
        .expect("Metal TryAll fit should succeed");
    let early_bad = fit_metal_sparse_csv_with_config(bad_path.to_str().unwrap(), &early_cfg)
        .expect("Metal EarlyStop fit should succeed");
    let try_all_bad = fit_metal_sparse_csv_with_config(bad_path.to_str().unwrap(), &try_all_cfg)
        .expect("Metal TryAll fit should succeed");

    cleanup_good_bad_fixture(&good_path, &bad_path);
    assert_strategy_try_all_not_worse("Metal", early_good, try_all_good, early_bad, try_all_bad);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn synthetic_cpu_metal_distribution_parity_within_tolerance() {
    let strategies = [MultiSeedStrategy::EarlyStop, MultiSeedStrategy::TryAll];

    for strategy in strategies {
        let config = PsoConfig {
            n_particles: 56,
            max_iters: 240,
            stall_iters: 45,
            multi_seed_strategy: strategy,
            ..PsoConfig::default()
        };

        let mut deltas = Vec::new();
        for i in 0..3 {
            let good_path = temp_csv_path(&format!("good_dist_{}", i));
            let bad_path = temp_csv_path(&format!("bad_dist_{}", i));

            write_csv(
                &good_path,
                &generate_good_points_with_seed(7 + i as u64 * 101),
            );
            write_csv(
                &bad_path,
                &generate_bad_points_with_seed(11 + i as u64 * 103),
            );

            let cpu_good = fit_cpu_sparse_csv_with_config(good_path.to_str().unwrap(), &config)
                .expect("CPU fit should succeed");
            let cpu_bad = fit_cpu_sparse_csv_with_config(bad_path.to_str().unwrap(), &config)
                .expect("CPU fit should succeed");
            let metal_good = fit_metal_sparse_csv_with_config(good_path.to_str().unwrap(), &config)
                .expect("Metal fit should succeed");
            let metal_bad = fit_metal_sparse_csv_with_config(bad_path.to_str().unwrap(), &config)
                .expect("Metal fit should succeed");

            deltas.push((cpu_good - metal_good).abs());
            deltas.push((cpu_bad - metal_bad).abs());

            let _ = fs::remove_file(&good_path);
            let _ = fs::remove_file(&bad_path);
        }

        let max_delta = deltas.iter().copied().fold(0.0f64, f64::max);
        let med_delta = median(&mut deltas);
        assert!(
            med_delta <= GPU_CPU_MAX_ABS_DELTA,
            "CPU/Metal median delta too large for strategy {:?}: median={med_delta:.6}, max={max_delta:.6}",
            strategy
        );
        assert!(
            max_delta <= 0.25,
            "CPU/Metal worst-case delta too large for strategy {:?}: max={max_delta:.6}",
            strategy
        );
    }
}

#[cfg(all(feature = "cuda", feature = "metal", target_os = "macos"))]
#[test]
fn synthetic_cpu_cuda_metal_triplet_point_parity_within_tolerance() {
    let good_path = temp_csv_path("good_triplet_cmp");
    let bad_path = temp_csv_path("bad_triplet_cmp");

    write_csv(&good_path, &generate_good_points());
    write_csv(&bad_path, &generate_bad_points());

    let cpu_good = fit_cpu_sparse_csv(good_path.to_str().unwrap()).expect("CPU fit should succeed");
    let cuda_good =
        fit_cuda_sparse_csv(good_path.to_str().unwrap()).expect("CUDA fit should succeed");
    let metal_good =
        fit_metal_sparse_csv(good_path.to_str().unwrap()).expect("Metal fit should succeed");

    let cpu_bad = fit_cpu_sparse_csv(bad_path.to_str().unwrap()).expect("CPU fit should succeed");
    let cuda_bad =
        fit_cuda_sparse_csv(bad_path.to_str().unwrap()).expect("CUDA fit should succeed");
    let metal_bad =
        fit_metal_sparse_csv(bad_path.to_str().unwrap()).expect("Metal fit should succeed");

    let _ = fs::remove_file(&good_path);
    let _ = fs::remove_file(&bad_path);

    assert!(
        (cpu_good - cuda_good).abs() <= GPU_CPU_MAX_ABS_DELTA
            && (cpu_good - metal_good).abs() <= GPU_CPU_MAX_ABS_DELTA,
        "good curve triplet mismatch: cpu={cpu_good:.6}, cuda={cuda_good:.6}, metal={metal_good:.6}"
    );
    assert!(
        (cpu_bad - cuda_bad).abs() <= GPU_CPU_MAX_ABS_DELTA
            && (cpu_bad - metal_bad).abs() <= GPU_CPU_MAX_ABS_DELTA,
        "bad curve triplet mismatch: cpu={cpu_bad:.6}, cuda={cuda_bad:.6}, metal={metal_bad:.6}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn synthetic_metal_backend_smoke_runs() {
    use villar_pso::gpu::{GpuBackend, GpuContext, SourceData};

    let path = temp_csv_path("metal_stub");
    write_csv(&path, &generate_good_points());

    let data = preprocess(path.to_str().unwrap()).expect("preprocessing should work");
    let source = SourceData {
        name: "metal_stub".to_string(),
        data,
    };
    let source_refs = vec![&source];

    let ctx = GpuContext::new_with_backend(0, GpuBackend::Metal)
        .expect("metal context construction should succeed");

    #[cfg(target_os = "macos")]
    {
        let batch = ctx
            .pack_batch(&source_refs)
            .expect("metal batch packing should work on macOS");
        let config = PsoConfig {
            n_particles: 24,
            max_iters: 80,
            stall_iters: 20,
            ..PsoConfig::default()
        };
        let results = ctx
            .batch_pso_multi_seed(&batch, &source_refs, &config)
            .expect("metal pso should run on macOS");
        assert!(results[0].reduced_chi2.is_finite());
    }

    #[cfg(not(target_os = "macos"))]
    {
        let err = match ctx.pack_batch(&source_refs) {
            Ok(_) => panic!("metal backend should be unavailable on non-macOS"),
            Err(err) => err,
        };
        assert!(
            err.contains("requires macOS") || err.contains("not implemented"),
            "expected non-macOS metal error, got: {err}"
        );
    }

    let _ = fs::remove_file(&path);
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_cuda_backend_smoke_runs() {
    use villar_pso::gpu::{GpuContext, SourceData};

    let path = temp_csv_path("cuda_stub");
    write_csv(&path, &generate_good_points());

    let data = preprocess(path.to_str().unwrap()).expect("preprocessing should work");
    let source = SourceData {
        name: "cuda_stub".to_string(),
        data,
    };
    let source_refs = vec![&source];

    let ctx = GpuContext::new(0).expect("cuda context construction should succeed");
    let batch = ctx
        .pack_batch(&source_refs)
        .expect("cuda batch packing should work");
    let config = PsoConfig {
        n_particles: 24,
        max_iters: 80,
        stall_iters: 20,
        ..PsoConfig::default()
    };
    let results = ctx
        .batch_pso_multi_seed(&batch, &source_refs, &config)
        .expect("cuda pso should run");
    assert!(results[0].reduced_chi2.is_finite());

    let _ = fs::remove_file(&path);
}
