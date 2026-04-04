use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use villar_pso::{
    build_param_map, preprocess, pso_cost, pso_minimize, reduced_chi2, villar_flux_at, Band,
    BandIndices, Obs, PhotometryMag, PriorArrays, PsoConfig, N_PARAMS, VILLAR_BANDS,
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

fn generate_good_points() -> Vec<PhotometryMag> {
    let mut rng = SmallRng::seed_from_u64(7);
    let phys = dense_villar_phys();
    let mut points = Vec::new();

    for &band in &VILLAR_BANDS {
        let anchors = if band == Band::R {
            vec![0.0, 6.0, 18.0]
        } else {
            vec![1.0, 7.0, 17.0]
        };
        let times = sample_times(&mut rng, -18.0, 82.0, 4.0, &anchors, 14);
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
            let noisy_flux = (true_flux + normal_sample(&mut rng, true_flux * 0.03)).max(1e-6);
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
    let mut rng = SmallRng::seed_from_u64(11);
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
        let times = sample_times(&mut rng, 0.0, 360.0, 6.0, &anchors, 20);
        let (period, amp, base, shift) = if band == Band::R {
            (period_r, amp_r, base_r, 60.0)
        } else {
            (period_g, amp_g, base_g, 62.0)
        };

        for t in times {
            let flux =
                base * (1.0 + amp * (2.0 * std::f64::consts::PI * (t - shift) / period).sin());
            let noisy_flux = (flux + normal_sample(&mut rng, base * 0.03)).max(1e-6);
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

fn fit_sparse_csv(path: &str) -> Result<f64, String> {
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
    let config = PsoConfig {
        n_particles: 56,
        max_iters: 240,
        stall_iters: 45,
        ..PsoConfig::default()
    };

    let seeds = [42_u64, 137_u64, 271_u64];
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

    for (i, &seed) in seeds.iter().enumerate() {
        if i >= 2 && (first_cost - best_cost).abs() < 0.05 * best_cost.abs().max(1e-10) {
            break;
        }

        let (params, cost) = pso_minimize(&cost_fn, &lower, &upper, &priors, &config, seed);
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

#[test]
fn synthetic_good_curve_fits_better_than_bad_curve_cpu() {
    let good_path = temp_csv_path("good");
    let bad_path = temp_csv_path("bad");

    write_csv(&good_path, &generate_good_points());
    write_csv(&bad_path, &generate_bad_points());

    let good = fit_sparse_csv(good_path.to_str().unwrap()).expect("good curve should fit");
    let bad = fit_sparse_csv(bad_path.to_str().unwrap()).expect("bad curve should fit");

    let _ = fs::remove_file(&good_path);
    let _ = fs::remove_file(&bad_path);

    assert!(
        good < 3.0,
        "expected a SN-like curve to fit well, got {good:.3}"
    );
    assert!(
        bad > good * 2.5,
        "expected a sinusoid to fit worse than the SN-like curve: good={good:.3}, bad={bad:.3}"
    );
}

#[test]
fn synthetic_inputs_survive_preprocessing() {
    let path = temp_csv_path("preprocess");
    write_csv(&path, &generate_good_points());

    let data = preprocess(path.to_str().unwrap()).expect("preprocessing should work");
    let _ = fs::remove_file(&path);

    assert!(data.orig_size >= 20, "expected sparse but usable data");
    assert!(data.obs.iter().any(|o| o.band == Band::R));
    assert!(data.obs.iter().any(|o| o.band == Band::G));
}

#[cfg(feature = "cuda")]
fn fit_sparse_csv_gpu_cuda(path: &str) -> Result<f64, String> {
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
    let config = PsoConfig {
        n_particles: 56,
        max_iters: 240,
        stall_iters: 45,
        ..PsoConfig::default()
    };

    let result = gpu.batch_pso_multi_seed(&batch, &source_refs, &config)?;
    Ok(result[0].reduced_chi2)
}

#[cfg(feature = "cuda")]
#[test]
fn synthetic_good_curve_fits_better_than_bad_curve_gpu_cuda() {
    let good_path = temp_csv_path("good_gpu");
    let bad_path = temp_csv_path("bad_gpu");

    write_csv(&good_path, &generate_good_points());
    write_csv(&bad_path, &generate_bad_points());

    let good = fit_sparse_csv_gpu_cuda(good_path.to_str().unwrap())
        .expect("good curve should fit on GPU CUDA");
    let bad = fit_sparse_csv_gpu_cuda(bad_path.to_str().unwrap())
        .expect("bad curve should fit on GPU CUDA");

    let _ = fs::remove_file(&good_path);
    let _ = fs::remove_file(&bad_path);

    assert!(
        good < 3.0,
        "expected a SN-like curve to fit well on GPU, got {good:.3}"
    );
    assert!(
        bad > good * 2.5,
        "expected a sinusoid to fit worse than the SN-like curve on GPU: good={good:.3}, bad={bad:.3}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn synthetic_metal_backend_returns_not_implemented_yet() {
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
    let err = ctx
        .pack_batch(&source_refs)
        .expect_err("metal backend should be stubbed for now");

    let _ = fs::remove_file(&path);

    assert!(
        err.contains("not implemented yet"),
        "expected not implemented error, got: {err}"
    );
}
