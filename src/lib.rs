//! Joint two-band Villar light-curve fitter (PSO).
//!
//! Matches the preprocessing and model from `fit_jax_best.py` exactly.
//! Replaces NumPyro SVI with Particle Swarm Optimisation over the 14-parameter
//! space (7 r-band + 7 g-band), minimising reduced chi-squared.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub mod gpu;

// ─── Prior tables (from fit_jax_best.py) ─────────────────────────────────────
// Each entry: (min, max, mean, std, logged)
// logged = true → parameter lives in log10 space; 10^x applied after sampling.

/// 7 base parameter names, in order.
pub const PARAM_NAMES: [&str; 7] = [
    "A",
    "beta",
    "gamma",
    "t_0",
    "tau_rise",
    "tau_fall",
    "extra_sigma",
];

pub const N_BASE: usize = 7;
pub const N_PARAMS: usize = 14; // 7 r + 7 g

pub const FILTERS: [&str; 2] = ["ZTF_r", "ZTF_g"];

// (min, max, mean, std, logged)
// JAX-calibrated priors: ranges and means from fit_jax_best.py on reference sources.
// log10 ranges include ~2 sigma margin beyond observed JAX fits.
const PRIOR_R: [(f64, f64, f64, f64, bool); 7] = [
    // A:  JAX log10 range [-0.04, 0.01], mean -0.009
    (-0.20, 0.15, -0.009, 0.100, true),    // A  (phys 0.63–1.41)
    (-0.01, 0.03, 0.009, 0.010, false),    // beta
    // gamma: JAX log10 range [1.21, 1.60], mean 1.33
    (0.80, 2.00, 1.327, 0.400, true),      // gamma (phys 6.3–100)
    (-50.0, 30.0, -12.0, 15.000, false),   // t_0
    // tau_rise: JAX log10 range [0.17, 0.49], mean 0.36
    (-0.30, 1.00, 0.360, 0.350, true),     // tau_rise (phys 0.5–10)
    // tau_fall: JAX log10 range [1.17, 1.87], mean 1.45
    (0.80, 2.20, 1.454, 0.400, true),      // tau_fall (phys 6.3–158)
    (-2.50, -0.30, -1.033, 0.500, true),   // extra_sigma
];

// g-band: relative offsets from r-band.
// JAX-calibrated from observed offset distributions.
const PRIOR_G: [(f64, f64, f64, f64, bool); 7] = [
    // A offset: JAX range [-0.07, +0.04], mean -0.015, std 0.036
    (-0.30, 0.20, -0.015, 0.080, true),    // A  (keep g close to r)
    (-0.02, 0.02, -0.001, 0.008, false),   // beta
    // gamma offset: JAX range [-0.47, +0.09], mean -0.095, std 0.22
    (-0.80, 0.50, -0.095, 0.250, true),    // gamma
    // t_0 offset: JAX range [-2.3, +0.1], mean -1.18, std 0.86
    (-5.00, 3.0, -1.181, 1.500, false),    // t_0
    // tau_rise offset: JAX range [-0.63, +0.09], mean -0.195, std 0.28
    (-1.00, 0.50, -0.195, 0.350, true),    // tau_rise
    // tau_fall offset: JAX range [-0.44, -0.22], mean -0.323, std 0.10
    (-0.80, 0.30, -0.323, 0.250, true),    // tau_fall
    // extra_sigma offset: JAX range [-0.29, +0.32], mean -0.025, std 0.23
    (-0.60, 0.60, -0.025, 0.300, true),    // extra_sigma
];

/// Flattened prior arrays: indices 0..7 = r-band, 7..14 = g-band (absolute for g).
pub struct PriorArrays {
    pub mins: [f64; N_PARAMS],
    pub maxs: [f64; N_PARAMS],
    pub means: [f64; N_PARAMS],
    pub stds: [f64; N_PARAMS],
    pub logged: [bool; N_PARAMS],
}

impl PriorArrays {
    pub fn new() -> Self {
        let mut pa = PriorArrays {
            mins: [0.0; N_PARAMS],
            maxs: [0.0; N_PARAMS],
            means: [0.0; N_PARAMS],
            stds: [0.0; N_PARAMS],
            logged: [false; N_PARAMS],
        };
        for i in 0..N_BASE {
            let (mn, mx, mu, sg, lg) = PRIOR_R[i];
            pa.mins[i] = mn;
            pa.maxs[i] = mx;
            pa.means[i] = mu;
            pa.stds[i] = sg;
            pa.logged[i] = lg;
        }
        // g-band stored as OFFSETS from r-band (matching JAX relative_samples).
        // PSO search space:
        //   raw[0..7]  = r-band raw values
        //   raw[7..14] = g-band OFFSETS (bounded by PRIOR_G)
        // Absolute g-band = raw[i] + raw[N_BASE + i], then 10^x for logged.
        for i in 0..N_BASE {
            let (mn, mx, mu, sg, lg) = PRIOR_G[i];
            pa.mins[N_BASE + i] = mn;
            pa.maxs[N_BASE + i] = mx;
            pa.means[N_BASE + i] = mu;
            pa.stds[N_BASE + i] = sg;
            pa.logged[N_BASE + i] = lg;
        }
        pa
    }

    /// Indices where 10^x transform is applied.
    pub fn logged_indices(&self) -> Vec<usize> {
        (0..N_PARAMS).filter(|&i| self.logged[i]).collect()
    }
}

/// Convert raw search vector (r-band + g-band offsets) to absolute raw values.
/// raw[0..7] stays as-is (r-band), raw[7..14] = r + offset → absolute g-band.
pub fn offsets_to_absolute(raw: &[f64; N_PARAMS]) -> [f64; N_PARAMS] {
    let mut abs = *raw;
    for i in 0..N_BASE {
        abs[N_BASE + i] = raw[i] + raw[N_BASE + i];
    }
    abs
}

// ─── Photometry helpers ──────────────────────────────────────────────────────

/// Convert magnitude + error to flux + flux_error (zeropoint = 23.9).
pub fn mag_to_flux(mag: f64, mag_err: f64) -> (f64, f64) {
    let zp = 23.9;
    let flux = ((zp - mag) / 2.5 * std::f64::consts::LN_10).exp();
    let flux_err = flux * mag_err * std::f64::consts::LN_10 / 2.5;
    (flux, flux_err)
}

/// Observation for a single photometric point.
#[derive(Debug, Clone)]
pub struct Obs {
    pub phase: f64,
    pub flux: f64,
    pub flux_err: f64,
    pub band: usize, // 0 = ZTF_r, 1 = ZTF_g
}

/// Raw CSV row.
#[derive(Debug, serde::Deserialize)]
pub struct RawRow {
    #[serde(alias = "jd")]
    pub mjd: Option<f64>,
    #[serde(alias = "magpsf")]
    pub mag: Option<f64>,
    #[serde(alias = "sigmapsf")]
    pub mag_err: Option<f64>,
    #[serde(alias = "fid")]
    pub fid: Option<u32>,
    pub filter: Option<String>,
}

/// Weighted-average observations within `eps` days per filter.
fn merge_close_times(obs: &mut Vec<Obs>, eps: f64) {
    let mut merged = Vec::new();
    for band_idx in 0..2 {
        let mut band_obs: Vec<&Obs> = obs.iter().filter(|o| o.band == band_idx).collect();
        band_obs.sort_by(|a, b| a.phase.partial_cmp(&b.phase).unwrap());

        let mut i = 0;
        while i < band_obs.len() {
            let mut j = i + 1;
            while j < band_obs.len() && band_obs[j].phase - band_obs[i].phase < eps {
                j += 1;
            }
            // Weighted average of obs[i..j]
            let mut w_sum = 0.0;
            let mut wt_phase = 0.0;
            let mut wt_flux = 0.0;
            for k in i..j {
                let w = 1.0 / (band_obs[k].flux_err * band_obs[k].flux_err);
                w_sum += w;
                wt_phase += band_obs[k].phase * w;
                wt_flux += band_obs[k].flux * w;
            }
            merged.push(Obs {
                phase: wt_phase / w_sum,
                flux: wt_flux / w_sum,
                flux_err: (1.0 / w_sum).sqrt(),
                band: band_idx,
            });
            i = j;
        }
    }
    *obs = merged;
}

/// Full preprocessing: CSV path → processed observations + metadata.
///
/// Steps (matching fit_jax_best.py exactly):
/// 1. Map filter names
/// 2. mag → flux (zp=23.9)
/// 3. Phase relative to brightest r-band
/// 4. Merge close times (eps=0.04 days)
/// 5. Truncate to [-50, 100] days
/// 6. Normalise by peak flux
/// 7. Pad each filter to equal length
pub fn preprocess(csv_path: &str) -> Result<PreprocessedData, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(csv_path)
        .map_err(|e| format!("Cannot open CSV: {}", e))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("Cannot read headers: {}", e))?
        .clone();

    // Determine column indices
    let col_idx = |name: &str| -> Option<usize> {
        headers.iter().position(|h| h == name)
    };

    let has_jd = col_idx("jd").is_some();
    let has_mjd = col_idx("mjd").is_some();
    let has_fid = col_idx("fid").is_some();
    let has_filter = col_idx("filter").is_some();
    let has_magpsf = col_idx("magpsf").is_some();
    let has_sigmapsf = col_idx("sigmapsf").is_some();
    let has_mag = col_idx("mag").is_some();
    let has_mag_err = col_idx("mag_err").is_some();

    // Get the right column indices
    let time_col = if has_mjd {
        col_idx("mjd").unwrap()
    } else if has_jd {
        col_idx("jd").unwrap()
    } else {
        return Err("CSV must have 'mjd' or 'jd' column".to_string());
    };

    let mag_col = if has_mag {
        col_idx("mag").unwrap()
    } else if has_magpsf {
        col_idx("magpsf").unwrap()
    } else {
        return Err("CSV must have 'mag' or 'magpsf' column".to_string());
    };

    let magerr_col = if has_mag_err {
        col_idx("mag_err").unwrap()
    } else if has_sigmapsf {
        col_idx("sigmapsf").unwrap()
    } else {
        return Err("CSV must have 'mag_err' or 'sigmapsf' column".to_string());
    };

    // Parse rows
    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut mag_errs = Vec::new();
    let mut band_indices = Vec::new();

    for result in rdr.records() {
        let record = result.map_err(|e| format!("CSV parse error: {}", e))?;

        let time_str = record.get(time_col).unwrap_or("");
        let mag_str = record.get(mag_col).unwrap_or("");
        let magerr_str = record.get(magerr_col).unwrap_or("");

        let time: f64 = match time_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mag: f64 = match mag_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mag_err: f64 = match magerr_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Determine band
        let band_idx = if has_fid && !has_filter {
            let fid_str = record.get(col_idx("fid").unwrap()).unwrap_or("");
            match fid_str.parse::<u32>() {
                Ok(1) => 1, // fid=1 → g → band 1
                Ok(2) => 0, // fid=2 → r → band 0
                _ => continue,
            }
        } else if has_filter {
            let filt = record.get(col_idx("filter").unwrap()).unwrap_or("");
            match filt {
                "r" | "ZTF_r" => 0,
                "g" | "ZTF_g" => 1,
                _ => continue,
            }
        } else {
            continue;
        };

        // Convert JD → MJD if needed
        let mjd = if has_jd && !has_mjd {
            time - 2400000.5
        } else {
            time
        };

        times.push(mjd);
        mags.push(mag);
        mag_errs.push(mag_err);
        band_indices.push(band_idx);
    }

    if times.is_empty() {
        return Err("No valid r/g band data in CSV.".to_string());
    }

    // mag → flux
    let mut obs: Vec<Obs> = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        let (flux, flux_err) = mag_to_flux(mags[i], mag_errs[i]);
        if !flux.is_finite() || !flux_err.is_finite() || flux <= 0.0 || flux_err <= 0.0 {
            continue;
        }
        obs.push(Obs {
            phase: times[i], // still MJD, will convert to phase below
            flux,
            flux_err,
            band: band_indices[i],
        });
    }

    // Phase: t=0 at brightest r-band point
    let r_obs: Vec<&Obs> = obs.iter().filter(|o| o.band == 0).collect();
    if r_obs.is_empty() {
        return Err("No r-band data.".to_string());
    }
    let t0 = r_obs
        .iter()
        .max_by(|a, b| a.flux.partial_cmp(&b.flux).unwrap())
        .unwrap()
        .phase;
    for o in obs.iter_mut() {
        o.phase -= t0;
    }

    // Merge close times
    merge_close_times(&mut obs, 0.04);

    // Truncate to [-50, 100]
    obs.retain(|o| o.phase >= -50.0 && o.phase <= 100.0);

    let n_r = obs.iter().filter(|o| o.band == 0).count();
    let n_g = obs.iter().filter(|o| o.band == 1).count();
    if n_r <= 2 {
        return Err(format!(
            "Only {} points in ZTF_r after preprocessing (need >2).",
            n_r
        ));
    }
    if n_g <= 2 {
        return Err(format!(
            "Only {} points in ZTF_g after preprocessing (need >2).",
            n_g
        ));
    }

    // Skip MW extinction (E(B-V)=0 assumed, matching JAX fallback)

    // Normalise by peak flux
    let peak = obs
        .iter()
        .map(|o| o.flux)
        .fold(f64::NEG_INFINITY, f64::max);
    if peak <= 0.0 {
        return Err("Non-positive peak flux.".to_string());
    }
    for o in obs.iter_mut() {
        o.flux /= peak;
        o.flux_err /= peak;
    }

    let orig_size = obs.len();

    // Pad each filter to equal length (next power-of-2 / 2)
    let n_total = (orig_size as f64).log2().ceil().exp2() as usize;
    let n_per_filt = n_total / 2;
    for band_idx in 0..2 {
        let n_filt = obs.iter().filter(|o| o.band == band_idx).count();
        let n_extra = if n_per_filt > n_filt {
            n_per_filt - n_filt
        } else {
            0
        };
        for _ in 0..n_extra {
            obs.push(Obs {
                phase: 1000.0,
                flux: 0.1,
                flux_err: 1000.0,
                band: band_idx,
            });
        }
    }

    // Sort by (band, phase)
    obs.sort_by(|a, b| {
        a.band
            .cmp(&b.band)
            .then(a.phase.partial_cmp(&b.phase).unwrap())
    });

    Ok(PreprocessedData { obs, orig_size })
}

pub struct PreprocessedData {
    pub obs: Vec<Obs>,
    pub orig_size: usize,
}

// ─── Parameter map ───────────────────────────────────────────────────────────

/// Build param_map: (7, n_times) where param_map[i][j] = flat index for parameter i at time j.
/// r-band → 0..7, g-band → 7..14.
pub fn build_param_map(obs: &[Obs]) -> Vec<[usize; 7]> {
    obs.iter()
        .map(|o| {
            let offset = o.band * N_BASE;
            [
                offset,
                offset + 1,
                offset + 2,
                offset + 3,
                offset + 4,
                offset + 5,
                offset + 6,
            ]
        })
        .collect()
}

// ─── Villar model ────────────────────────────────────────────────────────────

/// Transform raw search vector to physical space.
/// First converts g-band offsets to absolute, then applies 10^x to logged params.
pub fn to_physical(raw: &[f64; N_PARAMS], priors: &PriorArrays) -> [f64; N_PARAMS] {
    let abs = offsets_to_absolute(raw);
    let mut phys = abs;
    for i in 0..N_PARAMS {
        if priors.logged[i] {
            phys[i] = (abs[i] * std::f64::consts::LN_10).exp();
        }
    }
    phys
}

/// Evaluate Villar flux at one observation point.
/// `phys` is in physical (exponentiated) space, length 14.
/// If `min_extra_sigma > 0`, clamp extra_sigma to at least that value.
#[inline]
pub fn villar_flux_at(
    phys: &[f64; N_PARAMS],
    obs: &Obs,
    pm: &[usize; 7],
    min_extra_sigma: f64,
) -> (f64, f64) {
    let amp = phys[pm[0]];
    let beta = phys[pm[1]];
    let gamma = phys[pm[2]].max(0.0);
    let t_0 = phys[pm[3]];
    let tau_rise = phys[pm[4]];
    let tau_fall = phys[pm[5]];
    let extra_sigma = phys[pm[6]].max(min_extra_sigma);

    let phase = (obs.phase - t_0).max(-50.0 * tau_rise);
    let f_const = amp / (1.0 + (-phase / tau_rise).exp());
    let flux = if gamma - phase >= 0.0 {
        f_const * (1.0 - beta * phase)
    } else {
        f_const * (1.0 - beta * gamma) * (-(phase - gamma) / tau_fall).exp()
    };
    (flux, extra_sigma)
}

/// Physical validity constraint (matches JAX `_villar_constraint`).
/// Evaluates for one band's worth of physical params.
fn villar_constraint_band(phys: &[f64; N_BASE]) -> f64 {
    let beta = phys[1];
    let gamma = phys[2].max(0.0);
    let tau_rise = phys[4];
    let tau_fall = phys[5];

    let c1 = (gamma * beta - 1.0).max(0.0);
    let c2 = ((-gamma / tau_rise).exp() * (tau_fall / tau_rise - 1.0) - 1.0).max(0.0);
    let c3 = (beta * tau_fall - 1.0 + beta * gamma).max(0.0);
    c1 + c2 + c3
}

// ─── PSO cost function ───────────────────────────────────────────────────────

/// Reduced chi-squared for a candidate parameter vector (in raw/log10 space).
pub fn reduced_chi2(
    raw: &[f64; N_PARAMS],
    obs: &[Obs],
    param_map: &[[usize; 7]],
    orig_size: usize,
    priors: &PriorArrays,
) -> f64 {
    let phys = to_physical(raw, priors);

    // Constraint penalty (evaluate per band, take max like JAX)
    let r_params: [f64; N_BASE] = std::array::from_fn(|i| phys[i]);
    let g_params: [f64; N_BASE] = std::array::from_fn(|i| phys[N_BASE + i]);
    let constraint = villar_constraint_band(&r_params).max(villar_constraint_band(&g_params));
    if constraint > 0.0 {
        return 1e12 + 10_000.0 * constraint;
    }

    let mut chi2_sum = 0.0;
    let dof = (orig_size as f64 - N_PARAMS as f64).max(1.0);

    for (i, ob) in obs.iter().enumerate().take(orig_size) {
        let (model_flux, extra_sigma) = villar_flux_at(&phys, ob, &param_map[i], 0.0);
        if !model_flux.is_finite() {
            return 1e12;
        }
        let sigma2 = ob.flux_err * ob.flux_err + extra_sigma * extra_sigma;
        let diff = ob.flux - model_flux;
        chi2_sum += diff * diff / sigma2;
    }

    chi2_sum / dof
}

/// Precomputed per-band observation indices for branch-free cost evaluation.
pub struct BandIndices {
    pub r_indices: Vec<usize>,
    pub g_indices: Vec<usize>,
}

impl BandIndices {
    pub fn new(obs: &[Obs], orig_size: usize) -> Self {
        let mut r_indices = Vec::new();
        let mut g_indices = Vec::new();
        for i in 0..orig_size {
            if obs[i].band == 0 {
                r_indices.push(i);
            } else {
                g_indices.push(i);
            }
        }
        BandIndices { r_indices, g_indices }
    }
}

/// PSO cost with prior penalty (for guiding the search).
/// raw[0..7] = r-band values, raw[7..14] = g-band offsets from r-band.
/// Uses band-balanced likelihood so each filter contributes equally.
/// `min_extra_sigma`: floor for extra_sigma in physical space (smooths cost landscape).
pub fn pso_cost(
    raw: &[f64; N_PARAMS],
    obs: &[Obs],
    param_map: &[[usize; 7]],
    _orig_size: usize,
    priors: &PriorArrays,
    min_extra_sigma: f64,
    band_indices: &BandIndices,
) -> f64 {
    let phys = to_physical(raw, priors);

    // Constraint penalty
    let r_params: [f64; N_BASE] = std::array::from_fn(|i| phys[i]);
    let g_params: [f64; N_BASE] = std::array::from_fn(|i| phys[N_BASE + i]);
    let constraint = villar_constraint_band(&r_params).max(villar_constraint_band(&g_params));

    if constraint > 0.0 {
        return 1e12 + 10_000.0 * constraint;
    }

    // Band-balanced data likelihood — branch-free loops over precomputed indices
    let mut ll_r = 0.0;
    for &i in &band_indices.r_indices {
        let (model_flux, extra_sigma) =
            villar_flux_at(&phys, &obs[i], &param_map[i], min_extra_sigma);
        if !model_flux.is_finite() {
            return 1e12;
        }
        let sigma2 = obs[i].flux_err * obs[i].flux_err + extra_sigma * extra_sigma;
        let diff = obs[i].flux - model_flux;
        ll_r += diff * diff / sigma2;
    }

    let mut ll_g = 0.0;
    for &i in &band_indices.g_indices {
        let (model_flux, extra_sigma) =
            villar_flux_at(&phys, &obs[i], &param_map[i], min_extra_sigma);
        if !model_flux.is_finite() {
            return 1e12;
        }
        let sigma2 = obs[i].flux_err * obs[i].flux_err + extra_sigma * extra_sigma;
        let diff = obs[i].flux - model_flux;
        ll_g += diff * diff / sigma2;
    }

    let n_r = band_indices.r_indices.len();
    let n_g = band_indices.g_indices.len();
    let n_total = (n_r + n_g) as f64;

    let mut neg_ll = 0.0;
    if n_r > 0 {
        neg_ll += (ll_r / n_r as f64) * (n_total / 2.0);
    }
    if n_g > 0 {
        neg_ll += (ll_g / n_g as f64) * (n_total / 2.0);
    }

    // Prior penalty (standard 0.5 weight)
    for i in 0..N_PARAMS {
        let z = (raw[i] - priors.means[i]) / priors.stds[i];
        neg_ll += 0.5 * z * z;
    }

    neg_ll
}

// ─── PSO minimizer ───────────────────────────────────────────────────────────

pub struct PsoConfig {
    pub n_particles: usize,
    pub max_iters: usize,
    pub stall_iters: usize,
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
}

impl Default for PsoConfig {
    fn default() -> Self {
        PsoConfig {
            n_particles: 200,
            max_iters: 1500,
            stall_iters: 60,
            w: 0.9,   // initial inertia (decays to 0.4 over iterations)
            c1: 1.5,
            c2: 1.5,
        }
    }
}

pub fn pso_minimize(
    cost_fn: &dyn Fn(&[f64; N_PARAMS]) -> f64,
    lower: &[f64; N_PARAMS],
    upper: &[f64; N_PARAMS],
    priors: &PriorArrays,
    config: &PsoConfig,
    seed: u64,
) -> ([f64; N_PARAMS], f64) {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut positions: Vec<[f64; N_PARAMS]> = Vec::with_capacity(config.n_particles);
    let mut velocities: Vec<[f64; N_PARAMS]> = Vec::with_capacity(config.n_particles);
    let mut pbest_pos: Vec<[f64; N_PARAMS]> = Vec::with_capacity(config.n_particles);
    let mut pbest_cost: Vec<f64> = Vec::with_capacity(config.n_particles);
    let mut gbest_pos = [0.0; N_PARAMS];
    let mut gbest_cost = f64::INFINITY;

    // Initialize swarm: 30% near prior means (Gaussian jitter), 70% uniform random.
    // Seeding near the prior means helps PSO find the physically-reasonable basin.
    let n_seeded = config.n_particles * 3 / 10;
    let means = &priors.means;
    let stds = &priors.stds;
    for p in 0..config.n_particles {
        let mut pos = [0.0; N_PARAMS];
        let mut vel = [0.0; N_PARAMS];
        if p < n_seeded {
            // Sample from a Gaussian centred on the prior means (Box-Muller).
            for d in 0..N_PARAMS {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                pos[d] = (means[d] + z * stds[d]).clamp(lower[d], upper[d]);
            }
        } else {
            for d in 0..N_PARAMS {
                pos[d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
            }
        }
        for d in 0..N_PARAMS {
            vel[d] = (upper[d] - lower[d]) * 0.1 * (2.0 * rng.random::<f64>() - 1.0);
        }
        let cost = cost_fn(&pos);
        if cost < gbest_cost {
            gbest_cost = cost;
            gbest_pos = pos;
        }
        pbest_cost.push(cost);
        pbest_pos.push(pos);
        positions.push(pos);
        velocities.push(vel);
    }

    let mut iters_without_improvement = 0usize;
    let mut prev_gbest = gbest_cost;
    let w_start = config.w;
    let w_end = 0.4;
    let restart_threshold = 40; // reinitialize some particles after this many stall iters
    let restart_fraction = 0.3; // fraction of particles to reinitialize

    for iter in 0..config.max_iters {
        // Linearly decay inertia weight from w_start to w_end
        let w = w_start - (w_start - w_end) * (iter as f64 / config.max_iters as f64);

        for p in 0..config.n_particles {
            for d in 0..N_PARAMS {
                let r1: f64 = rng.random();
                let r2: f64 = rng.random();
                velocities[p][d] = w * velocities[p][d]
                    + config.c1 * r1 * (pbest_pos[p][d] - positions[p][d])
                    + config.c2 * r2 * (gbest_pos[d] - positions[p][d]);
                positions[p][d] =
                    (positions[p][d] + velocities[p][d]).clamp(lower[d], upper[d]);
            }
            let cost = cost_fn(&positions[p]);
            if cost < pbest_cost[p] {
                pbest_cost[p] = cost;
                pbest_pos[p] = positions[p];
                if cost < gbest_cost {
                    gbest_cost = cost;
                    gbest_pos = positions[p];
                }
            }
        }
        let improved =
            prev_gbest - gbest_cost > 1e-4 * prev_gbest.abs().max(1e-10);
        if improved {
            iters_without_improvement = 0;
            prev_gbest = gbest_cost;
        } else {
            iters_without_improvement += 1;
            // Partial restart: reinitialize worst particles to escape local optima
            if iters_without_improvement % restart_threshold == 0
                && iters_without_improvement < config.stall_iters
            {
                let n_restart = (config.n_particles as f64 * restart_fraction) as usize;
                // Sort particles by personal best cost (worst first)
                let mut indices: Vec<usize> = (0..config.n_particles).collect();
                indices.sort_by(|&a, &b| {
                    pbest_cost[b]
                        .partial_cmp(&pbest_cost[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (ri, &p) in indices.iter().take(n_restart).enumerate() {
                    for d in 0..N_PARAMS {
                        // Half near prior means, half uniform random
                        if ri < n_restart / 2 {
                            let u1: f64 = rng.random::<f64>().max(1e-10);
                            let u2: f64 = rng.random::<f64>();
                            let z = (-2.0 * u1.ln()).sqrt()
                                * (2.0 * std::f64::consts::PI * u2).cos();
                            positions[p][d] =
                                (means[d] + z * stds[d]).clamp(lower[d], upper[d]);
                        } else {
                            positions[p][d] =
                                lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                        }
                        velocities[p][d] =
                            (upper[d] - lower[d]) * 0.1 * (2.0 * rng.random::<f64>() - 1.0);
                    }
                    let cost = cost_fn(&positions[p]);
                    pbest_cost[p] = cost;
                    pbest_pos[p] = positions[p];
                    if cost < gbest_cost {
                        gbest_cost = cost;
                        gbest_pos = positions[p];
                        iters_without_improvement = 0;
                        prev_gbest = gbest_cost;
                    }
                }
            }
            if iters_without_improvement >= config.stall_iters {
                break;
            }
        }
    }

    (gbest_pos, gbest_cost)
}

// ─── Top-level fit function ──────────────────────────────────────────────────

pub struct FitResult {
    /// Best-fit parameters in raw (log10) space, length 14.
    pub raw_params: [f64; N_PARAMS],
    /// Best-fit parameters in physical space, length 14.
    pub phys_params: [f64; N_PARAMS],
    /// Reduced chi-squared of the best fit.
    pub reduced_chi2: f64,
    /// Number of real observations (before padding).
    pub orig_size: usize,
    /// Preprocessed observations (unpadded kept for plotting).
    pub obs: Vec<Obs>,
}

impl FitResult {
    /// Get named parameters as a HashMap.
    pub fn named_params(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        for (filt_idx, filt) in FILTERS.iter().enumerate() {
            for (p_idx, pname) in PARAM_NAMES.iter().enumerate() {
                let key = format!("{}_{}", pname, filt);
                map.insert(key, self.phys_params[filt_idx * N_BASE + p_idx]);
            }
        }
        map
    }

    /// Print results in same format as fit_jax_best.py.
    pub fn print_summary(&self) {
        println!(
            "\nBest-fit parameters (reduced chi2 = {:.3}):",
            self.reduced_chi2
        );
        for (filt_idx, filt) in FILTERS.iter().enumerate() {
            for (p_idx, pname) in PARAM_NAMES.iter().enumerate() {
                let idx = filt_idx * N_BASE + p_idx;
                println!(
                    "  {}_{:<14} = {:>12.6}  (raw: {:>10.6})",
                    pname, filt, self.phys_params[idx], self.raw_params[idx]
                );
            }
        }
    }
}

/// Run the full fitting pipeline on a CSV file.
pub fn fit_lightcurve(csv_path: &str) -> Result<FitResult, String> {
    let data = preprocess(csv_path)?;

    let param_map = build_param_map(&data.obs);
    let priors = PriorArrays::new();

    // PSO bounds = prior mins/maxs
    let mut lower = [0.0; N_PARAMS];
    let mut upper = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        lower[i] = priors.mins[i];
        upper[i] = priors.maxs[i];
    }

    let config = PsoConfig::default();
    let band_indices = BandIndices::new(&data.obs, data.orig_size);

    // Multi-seed PSO (3 seeds, skip later ones if first two agree)
    let seeds: [u64; 3] = [42, 137, 271];
    let mut best_params = [0.0; N_PARAMS];
    let mut best_cost = f64::INFINITY;
    let mut first_cost = f64::INFINITY;

    let cost_fn = |raw: &[f64; N_PARAMS]| {
        pso_cost(raw, &data.obs, &param_map, data.orig_size, &priors, 0.0, &band_indices)
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
    let abs_raw = offsets_to_absolute(&best_params);
    let phys = to_physical(&best_params, &priors);
    let rchi2 = reduced_chi2(&best_params, &data.obs, &param_map, data.orig_size, &priors);

    // Keep only the real (unpadded) observations for plotting
    // Padded obs have phase=1000.0 and flux_err=1000.0
    let real_obs: Vec<Obs> = data.obs.into_iter()
        .filter(|o| o.phase < 999.0 && o.flux_err < 999.0)
        .collect();

    Ok(FitResult {
        raw_params: abs_raw,  // absolute raw values (not offsets)
        phys_params: phys,
        reduced_chi2: rchi2,
        orig_size: data.orig_size,
        obs: real_obs,
    })
}

// ─── Plotting (via Python/matplotlib) ────────────────────────────────────────

/// Write observation data + best-fit params to a JSON file, then call a
/// bundled Python script to produce the diagnostic PNG.
pub fn make_plot(result: &FitResult, name: &str, output_dir: &str) -> Result<String, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Cannot create output dir: {}", e))?;

    let json_path = format!("{}/{}_fitdata.json", output_dir, name);
    let png_path = format!("{}/{}_best_fit.png", output_dir, name);

    // Build JSON payload
    let obs_json: Vec<String> = result.obs.iter().map(|o| {
        format!(
            r#"{{"phase":{}, "flux":{}, "flux_err":{}, "band":{}}}"#,
            o.phase, o.flux, o.flux_err, o.band
        )
    }).collect();

    let params_json: Vec<String> = result.phys_params.iter().map(|v| format!("{}", v)).collect();

    let json = format!(
        r#"{{"name":"{}","reduced_chi2":{},"params":[{}],"obs":[{}]}}"#,
        name,
        result.reduced_chi2,
        params_json.join(","),
        obs_json.join(","),
    );

    std::fs::write(&json_path, &json)
        .map_err(|e| format!("Cannot write JSON: {}", e))?;

    // Inline Python plotting script
    let py_script = format!(r#"
import json, sys
import numpy as np
import matplotlib.pyplot as plt

with open("{json_path}") as f:
    d = json.load(f)

name = d["name"]
rchi2 = d["reduced_chi2"]
params = d["params"]  # 14 values: [r0..r6, g0..g6] in physical space

def villar(t, p_offset):
    A, beta, gamma, t0, tau_rise, tau_fall, _ = params[p_offset:p_offset+7]
    gamma = max(gamma, 0.0)
    phase = np.clip(t - t0, a_min=-50.0 * tau_rise, a_max=None)
    f_const = A / (1.0 + np.exp(-phase / tau_rise))
    return np.where(
        gamma - phase >= 0,
        f_const * (1.0 - beta * phase),
        f_const * (1.0 - beta * gamma) * np.exp(-(phase - gamma) / tau_fall),
    )

obs = d["obs"]
colors = {{0: "tomato", 1: "steelblue"}}
labels = {{0: "ZTF_r", 1: "ZTF_g"}}
t_dense = np.linspace(-50, 100, 300)

fig, ax = plt.subplots(figsize=(8, 5))
for band_idx in [0, 1]:
    c = colors[band_idx]
    pts = [o for o in obs if o["band"] == band_idx]
    if not pts:
        continue
    ph = [o["phase"] for o in pts]
    fl = [o["flux"] for o in pts]
    fe = [o["flux_err"] for o in pts]
    ax.errorbar(ph, fl, yerr=fe, fmt="o", color=c, label=f"data {{labels[band_idx]}}", zorder=3)
    curve = villar(t_dense, band_idx * 7)
    ax.plot(t_dense, curve, color=c, lw=2, label=f"fit {{labels[band_idx]}}")

ax.set_xlabel("Phase (days)", fontsize=13)
ax.set_ylabel("Normalised flux", fontsize=13)
ax.set_title(f"{{name}} — best fit (χ² = {{rchi2:.3f}})", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("{png_path}", dpi=150)
plt.close(fig)
"#, json_path=json_path, png_path=png_path);

    // Try python first (anaconda), fall back to python3
    let status = std::process::Command::new("python")
        .arg("-c")
        .arg(&py_script)
        .status()
        .or_else(|_| {
            std::process::Command::new("python3")
                .arg("-c")
                .arg(&py_script)
                .status()
        })
        .map_err(|e| format!("Failed to run python/python3: {}", e))?;

    if !status.success() {
        return Err("Python plotting script failed".to_string());
    }

    // Clean up JSON
    let _ = std::fs::remove_file(&json_path);

    Ok(png_path)
}

// ─── PyO3 bindings ───────────────────────────────────────────────────────────

#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// Fit a ZTF light curve CSV and return a dict with results.
    #[pyfunction]
    fn fit(csv_path: &str) -> PyResult<PyObject> {
        let result = fit_lightcurve(csv_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            let params = PyDict::new(py);
            let raw_params = PyDict::new(py);
            for (filt_idx, filt) in FILTERS.iter().enumerate() {
                for (p_idx, pname) in PARAM_NAMES.iter().enumerate() {
                    let idx = filt_idx * N_BASE + p_idx;
                    let key = format!("{}_{}", pname, filt);
                    params.set_item(&key, result.phys_params[idx])?;
                    raw_params.set_item(&key, result.raw_params[idx])?;
                }
            }
            dict.set_item("params", params)?;
            dict.set_item("raw_params", raw_params)?;
            dict.set_item("reduced_chi2", result.reduced_chi2)?;
            dict.set_item("orig_size", result.orig_size)?;

            let obs_list: Vec<PyObject> = result.obs.iter().map(|o| {
                let d = PyDict::new(py);
                d.set_item("phase", o.phase).unwrap();
                d.set_item("flux", o.flux).unwrap();
                d.set_item("flux_err", o.flux_err).unwrap();
                d.set_item("band", o.band).unwrap();
                d.into_any().unbind()
            }).collect();
            dict.set_item("obs", obs_list)?;

            Ok(dict.into_any().unbind())
        })
    }

    /// Evaluate the Villar model on a dense time grid for one filter.
    #[pyfunction]
    fn eval_villar(params: &Bound<'_, PyDict>, t_dense: Vec<f64>, filt: &str) -> PyResult<Vec<f64>> {
        let filt_idx = match filt {
            "ZTF_r" => 0,
            "ZTF_g" => 1,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown filter: {}. Use 'ZTF_r' or 'ZTF_g'", filt)
            )),
        };

        let offset = filt_idx * N_BASE;
        let pm = [offset, offset + 1, offset + 2, offset + 3, offset + 4, offset + 5, offset + 6];

        let mut phys = [0.0; N_PARAMS];
        for (fi, f) in FILTERS.iter().enumerate() {
            for (pi, pn) in PARAM_NAMES.iter().enumerate() {
                let key = format!("{}_{}", pn, f);
                let val: f64 = params.get_item(&key)?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.clone()))?
                    .extract()?;
                phys[fi * N_BASE + pi] = val;
            }
        }

        let result: Vec<f64> = t_dense.iter().map(|&t| {
            let dummy = Obs { phase: t, flux: 0.0, flux_err: 0.0, band: filt_idx };
            let (f, _) = villar_flux_at(&phys, &dummy, &pm, 0.0);
            f
        }).collect();

        Ok(result)
    }

    /// Fit one or more ZTF light curve CSVs using the GPU batch PSO.
    /// Accepts a single path (str) or a list of paths.
    /// Returns a list of result dicts (same format as `fit()`).
    #[cfg(feature = "cuda")]
    #[pyfunction]
    fn fit_gpu(csv_paths: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyObject> {
        use crate::gpu::{GpuBatchData, GpuContext, SourceData};

        // Accept str or list of str
        let paths: Vec<String> = if let Ok(s) = csv_paths.extract::<String>() {
            vec![s]
        } else {
            csv_paths.extract::<Vec<String>>()?
        };

        // Preprocess
        let mut sources = Vec::new();
        let mut failed: Vec<(usize, String)> = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            match preprocess(p) {
                Ok(data) => {
                    let name = std::path::Path::new(p)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                        .to_string();
                    sources.push((i, SourceData { name, data }));
                }
                Err(e) => failed.push((i, e)),
            }
        }

        if sources.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No valid sources after preprocessing",
            ));
        }

        let config = PsoConfig::default();
        let gpu = GpuContext::new(0)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let source_data: Vec<&SourceData> = sources.iter().map(|(_, s)| s).collect();
        let batch_data = GpuBatchData::new(&source_data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let results = gpu
            .batch_pso_multi_seed(&batch_data, &source_data, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // Build output: one dict per input path (None for failed ones)
        Python::with_gil(|py| {
            let out_list = pyo3::types::PyList::empty(py);

            // Map gpu results back to original indices
            let mut result_iter = results.into_iter();
            let mut source_iter = sources.iter();
            let mut next_source = source_iter.next();
            let mut next_failed = failed.iter().peekable();

            for orig_idx in 0..paths.len() {
                // Check if this index failed
                if next_failed.peek().map_or(false, |(fi, _)| *fi == orig_idx) {
                    let (_, err) = next_failed.next().unwrap();
                    let d = PyDict::new(py);
                    d.set_item("error", err.as_str())?;
                    out_list.append(d)?;
                    continue;
                }

                if next_source.map_or(false, |(si, _)| *si == orig_idx) {
                    let res = result_iter.next().unwrap();
                    next_source = source_iter.next();

                    let dict = PyDict::new(py);
                    let params = PyDict::new(py);
                    let raw_params = PyDict::new(py);
                    for (filt_idx, filt) in FILTERS.iter().enumerate() {
                        for (p_idx, pname) in PARAM_NAMES.iter().enumerate() {
                            let idx = filt_idx * N_BASE + p_idx;
                            let key = format!("{}_{}", pname, filt);
                            params.set_item(&key, res.phys_params[idx])?;
                            raw_params.set_item(&key, res.raw_params[idx])?;
                        }
                    }
                    dict.set_item("params", params)?;
                    dict.set_item("raw_params", raw_params)?;
                    dict.set_item("reduced_chi2", res.reduced_chi2)?;
                    out_list.append(dict)?;
                }
            }

            // If single input, return single dict instead of list
            if paths.len() == 1 {
                Ok(out_list.get_item(0)?.unbind())
            } else {
                Ok(out_list.into_any().unbind())
            }
        })
    }

    #[pymodule]
    fn villar_pso(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fit, m)?)?;
        m.add_function(wrap_pyfunction!(eval_villar, m)?)?;
        #[cfg(feature = "cuda")]
        m.add_function(wrap_pyfunction!(fit_gpu, m)?)?;
        Ok(())
    }
}

