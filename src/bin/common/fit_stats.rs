pub struct FitStats {
    pub n_ok: usize,
    pub n_fail: usize,
    pub per_source_ms: f64,
    pub chi2_mean: f64,
    pub chi2_med: f64,
    pub chi2_std: f64,
}

pub fn compute_fit_stats(mut chi2_vals: Vec<f64>, n_sources: usize, total_ms: f64) -> FitStats {
    let n_ok = chi2_vals.len();
    let n_fail = n_sources.saturating_sub(n_ok);
    let per_source_ms = total_ms / n_ok.max(1) as f64;

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

    FitStats {
        n_ok,
        n_fail,
        per_source_ms,
        chi2_mean,
        chi2_med,
        chi2_std,
    }
}
