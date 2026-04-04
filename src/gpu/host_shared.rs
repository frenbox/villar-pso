use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::{
    build_param_map, reduced_chi2, to_physical, BandIndices, FitResult, Obs, PriorArrays,
    PsoConfig, VillarParams, N_PARAMS,
};

use super::SourceData;

pub struct PackedHostBatch {
    pub all_times: Vec<f64>,
    pub all_flux: Vec<f64>,
    pub all_flux_err_sq: Vec<f64>,
    pub all_band: Vec<i32>,
    pub source_offsets: Vec<i32>,
    pub n_r_per_source: Vec<i32>,
    pub n_g_per_source: Vec<i32>,
}

pub fn pack_host_batch<S: AsRef<SourceData>>(sources: &[S]) -> PackedHostBatch {
    let n_sources = sources.len();
    let mut all_times = Vec::new();
    let mut all_flux = Vec::new();
    let mut all_flux_err_sq = Vec::new();
    let mut all_band: Vec<i32> = Vec::new();
    let mut source_offsets: Vec<i32> = Vec::with_capacity(n_sources + 1);
    let mut n_r_per_source: Vec<i32> = Vec::with_capacity(n_sources);
    let mut n_g_per_source: Vec<i32> = Vec::with_capacity(n_sources);

    source_offsets.push(0);
    for src in sources {
        let d = &src.as_ref().data;
        let bi = BandIndices::new(&d.obs, d.orig_size);
        for ob in d.obs.iter().take(d.orig_size) {
            all_times.push(ob.phase);
            all_flux.push(ob.flux);
            all_flux_err_sq.push(ob.flux_err * ob.flux_err);
            all_band.push(ob.band.idx() as i32);
        }
        source_offsets.push(all_times.len() as i32);
        n_r_per_source.push(bi.r_indices.len() as i32);
        n_g_per_source.push(bi.g_indices.len() as i32);
    }

    PackedHostBatch {
        all_times,
        all_flux,
        all_flux_err_sq,
        all_band,
        source_offsets,
        n_r_per_source,
        n_g_per_source,
    }
}

fn init_particles(
    n_sources: usize,
    n_particles: usize,
    dim: usize,
    seed: u64,
    priors: &PriorArrays,
    lower: &[f64; N_PARAMS],
    upper: &[f64; N_PARAMS],
    v_max: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let total_particles = n_sources * n_particles;
    let mut h_positions = vec![0.0f64; total_particles * dim];
    let mut h_velocities = vec![0.0f64; total_particles * dim];

    let n_seeded = n_particles * 3 / 10;
    for s in 0..n_sources {
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(s as u64));
        for p in 0..n_particles {
            let idx = s * n_particles + p;
            let base = idx * dim;
            if p < n_seeded {
                for d in 0..dim {
                    let u1: f64 = rng.random::<f64>().max(1e-10);
                    let u2: f64 = rng.random::<f64>();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    h_positions[base + d] =
                        (priors.means[d] + z * priors.stds[d]).clamp(lower[d], upper[d]);
                }
            } else {
                for d in 0..dim {
                    h_positions[base + d] = lower[d] + rng.random::<f64>() * (upper[d] - lower[d]);
                }
            }
            for d in 0..dim {
                h_velocities[base + d] = v_max[d] * 0.2 * (2.0 * rng.random::<f64>() - 1.0);
            }
        }
    }

    (h_positions, h_velocities)
}

pub struct HostPsoContext<'a> {
    pub n_sources: usize,
    pub n_particles: usize,
    pub dim: usize,
    pub seed: u64,
    pub config: &'a PsoConfig,
    pub priors: &'a PriorArrays,
    pub lower: &'a [f64; N_PARAMS],
    pub upper: &'a [f64; N_PARAMS],
    pub v_max: &'a [f64],
}

pub struct HostPsoOutcome {
    pub gbest_pos: Vec<f64>,
}

pub fn run_host_pso_loop<F>(
    ctx: HostPsoContext<'_>,
    mut evaluate_costs: F,
) -> Result<HostPsoOutcome, String>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), String>,
{
    let n_sources = ctx.n_sources;
    let n_particles = ctx.n_particles;
    let dim = ctx.dim;
    let total_particles = n_sources * n_particles;

    let (mut h_positions, mut h_velocities) = init_particles(
        n_sources,
        n_particles,
        dim,
        ctx.seed,
        ctx.priors,
        ctx.lower,
        ctx.upper,
        ctx.v_max,
    );

    let mut h_pbest_pos = h_positions.clone();
    let mut h_pbest_cost = vec![f64::INFINITY; total_particles];
    let mut h_gbest_pos = vec![0.0f64; n_sources * dim];
    let mut h_gbest_cost = vec![f64::INFINITY; n_sources];
    let mut h_costs = vec![0.0f64; total_particles];

    let mut prev_gbest = vec![f64::INFINITY; n_sources];
    let mut stall_count = vec![0usize; n_sources];
    let mut source_done = vec![false; n_sources];

    let restart_threshold = 40usize;
    let restart_fraction = 0.3;
    let w_start = ctx.config.w;
    let w_end = 0.4;
    let c1 = ctx.config.c1;
    let c2 = ctx.config.c2;
    let inv_max_iters = 1.0 / ctx.config.max_iters as f64;

    let mut rng = SmallRng::seed_from_u64(ctx.seed.wrapping_add(n_sources as u64 + 1));

    for iter in 0..ctx.config.max_iters {
        let w = w_start - (w_start - w_end) * (iter as f64) * inv_max_iters;

        evaluate_costs(&h_positions, &mut h_costs)?;

        let mut all_done = true;
        for s in 0..n_sources {
            if source_done[s] {
                continue;
            }

            for p in 0..n_particles {
                let idx = s * n_particles + p;
                let cost = h_costs[idx];
                if cost < h_pbest_cost[idx] {
                    h_pbest_cost[idx] = cost;
                    let base = idx * dim;
                    h_pbest_pos[base..base + dim].copy_from_slice(&h_positions[base..base + dim]);
                    if cost < h_gbest_cost[s] {
                        h_gbest_cost[s] = cost;
                        let gb = s * dim;
                        h_gbest_pos[gb..gb + dim].copy_from_slice(&h_positions[base..base + dim]);
                    }
                }
            }

            let improved = prev_gbest[s] - h_gbest_cost[s] > 1e-4 * prev_gbest[s].abs().max(1e-10);
            if improved {
                stall_count[s] = 0;
                prev_gbest[s] = h_gbest_cost[s];
            } else {
                stall_count[s] += 1;
                if stall_count[s] % restart_threshold == 0
                    && stall_count[s] < ctx.config.stall_iters
                {
                    let n_restart = (n_particles as f64 * restart_fraction) as usize;
                    let mut indices: Vec<usize> = (0..n_particles).collect();
                    indices.sort_by(|&a, &b| {
                        let ia = s * n_particles + a;
                        let ib = s * n_particles + b;
                        h_pbest_cost[ib]
                            .partial_cmp(&h_pbest_cost[ia])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for (ri, &p) in indices.iter().take(n_restart).enumerate() {
                        let idx = s * n_particles + p;
                        let base = idx * dim;
                        for d in 0..dim {
                            if ri < n_restart / 2 {
                                let u1: f64 = rng.random::<f64>().max(1e-10);
                                let u2: f64 = rng.random::<f64>();
                                let z = (-2.0 * u1.ln()).sqrt()
                                    * (2.0 * std::f64::consts::PI * u2).cos();
                                h_positions[base + d] = (ctx.priors.means[d]
                                    + z * ctx.priors.stds[d])
                                    .clamp(ctx.lower[d], ctx.upper[d]);
                            } else {
                                h_positions[base + d] = ctx.lower[d]
                                    + rng.random::<f64>() * (ctx.upper[d] - ctx.lower[d]);
                            }
                            h_velocities[base + d] = (ctx.upper[d] - ctx.lower[d])
                                * 0.1
                                * (2.0 * rng.random::<f64>() - 1.0);
                        }
                        h_pbest_cost[idx] = f64::INFINITY;
                    }
                }

                if stall_count[s] >= ctx.config.stall_iters {
                    source_done[s] = true;
                }
            }

            if !source_done[s] {
                all_done = false;
                for p in 0..n_particles {
                    let idx = s * n_particles + p;
                    let base = idx * dim;
                    let gb = s * dim;
                    for d in 0..dim {
                        let r1: f64 = rng.random();
                        let r2: f64 = rng.random();
                        let mut v = w * h_velocities[base + d]
                            + c1 * r1 * (h_pbest_pos[base + d] - h_positions[base + d])
                            + c2 * r2 * (h_gbest_pos[gb + d] - h_positions[base + d]);
                        v = v.clamp(-ctx.v_max[d], ctx.v_max[d]);
                        let new_pos = h_positions[base + d] + v;
                        if new_pos <= ctx.lower[d] {
                            h_positions[base + d] = ctx.lower[d];
                            h_velocities[base + d] = 0.0;
                        } else if new_pos >= ctx.upper[d] {
                            h_positions[base + d] = ctx.upper[d];
                            h_velocities[base + d] = 0.0;
                        } else {
                            h_positions[base + d] = new_pos;
                            h_velocities[base + d] = v;
                        }
                    }
                }
            }
        }

        if all_done {
            break;
        }
    }

    Ok(HostPsoOutcome {
        gbest_pos: h_gbest_pos,
    })
}

pub fn collect_fit_results<S: AsRef<SourceData>>(
    sources: &[S],
    h_gbest_pos: &[f64],
    dim: usize,
    priors: &PriorArrays,
) -> Vec<FitResult> {
    (0..sources.len())
        .map(|s| {
            let gb = s * dim;
            let mut raw = [0.0f64; N_PARAMS];
            raw.copy_from_slice(&h_gbest_pos[gb..gb + dim]);

            let phys = to_physical(&raw, priors);
            let d = &sources[s].as_ref().data;
            let param_map = build_param_map(&d.obs);
            let rchi2 = reduced_chi2(&raw, &d.obs, &param_map, d.orig_size, priors);

            let params = VillarParams::from_phys(&phys);
            let real_obs: Vec<Obs> = d
                .obs
                .iter()
                .filter(|o| o.phase < 999.0 && o.flux_err < 999.0)
                .cloned()
                .collect();

            FitResult {
                params,
                params_unnorm: params.unnormalized(d.peak_flux),
                peak_flux: d.peak_flux,
                reduced_chi2: rchi2,
                orig_size: d.orig_size,
                obs: real_obs,
            }
        })
        .collect()
}

pub fn merge_best_results(
    best: Option<Vec<FitResult>>,
    results: Vec<FitResult>,
) -> Option<Vec<FitResult>> {
    Some(match best {
        None => results,
        Some(prev) => prev
            .into_iter()
            .zip(results)
            .map(|(a, b)| {
                if b.reduced_chi2 < a.reduced_chi2 {
                    b
                } else {
                    a
                }
            })
            .collect(),
    })
}
