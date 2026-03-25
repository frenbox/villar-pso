// CUDA kernel for joint two-band Villar PSO cost evaluation.
//
// One thread per (source, particle). Each thread loops over that source's
// observations, evaluates the Villar model with band-balanced likelihood,
// adds constraint penalties and prior penalty.

#include <math.h>

#define LN10 2.302585092994046

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Villar model flux at a single time point, given physical-space parameters.
__device__ inline double villar_flux(
    double A, double beta, double gamma, double t0,
    double tau_rise, double tau_fall, double t)
{
    gamma = fmax(gamma, 0.0);
    double phase = fmax(t - t0, -50.0 * tau_rise);
    double f_const = A / (1.0 + exp(-phase / tau_rise));
    if (gamma - phase >= 0.0) {
        return f_const * (1.0 - beta * phase);
    } else {
        return f_const * (1.0 - beta * gamma) * exp(-(phase - gamma) / tau_fall);
    }
}

// Villar constraint for one band (physical params).
// Returns 0 if valid, >0 if violated.
__device__ inline double villar_constraint(
    double beta, double gamma, double tau_rise, double tau_fall)
{
    gamma = fmax(gamma, 0.0);
    double c1 = fmax(gamma * beta - 1.0, 0.0);
    double c2 = fmax(exp(-gamma / tau_rise) * (tau_fall / tau_rise - 1.0) - 1.0, 0.0);
    double c3 = fmax(beta * tau_fall - 1.0 + beta * gamma, 0.0);
    return c1 + c2 + c3;
}

// ---------------------------------------------------------------------------
// Batch PSO cost kernel
// ---------------------------------------------------------------------------
//
// Thread grid: n_sources * n_particles
// Each thread computes one cost value.
//
// Parameters layout per particle (14 doubles, row-major):
//   [0..7)  = r-band raw values
//   [7..14) = g-band OFFSETS from r-band
//
// logged_mask[d] == 1 means param d lives in log10 space (apply 10^x).
//
// Observation data is concatenated across sources; source_offsets[src] gives
// the start index, source_offsets[src+1] the end.

extern "C" __global__ void batch_pso_cost_villar_joint(
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_flux_err_sq,
    const int*    __restrict__ all_band,
    const int*    __restrict__ source_offsets,
    const int*    __restrict__ n_r_per_source,
    const int*    __restrict__ n_g_per_source,
    const double* __restrict__ positions,
    double*       __restrict__ costs,
    const double* __restrict__ prior_means,
    const double* __restrict__ prior_stds,
    const int*    __restrict__ logged_mask,
    int n_sources,
    int n_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;

    // Read 14 raw parameters
    const double* raw = positions + (long long)idx * 14;

    // Convert to absolute values: r-band stays, g-band = r + offset
    double abs_val[14];
    for (int d = 0; d < 7; d++) {
        abs_val[d] = raw[d];           // r-band
        abs_val[7 + d] = raw[d] + raw[7 + d]; // g-band absolute
    }

    // Apply 10^x transform to logged parameters
    double phys[14];
    for (int d = 0; d < 14; d++) {
        if (logged_mask[d]) {
            phys[d] = exp(abs_val[d] * LN10);
        } else {
            phys[d] = abs_val[d];
        }
    }

    // Check constraints for both bands
    // r-band: phys[0..7] = A, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma
    double c_r = villar_constraint(phys[1], phys[2], phys[4], phys[5]);
    double c_g = villar_constraint(phys[8], phys[9], phys[11], phys[12]);
    double constraint = fmax(c_r, c_g);
    if (constraint > 0.0) {
        costs[idx] = 1e12 + 10000.0 * constraint;
        return;
    }

    // Evaluate band-balanced likelihood
    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];

    double ll_r = 0.0;
    double ll_g = 0.0;

    for (int i = obs_start; i < obs_end; i++) {
        int band = all_band[i];
        int off = band * 7;  // 0 for r-band, 7 for g-band

        double A         = phys[off + 0];
        double beta      = phys[off + 1];
        double gamma     = phys[off + 2];
        double t0        = phys[off + 3];
        double tau_rise  = phys[off + 4];
        double tau_fall  = phys[off + 5];
        double extra_sig = phys[off + 6];

        double model = villar_flux(A, beta, gamma, t0, tau_rise, tau_fall, all_times[i]);
        if (!isfinite(model)) {
            costs[idx] = 1e12;
            return;
        }

        double sigma2 = all_flux_err_sq[i] + extra_sig * extra_sig;
        double diff = all_flux[i] - model;
        double contrib = diff * diff / sigma2;

        if (band == 0) {
            ll_r += contrib;
        } else {
            ll_g += contrib;
        }
    }

    // Band-balanced: each band contributes equally
    int nr = n_r_per_source[src];
    int ng = n_g_per_source[src];
    double n_total = (double)(nr + ng);

    double neg_ll = 0.0;
    if (nr > 0) neg_ll += (ll_r / (double)nr) * (n_total / 2.0);
    if (ng > 0) neg_ll += (ll_g / (double)ng) * (n_total / 2.0);

    // Prior penalty
    for (int d = 0; d < 14; d++) {
        double z = (raw[d] - prior_means[d]) / prior_stds[d];
        neg_ll += 0.5 * z * z;
    }

    costs[idx] = neg_ll;
}

// ---------------------------------------------------------------------------
// PSO velocity + position update kernel
// ---------------------------------------------------------------------------
//
// One thread per (source, particle, dim_chunk). We flatten to one thread per
// (source * n_particles) and loop over dims internally (14 is small).
// Also updates personal bests and prepares per-source global-best reduction.

// Update personal bests from latest costs. Must run BEFORE velocity update
// and BEFORE gbest reduce, so that gbest_reduce sees updated pbest values.
extern "C" __global__ void pbest_update_kernel(
    const double* __restrict__ positions,
    double* __restrict__ pbest_pos,
    double* __restrict__ pbest_cost,
    const double* __restrict__ costs,
    int n_sources, int n_particles, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    double cost = costs[idx];
    if (cost < pbest_cost[idx]) {
        pbest_cost[idx] = cost;
        int base = idx * dim;
        for (int d = 0; d < dim; d++) {
            pbest_pos[base + d] = positions[base + d];
        }
    }
}

// Velocity + position update. Must run AFTER gbest_reduce so that gbest_pos
// is consistent across all particles for a given source.
extern "C" __global__ void pso_move_kernel(
    double* __restrict__ positions,
    double* __restrict__ velocities,
    const double* __restrict__ pbest_pos,
    const double* __restrict__ gbest_pos,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ v_max,
    unsigned long long* __restrict__ rng_states,
    double w, double c1, double c2,
    int n_sources, int n_particles, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;

    // Use iteration counter from rng_states as a simple counter-based RNG.
    // Each call to pso_move increments the counter.
    unsigned long long counter = rng_states[idx];

    int base = idx * dim;
    int gb = src * dim;
    for (int d = 0; d < dim; d++) {
        // MurmurHash3-style mixing for each (counter, d) pair
        counter++;
        unsigned long long h1 = counter;
        h1 ^= h1 >> 33;
        h1 *= 0xff51afd7ed558ccdULL;
        h1 ^= h1 >> 33;
        h1 *= 0xc4ceb9fe1a85ec53ULL;
        h1 ^= h1 >> 33;
        double r1 = (double)(h1 >> 11) * (1.0 / 4503599627370496.0);

        counter++;
        unsigned long long h2 = counter;
        h2 ^= h2 >> 33;
        h2 *= 0xff51afd7ed558ccdULL;
        h2 ^= h2 >> 33;
        h2 *= 0xc4ceb9fe1a85ec53ULL;
        h2 ^= h2 >> 33;
        double r2 = (double)(h2 >> 11) * (1.0 / 4503599627370496.0);
        double v = w * velocities[base + d]
            + c1 * r1 * (pbest_pos[base + d] - positions[base + d])
            + c2 * r2 * (gbest_pos[gb + d] - positions[base + d]);

        // Clamp velocity
        if (v > v_max[d]) v = v_max[d];
        if (v < -v_max[d]) v = -v_max[d];

        double new_pos = positions[base + d] + v;
        if (new_pos <= lower[d]) {
            positions[base + d] = lower[d];
            velocities[base + d] = 0.0;
        } else if (new_pos >= upper[d]) {
            positions[base + d] = upper[d];
            velocities[base + d] = 0.0;
        } else {
            positions[base + d] = new_pos;
            velocities[base + d] = v;
        }
    }
    rng_states[idx] = counter;
}

// ---------------------------------------------------------------------------
// Per-source global-best reduction kernel
// ---------------------------------------------------------------------------
// One thread per source. Scans all particles for that source to find the best.

extern "C" __global__ void gbest_reduce_kernel(
    const double* __restrict__ pbest_cost,
    const double* __restrict__ pbest_pos,
    double* __restrict__ gbest_cost,
    double* __restrict__ gbest_pos,
    int* __restrict__ stall_count,
    int* __restrict__ source_done,
    int n_sources, int n_particles, int dim, int stall_limit)
{
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_sources) return;
    if (source_done[src]) return;

    double prev = gbest_cost[src];
    int gb = src * dim;

    for (int p = 0; p < n_particles; p++) {
        int idx = src * n_particles + p;
        if (pbest_cost[idx] < gbest_cost[src]) {
            gbest_cost[src] = pbest_cost[idx];
            int base = idx * dim;
            for (int d = 0; d < dim; d++) {
                gbest_pos[gb + d] = pbest_pos[base + d];
            }
        }
    }

    // Stall detection
    double improved = prev - gbest_cost[src];
    if (improved > 1e-4 * fmax(fabs(prev), 1e-10)) {
        stall_count[src] = 0;
    } else {
        stall_count[src]++;
        if (stall_count[src] >= stall_limit) {
            source_done[src] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Partial restart kernel: reinitialize particles far from gbest
// ---------------------------------------------------------------------------
// Particles whose pbest_cost > restart_ratio * gbest_cost get reinitialized.
// At most restart_frac fraction of particles per source are restarted.
// Half are seeded near prior means, half uniform random.

extern "C" __global__ void pso_restart_kernel(
    double* __restrict__ positions,
    double* __restrict__ velocities,
    double* __restrict__ pbest_pos,
    double* __restrict__ pbest_cost,
    const double* __restrict__ gbest_cost,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ v_max,
    const double* __restrict__ prior_means,
    const double* __restrict__ prior_stds,
    unsigned long long* __restrict__ rng_states,
    int n_sources, int n_particles, int dim,
    double restart_ratio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;
    int pid = idx % n_particles;

    // Never restart the best particle (particle 0 is arbitrary, but skip low pid)
    if (pid == 0) return;

    double threshold = restart_ratio * fmax(fabs(gbest_cost[src]), 1e-6);
    if (pbest_cost[idx] <= threshold) return;

    // Reinitialize this particle
    unsigned long long state = rng_states[idx];

    int base = idx * dim;
    int do_seeded = (pid % 2 == 0); // half seeded, half uniform

    for (int d = 0; d < dim; d++) {
        double new_val;
        if (do_seeded) {
            // Box-Muller near prior mean
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            double u1 = (double)((state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 4503599627370496.0);
            u1 = fmax(u1, 1e-10);
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            double u2 = (double)((state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 4503599627370496.0);
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            new_val = prior_means[d] + z * prior_stds[d];
        } else {
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            double u = (double)((state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 4503599627370496.0);
            new_val = lower[d] + u * (upper[d] - lower[d]);
        }
        // Clamp
        if (new_val < lower[d]) new_val = lower[d];
        if (new_val > upper[d]) new_val = upper[d];
        positions[base + d] = new_val;
        pbest_pos[base + d] = new_val;

        // Reset velocity
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        double u = (double)((state * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 4503599627370496.0);
        velocities[base + d] = v_max[d] * 0.2 * (2.0 * u - 1.0);
    }
    pbest_cost[idx] = 1e30; // will be updated on next cost eval
    rng_states[idx] = state;
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers (callable from Rust via FFI)
// ---------------------------------------------------------------------------

extern "C" void launch_pbest_update(
    const double* positions, double* pbest_pos, double* pbest_cost,
    const double* costs,
    int n_sources, int n_particles, int dim,
    int grid, int block)
{
    pbest_update_kernel<<<grid, block>>>(
        positions, pbest_pos, pbest_cost, costs,
        n_sources, n_particles, dim);
}

extern "C" void launch_pso_move(
    double* positions, double* velocities,
    const double* pbest_pos, const double* gbest_pos,
    const double* lower, const double* upper, const double* v_max,
    unsigned long long* rng_states,
    double w, double c1, double c2,
    int n_sources, int n_particles, int dim,
    int grid, int block)
{
    pso_move_kernel<<<grid, block>>>(
        positions, velocities, pbest_pos, gbest_pos,
        lower, upper, v_max, rng_states,
        w, c1, c2, n_sources, n_particles, dim);
}

extern "C" void launch_pso_restart(
    double* positions, double* velocities,
    double* pbest_pos, double* pbest_cost,
    const double* gbest_cost,
    const double* lower, const double* upper, const double* v_max,
    const double* prior_means, const double* prior_stds,
    unsigned long long* rng_states,
    int n_sources, int n_particles, int dim,
    double restart_ratio,
    int grid, int block)
{
    pso_restart_kernel<<<grid, block>>>(
        positions, velocities, pbest_pos, pbest_cost, gbest_cost,
        lower, upper, v_max, prior_means, prior_stds, rng_states,
        n_sources, n_particles, dim, restart_ratio);
}

extern "C" void launch_gbest_reduce(
    const double* pbest_cost, const double* pbest_pos,
    double* gbest_cost, double* gbest_pos,
    int* stall_count, int* source_done,
    int n_sources, int n_particles, int dim, int stall_limit,
    int grid, int block)
{
    gbest_reduce_kernel<<<grid, block>>>(
        pbest_cost, pbest_pos, gbest_cost, gbest_pos,
        stall_count, source_done,
        n_sources, n_particles, dim, stall_limit);
}

extern "C" void launch_batch_pso_cost_villar_joint(
    const double* all_times,
    const double* all_flux,
    const double* all_flux_err_sq,
    const int*    all_band,
    const int*    source_offsets,
    const int*    n_r_per_source,
    const int*    n_g_per_source,
    const double* positions,
    double*       costs,
    const double* prior_means,
    const double* prior_stds,
    const int*    logged_mask,
    int n_sources,
    int n_particles,
    int grid,
    int block)
{
    batch_pso_cost_villar_joint<<<grid, block>>>(
        all_times, all_flux, all_flux_err_sq, all_band,
        source_offsets, n_r_per_source, n_g_per_source,
        positions, costs,
        prior_means, prior_stds, logged_mask,
        n_sources, n_particles);
}
