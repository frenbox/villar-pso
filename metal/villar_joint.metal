// Metal Shading Language port of villar_joint.cu — joint two-band Villar PSO
// cost evaluation for Apple Silicon GPUs (M1–M5).
//
// Apple GPUs do not support fp64, so every value is fp32. To recover most of
// the precision lost vs. the CUDA fp64 path, the per-band likelihood and the
// final accumulator use Kahan compensated summation. PSO is a stochastic
// optimiser and is robust to the residual fp32 error.
//
// Buffer layout (must match gpu_metal.rs):
//   0  all_times          float*   (per-observation, all sources concatenated)
//   1  all_flux           float*
//   2  all_flux_err_sq    float*
//   3  all_band           int*     (0=r, 1=g)
//   4  source_offsets     int*     (n_sources + 1)
//   5  n_r_per_source     int*
//   6  n_g_per_source     int*
//   7  positions          float*   (n_sources * n_particles * 14)
//   8  costs              float*   (n_sources * n_particles)
//   9  prior_means        float*   (14)
//   10 prior_stds         float*   (14)
//   11 logged_mask        int*     (14, 0 or 1)
//   12 CostParams         struct   { int n_sources; int n_particles; }

#include <metal_stdlib>
using namespace metal;

constant float LN10 = 2.302585092994046f;

// Villar model flux at a single time point, given physical-space parameters.
inline float villar_flux(float A, float beta, float gamma_p, float t0,
                         float tau_rise, float tau_fall, float t)
{
    float g = fmax(gamma_p, 0.0f);
    float phase = fmax(t - t0, -50.0f * tau_rise);
    float f_const = A / (1.0f + exp(-phase / tau_rise));
    if (g - phase >= 0.0f) {
        return f_const * (1.0f - beta * phase);
    } else {
        return f_const * (1.0f - beta * g) * exp(-(phase - g) / tau_fall);
    }
}

// Villar physical validity constraint. Returns 0 if valid, > 0 if violated.
inline float villar_constraint(float beta, float gamma_p,
                               float tau_rise, float tau_fall)
{
    float g = fmax(gamma_p, 0.0f);
    float c1 = fmax(g * beta - 1.0f, 0.0f);
    float c2 = fmax(exp(-g / tau_rise) * (tau_fall / tau_rise - 1.0f) - 1.0f, 0.0f);
    float c3 = fmax(beta * tau_fall - 1.0f + beta * g, 0.0f);
    return c1 + c2 + c3;
}

// Kahan compensated add: accumulates `x` into `sum` while tracking the running
// rounding error in `comp`. Recovers ~fp64-level precision for long sums of
// fp32 values at a cost of ~3 extra fp32 ops per add.
inline void kahan_add(thread float& sum, thread float& comp, float x)
{
    float y = x - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

struct CostParams {
    int n_sources;
    int n_particles;
};

kernel void batch_pso_cost_villar_joint(
    device const float* all_times       [[buffer(0)]],
    device const float* all_flux        [[buffer(1)]],
    device const float* all_flux_err_sq [[buffer(2)]],
    device const int*   all_band        [[buffer(3)]],
    device const int*   source_offsets  [[buffer(4)]],
    device const int*   n_r_per_source  [[buffer(5)]],
    device const int*   n_g_per_source  [[buffer(6)]],
    device const float* positions       [[buffer(7)]],
    device float*       costs           [[buffer(8)]],
    device const float* prior_means     [[buffer(9)]],
    device const float* prior_stds      [[buffer(10)]],
    device const int*   logged_mask     [[buffer(11)]],
    constant CostParams& params         [[buffer(12)]],
    uint idx                            [[thread_position_in_grid]])
{
    int total = params.n_sources * params.n_particles;
    if ((int)idx >= total) return;

    int src = (int)idx / params.n_particles;

    // Pull the 14 raw parameters for this particle.
    device const float* raw = positions + (uint)idx * 14u;

    // r-band stays absolute, g-band = r + offset.
    float abs_val[14];
    for (int d = 0; d < 7; d++) {
        abs_val[d]     = raw[d];
        abs_val[7 + d] = raw[d] + raw[7 + d];
    }

    // Apply 10^x to logged parameters.
    float phys[14];
    for (int d = 0; d < 14; d++) {
        if (logged_mask[d]) {
            phys[d] = exp(abs_val[d] * LN10);
        } else {
            phys[d] = abs_val[d];
        }
    }

    // Constraint check (per-band, take the max) — invalid particles get a huge
    // cost proportional to violation magnitude, giving PSO a gradient out.
    float c_r = villar_constraint(phys[1], phys[2], phys[4], phys[5]);
    float c_g = villar_constraint(phys[8], phys[9], phys[11], phys[12]);
    float constraint = fmax(c_r, c_g);
    if (constraint > 0.0f) {
        costs[idx] = 1e12f + 10000.0f * constraint;
        return;
    }

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];

    float ll_r = 0.0f, ll_r_c = 0.0f;
    float ll_g = 0.0f, ll_g_c = 0.0f;

    for (int i = obs_start; i < obs_end; i++) {
        int band = all_band[i];
        int off  = band * 7;

        float A         = phys[off + 0];
        float beta      = phys[off + 1];
        float gamma_p   = phys[off + 2];
        float t0        = phys[off + 3];
        float tau_rise  = phys[off + 4];
        float tau_fall  = phys[off + 5];
        float extra_sig = phys[off + 6];

        float model = villar_flux(A, beta, gamma_p, t0, tau_rise, tau_fall, all_times[i]);
        if (!isfinite(model)) {
            costs[idx] = 1e12f;
            return;
        }

        float sigma2  = all_flux_err_sq[i] + extra_sig * extra_sig;
        float diff    = all_flux[i] - model;
        float contrib = diff * diff / sigma2;

        if (band == 0) kahan_add(ll_r, ll_r_c, contrib);
        else           kahan_add(ll_g, ll_g_c, contrib);
    }

    // Band-balanced negative log-likelihood (each band weighted equally).
    int nr = n_r_per_source[src];
    int ng = n_g_per_source[src];
    float n_total = (float)(nr + ng);

    float neg_ll = 0.0f, neg_ll_c = 0.0f;
    if (nr > 0) kahan_add(neg_ll, neg_ll_c, (ll_r / (float)nr) * (n_total / 2.0f));
    if (ng > 0) kahan_add(neg_ll, neg_ll_c, (ll_g / (float)ng) * (n_total / 2.0f));

    // Gaussian prior penalty in raw search space.
    for (int d = 0; d < 14; d++) {
        float z = (raw[d] - prior_means[d]) / prior_stds[d];
        kahan_add(neg_ll, neg_ll_c, 0.5f * z * z);
    }

    costs[idx] = neg_ll;
}
