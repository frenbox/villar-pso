#include <metal_stdlib>
using namespace metal;

constant float LN10 = 2.302585092994046f;

struct KernelParams {
    uint n_sources;
    uint n_particles;
};

inline float villar_flux(
    float A,
    float beta,
    float gamma,
    float t0,
    float tau_rise,
    float tau_fall,
    float t
) {
    gamma = max(gamma, 0.0f);
    float phase = max(t - t0, -50.0f * tau_rise);
    float f_const = A / (1.0f + exp(-phase / tau_rise));
    if (gamma - phase >= 0.0f) {
        return f_const * (1.0f - beta * phase);
    }
    return f_const * (1.0f - beta * gamma) * exp(-(phase - gamma) / tau_fall);
}

inline float villar_constraint(float beta, float gamma, float tau_rise, float tau_fall) {
    gamma = max(gamma, 0.0f);
    float c1 = max(gamma * beta - 1.0f, 0.0f);
    float c2 = max(exp(-gamma / tau_rise) * (tau_fall / tau_rise - 1.0f) - 1.0f, 0.0f);
    float c3 = max(beta * tau_fall - 1.0f + beta * gamma, 0.0f);
    return c1 + c2 + c3;
}

kernel void batch_pso_cost_villar_joint_metal(
    const device float* all_times [[buffer(0)]],
    const device float* all_flux [[buffer(1)]],
    const device float* all_flux_err_sq [[buffer(2)]],
    const device int* all_band [[buffer(3)]],
    const device int* source_offsets [[buffer(4)]],
    const device int* n_r_per_source [[buffer(5)]],
    const device int* n_g_per_source [[buffer(6)]],
    const device float* positions [[buffer(7)]],
    device float* costs [[buffer(8)]],
    const device float* prior_means [[buffer(9)]],
    const device float* prior_stds [[buffer(10)]],
    const device int* logged_mask [[buffer(11)]],
    constant KernelParams& params [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.n_sources * params.n_particles;
    if (gid >= total) {
        return;
    }

    uint src = gid / params.n_particles;
    const device float* raw = positions + gid * 14;

    float abs_val[14];
    for (uint d = 0; d < 7; d++) {
        abs_val[d] = raw[d];
        abs_val[7 + d] = raw[d] + raw[7 + d];
    }

    float phys[14];
    for (uint d = 0; d < 14; d++) {
        phys[d] = logged_mask[d] ? exp(abs_val[d] * LN10) : abs_val[d];
    }

    float c_r = villar_constraint(phys[1], phys[2], phys[4], phys[5]);
    float c_g = villar_constraint(phys[8], phys[9], phys[11], phys[12]);
    float constraint = max(c_r, c_g);
    if (constraint > 0.0f) {
        costs[gid] = 1e12f + 10000.0f * constraint;
        return;
    }

    int obs_start = source_offsets[src];
    int obs_end = source_offsets[src + 1];

    float ll_r = 0.0f;
    float ll_g = 0.0f;

    for (int i = obs_start; i < obs_end; i++) {
        int band = all_band[i];
        int off = band * 7;

        float A = phys[off + 0];
        float beta = phys[off + 1];
        float gamma = phys[off + 2];
        float t0 = phys[off + 3];
        float tau_rise = phys[off + 4];
        float tau_fall = phys[off + 5];
        float extra_sig = phys[off + 6];

        float model = villar_flux(A, beta, gamma, t0, tau_rise, tau_fall, all_times[i]);
        if (!isfinite(model)) {
            costs[gid] = 1e12f;
            return;
        }

        float sigma2 = all_flux_err_sq[i] + extra_sig * extra_sig;
        float diff = all_flux[i] - model;
        float contrib = diff * diff / sigma2;

        if (band == 0) {
            ll_r += contrib;
        } else {
            ll_g += contrib;
        }
    }

    int nr = n_r_per_source[src];
    int ng = n_g_per_source[src];
    float n_total = float(nr + ng);

    float neg_ll = 0.0f;
    if (nr > 0) {
        neg_ll += (ll_r / float(nr)) * (n_total / 2.0f);
    }
    if (ng > 0) {
        neg_ll += (ll_g / float(ng)) * (n_total / 2.0f);
    }

    for (uint d = 0; d < 14; d++) {
        float z = (raw[d] - prior_means[d]) / prior_stds[d];
        neg_ll += 0.5f * z * z;
    }

    costs[gid] = neg_ll;
}
