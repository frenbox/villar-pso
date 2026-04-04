#include <metal_stdlib>
using namespace metal;

constant double LN10 = 2.302585092994046;

struct KernelParams {
    uint n_sources;
    uint n_particles;
};

inline double villar_flux(
    double A,
    double beta,
    double gamma,
    double t0,
    double tau_rise,
    double tau_fall,
    double t
) {
    gamma = max(gamma, 0.0);
    double phase = max(t - t0, -50.0 * tau_rise);
    double f_const = A / (1.0 + exp(-phase / tau_rise));
    if (gamma - phase >= 0.0) {
        return f_const * (1.0 - beta * phase);
    }
    return f_const * (1.0 - beta * gamma) * exp(-(phase - gamma) / tau_fall);
}

inline double villar_constraint(double beta, double gamma, double tau_rise, double tau_fall) {
    gamma = max(gamma, 0.0);
    double c1 = max(gamma * beta - 1.0, 0.0);
    double c2 = max(exp(-gamma / tau_rise) * (tau_fall / tau_rise - 1.0) - 1.0, 0.0);
    double c3 = max(beta * tau_fall - 1.0 + beta * gamma, 0.0);
    return c1 + c2 + c3;
}

kernel void batch_pso_cost_villar_joint_metal(
    const device double* all_times [[buffer(0)]],
    const device double* all_flux [[buffer(1)]],
    const device double* all_flux_err_sq [[buffer(2)]],
    const device int* all_band [[buffer(3)]],
    const device int* source_offsets [[buffer(4)]],
    const device int* n_r_per_source [[buffer(5)]],
    const device int* n_g_per_source [[buffer(6)]],
    const device double* positions [[buffer(7)]],
    device double* costs [[buffer(8)]],
    const device double* prior_means [[buffer(9)]],
    const device double* prior_stds [[buffer(10)]],
    const device int* logged_mask [[buffer(11)]],
    constant KernelParams& params [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.n_sources * params.n_particles;
    if (gid >= total) {
        return;
    }

    uint src = gid / params.n_particles;
    const device double* raw = positions + gid * 14;

    double abs_val[14];
    for (uint d = 0; d < 7; d++) {
        abs_val[d] = raw[d];
        abs_val[7 + d] = raw[d] + raw[7 + d];
    }

    double phys[14];
    for (uint d = 0; d < 14; d++) {
        phys[d] = logged_mask[d] ? exp(abs_val[d] * LN10) : abs_val[d];
    }

    double c_r = villar_constraint(phys[1], phys[2], phys[4], phys[5]);
    double c_g = villar_constraint(phys[8], phys[9], phys[11], phys[12]);
    double constraint = max(c_r, c_g);
    if (constraint > 0.0) {
        costs[gid] = 1e12 + 10000.0 * constraint;
        return;
    }

    int obs_start = source_offsets[src];
    int obs_end = source_offsets[src + 1];

    double ll_r = 0.0;
    double ll_g = 0.0;

    for (int i = obs_start; i < obs_end; i++) {
        int band = all_band[i];
        int off = band * 7;

        double A = phys[off + 0];
        double beta = phys[off + 1];
        double gamma = phys[off + 2];
        double t0 = phys[off + 3];
        double tau_rise = phys[off + 4];
        double tau_fall = phys[off + 5];
        double extra_sig = phys[off + 6];

        double model = villar_flux(A, beta, gamma, t0, tau_rise, tau_fall, all_times[i]);
        if (!isfinite(model)) {
            costs[gid] = 1e12;
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

    int nr = n_r_per_source[src];
    int ng = n_g_per_source[src];
    double n_total = double(nr + ng);

    double neg_ll = 0.0;
    if (nr > 0) {
        neg_ll += (ll_r / double(nr)) * (n_total / 2.0);
    }
    if (ng > 0) {
        neg_ll += (ll_g / double(ng)) * (n_total / 2.0);
    }

    for (uint d = 0; d < 14; d++) {
        double z = (raw[d] - prior_means[d]) / prior_stds[d];
        neg_ll += 0.5 * z * z;
    }

    costs[gid] = neg_ll;
}
