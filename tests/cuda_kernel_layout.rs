#[cfg(feature = "cuda")]
#[test]
fn cuda_kernel_contains_expected_entrypoints() {
    let src = include_str!("../cuda/villar_joint.cu");

    assert!(
        src.contains("batch_pso_cost_villar_joint"),
        "missing CUDA kernel entrypoint"
    );
    assert!(
        src.contains("__device__ inline double villar_flux"),
        "missing villar_flux helper"
    );
    assert!(
        src.contains("__device__ inline double villar_constraint"),
        "missing villar_constraint helper"
    );
    assert!(
        src.contains("prior_means") && src.contains("prior_stds") && src.contains("logged_mask"),
        "kernel appears to be missing prior-penalty inputs"
    );
}
