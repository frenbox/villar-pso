#[cfg(feature = "metal")]
#[test]
fn metal_shader_contains_expected_entrypoints() {
    let src = include_str!("../metal/villar_joint.metal");

    assert!(
        src.contains("kernel void batch_pso_cost_villar_joint_metal"),
        "missing Metal kernel entrypoint"
    );
    assert!(
        src.contains("inline double villar_flux"),
        "missing villar_flux helper"
    );
    assert!(
        src.contains("inline double villar_constraint"),
        "missing villar_constraint helper"
    );
    assert!(
        src.contains("prior_means") && src.contains("prior_stds") && src.contains("logged_mask"),
        "shader appears to be missing prior-penalty inputs"
    );
}
