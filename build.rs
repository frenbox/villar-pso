#[cfg(feature = "cuda")]
fn build_cuda() {
    cc::Build::new()
        .cuda(true)
        .flag("-gencode=arch=compute_89,code=sm_89")   // Ada (RTX 4090)
        .flag("-gencode=arch=compute_120,code=sm_120")  // Blackwell (RTX 5080)
        .flag("-Wno-deprecated-gpu-targets")
        .file("cuda/villar_joint.cu")
        .compile("villar_pso_cuda");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=cuda/villar_joint.cu");
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}
