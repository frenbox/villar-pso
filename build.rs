#[cfg(feature = "cuda")]
fn build_cuda() {
    let ccbin = std::env::var("NVCC_CCBIN").ok().or_else(|| {
        let candidate = "/usr/bin/g++-12";
        if std::path::Path::new(candidate).exists() {
            Some(candidate.to_string())
        } else {
            None
        }
    });

    let arches = select_cuda_arches().unwrap_or_else(|reason| {
        println!("cargo:warning=CUDA arch auto-detection failed: {}", reason);
        vec!["compute_80".to_string()]
    });

    println!("cargo:warning=CUDA arch targets: {}", arches.join(", "));

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-Wno-deprecated-gpu-targets")
        .flag("-allow-unsupported-compiler")
        .file("cuda/villar_joint.cu");

    if let Some(ccbin) = ccbin {
        std::env::set_var("CXX", &ccbin);
        std::env::set_var("NVCC_CCBIN", &ccbin);
    }

    for arch in &arches {
        let sm = arch.replacen("compute_", "sm_", 1);
        build.flag(&format!("-gencode=arch={},code={}", arch, sm));
        build.flag(&format!("-gencode=arch={},code={}", arch, arch));
    }

    build.compile("villar_pso_cuda");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=cuda/villar_joint.cu");
}

#[cfg(feature = "cuda")]
fn select_cuda_arches() -> Result<Vec<String>, String> {
    if let Ok(spec) = std::env::var("NVCC_CUDA_ARCHS").or_else(|_| std::env::var("CUDA_ARCHS")) {
        let mut arches = Vec::new();
        for token in spec.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            arches.push(normalize_arch_token(token)?);
        }
        if arches.is_empty() {
            return Err("NVCC_CUDA_ARCHS/CUDA_ARCHS was set but empty".to_string());
        }
        arches.sort();
        arches.dedup();
        return Ok(arches);
    }

    let nvcc_supported = query_nvcc_supported_arches()?;
    let gpu_caps = query_visible_gpu_arches().unwrap_or_default();

    if nvcc_supported.is_empty() {
        return Err("nvcc did not report any supported architectures".to_string());
    }

    let chosen = if gpu_caps.is_empty() {
        vec![format_compute_arch(*nvcc_supported.last().unwrap())]
    } else {
        let mut selected = Vec::new();
        for cap in gpu_caps {
            selected.push(best_supported_arch(&nvcc_supported, cap));
        }
        selected.sort();
        selected.dedup();
        selected.into_iter().map(format_compute_arch).collect()
    };

    Ok(chosen)
}

#[cfg(feature = "cuda")]
fn query_visible_gpu_arches() -> Result<Vec<(u32, u32)>, String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("failed to run nvidia-smi: {}", e))?;

    if !output.status.success() {
        return Err(format!("nvidia-smi exited with status {}", output.status));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut caps = Vec::new();
    for line in stdout.lines() {
        if let Some(cap) = parse_compute_cap(line) {
            caps.push(cap);
        }
    }
    caps.sort();
    caps.dedup();
    Ok(caps)
}

#[cfg(feature = "cuda")]
fn query_nvcc_supported_arches() -> Result<Vec<(u32, u32)>, String> {
    let output = std::process::Command::new("nvcc")
        .arg("--list-gpu-arch")
        .output()
        .map_err(|e| format!("failed to run nvcc --list-gpu-arch: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvcc --list-gpu-arch exited with status {}",
            output.status
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut arches = Vec::new();
    for line in stdout.lines() {
        let token = line.trim();
        if token.starts_with("compute_") {
            if let Some(cap) = parse_compute_cap(token) {
                arches.push(cap);
            }
        }
    }
    arches.sort();
    arches.dedup();
    Ok(arches)
}

#[cfg(feature = "cuda")]
fn best_supported_arch(supported: &[(u32, u32)], cap: (u32, u32)) -> (u32, u32) {
    if supported.iter().any(|&arch| arch == cap) {
        return cap;
    }

    let fallback = supported.iter().copied().filter(|&arch| arch <= cap).last();

    fallback.unwrap_or_else(|| supported.last().copied().unwrap_or(cap))
}

#[cfg(feature = "cuda")]
fn normalize_arch_token(token: &str) -> Result<String, String> {
    if token.starts_with("compute_") {
        return parse_compute_cap(token)
            .map(format_compute_arch)
            .ok_or_else(|| format!("invalid CUDA arch token: {}", token));
    }

    parse_compute_cap(token)
        .map(format_compute_arch)
        .ok_or_else(|| format!("invalid CUDA arch token: {}", token))
}

#[cfg(feature = "cuda")]
fn parse_compute_cap(text: &str) -> Option<(u32, u32)> {
    let raw = text.trim().trim_start_matches("compute_").trim();
    if raw.is_empty() {
        return None;
    }

    if let Some((major, minor)) = raw.split_once('.') {
        let major = major.parse().ok()?;
        let minor = minor.parse().ok()?;
        return Some((major, minor));
    }

    let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 2 {
        return None;
    }

    let (major, minor) = digits.split_at(digits.len() - 1);
    let major = major.parse().ok()?;
    let minor = minor.parse().ok()?;
    Some((major, minor))
}

#[cfg(feature = "cuda")]
fn format_compute_arch(cap: (u32, u32)) -> String {
    format!("compute_{}{}", cap.0, cap.1)
}

fn main() {
    #[cfg(feature = "cuda")]
    build_cuda();
}
