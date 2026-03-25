//! Batch GPU PSO fitting for all CSVs in a directory.
//!
//! Usage: gpu-batch <data_dir> [--particles N] [--max-iters N] [--output results.csv]

use std::time::Instant;
use villar_pso::gpu::{GpuBatchData, GpuContext, load_sources};
use villar_pso::{PsoConfig, FILTERS, N_BASE, PARAM_NAMES};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("../data/photometry");

    // Parse optional flags
    let mut n_particles = 200;
    let mut max_iters = 1500;
    let mut stall_iters = 60;
    let mut output_path: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--particles" => { n_particles = args[i + 1].parse().unwrap(); i += 2; }
            "--max-iters" => { max_iters = args[i + 1].parse().unwrap(); i += 2; }
            "--stall-iters" => { stall_iters = args[i + 1].parse().unwrap(); i += 2; }
            "--output" | "-o" => { output_path = Some(args[i + 1].clone()); i += 2; }
            _ => { i += 1; }
        }
    }

    let config = PsoConfig {
        n_particles,
        max_iters,
        stall_iters,
        ..PsoConfig::default()
    };

    // Load and preprocess all CSVs
    let t_load = Instant::now();
    let sources = load_sources(data_dir);
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Loaded {} sources in {:.1}ms", sources.len(), load_ms);

    if sources.is_empty() {
        eprintln!("No valid sources found in {}", data_dir);
        std::process::exit(1);
    }

    // Upload to GPU
    let t_upload = Instant::now();
    let gpu = GpuContext::new(0).expect("Failed to init GPU");
    let batch_data = GpuBatchData::new(&sources).expect("Failed to upload data to GPU");
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;
    eprintln!("GPU upload: {:.1}ms", upload_ms);

    // Run batch PSO
    let t_pso = Instant::now();
    let results = gpu.batch_pso_multi_seed(&batch_data, &sources, &config)
        .expect("GPU batch PSO failed");
    let pso_ms = t_pso.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "PSO: {:.1}ms ({:.2}ms/source) | Total: {:.1}ms",
        pso_ms,
        pso_ms / sources.len() as f64,
        total_ms,
    );

    // Print summary
    println!("{:<20} {:>12}", "object", "reduced_chi2");
    println!("{}", "-".repeat(34));
    for (src, res) in sources.iter().zip(results.iter()) {
        println!("{:<20} {:>12.6}", src.name, res.reduced_chi2);
    }

    // Write CSV output if requested
    if let Some(path) = output_path {
        let mut wtr = std::fs::File::create(&path).expect("Cannot create output file");
        use std::io::Write;

        // Header
        write!(wtr, "name").unwrap();
        for filt in &FILTERS {
            for pname in &PARAM_NAMES {
                write!(wtr, ",{}_{}", pname, filt).unwrap();
            }
        }
        writeln!(wtr, ",reduced_chi2").unwrap();

        // Rows
        for (src, res) in sources.iter().zip(results.iter()) {
            write!(wtr, "{}", src.name).unwrap();
            for (filt_idx, _filt) in FILTERS.iter().enumerate() {
                for (p_idx, _pname) in PARAM_NAMES.iter().enumerate() {
                    write!(wtr, ",{}", res.phys_params[filt_idx * N_BASE + p_idx]).unwrap();
                }
            }
            writeln!(wtr, ",{}", res.reduced_chi2).unwrap();
        }
        eprintln!("Results written to {}", path);
    }
}
