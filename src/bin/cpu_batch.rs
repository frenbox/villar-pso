//! CPU batch fitting for all CSVs in a directory (for timing comparison with GPU).

use std::time::Instant;
use villar_pso::{fit_lightcurve, FILTERS, N_BASE, PARAM_NAMES};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.get(1).map(|s| s.as_str()).unwrap_or("../data/photometry");
    let output_path = args.get(2).map(|s| s.as_str());

    let mut csvs: Vec<String> = std::fs::read_dir(data_dir)
        .expect("Cannot read data directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    csvs.sort();

    eprintln!("Found {} CSV files in {}", csvs.len(), data_dir);

    let t_start = Instant::now();
    let mut results = Vec::new();
    let mut n_ok = 0usize;
    let mut n_skip = 0usize;

    for csv in &csvs {
        let name = std::path::Path::new(csv)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        match fit_lightcurve(csv) {
            Ok(res) => {
                results.push((name, res));
                n_ok += 1;
            }
            Err(e) => {
                eprintln!("SKIP {}: {}", name, e);
                n_skip += 1;
            }
        }
    }
    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "\nCPU batch: {} sources fitted, {} skipped in {:.1}ms ({:.2}ms/source)",
        n_ok, n_skip, total_ms, total_ms / n_ok.max(1) as f64,
    );

    // Print summary
    println!("{:<20} {:>12}", "object", "reduced_chi2");
    println!("{}", "-".repeat(34));
    for (name, res) in &results {
        println!("{:<20} {:>12.6}", name, res.reduced_chi2);
    }

    // Write CSV output if requested
    if let Some(path) = output_path {
        let mut wtr = std::fs::File::create(path).expect("Cannot create output file");
        use std::io::Write;
        write!(wtr, "name").unwrap();
        for filt in &FILTERS {
            for pname in &PARAM_NAMES {
                write!(wtr, ",{}_{}", pname, filt).unwrap();
            }
        }
        writeln!(wtr, ",reduced_chi2").unwrap();
        for (name, res) in &results {
            write!(wtr, "{}", name).unwrap();
            for (filt_idx, _) in FILTERS.iter().enumerate() {
                for (p_idx, _) in PARAM_NAMES.iter().enumerate() {
                    write!(wtr, ",{}", res.phys_params[filt_idx * N_BASE + p_idx]).unwrap();
                }
            }
            writeln!(wtr, ",{}", res.reduced_chi2).unwrap();
        }
        eprintln!("Results written to {}", path);
    }
}
