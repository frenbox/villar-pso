use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: villar-pso <csv_path>");
        eprintln!("  Fits a ZTF light curve CSV with the joint two-band Villar model (PSO).");
        std::process::exit(1);
    }

    let csv_path = &args[1];

    // If path doesn't exist, try data/photometry/<name>
    let resolved = if Path::new(csv_path).exists() {
        csv_path.clone()
    } else {
        let candidate = format!("../data/photometry/{}", csv_path);
        if Path::new(&candidate).exists() {
            candidate
        } else {
            let candidate2 = format!("data/photometry/{}", csv_path);
            if Path::new(&candidate2).exists() {
                candidate2
            } else {
                eprintln!("File not found: {}", csv_path);
                std::process::exit(1);
            }
        }
    };

    eprintln!("Fitting: {}", resolved);

    // Extract name from CSV filename
    let name = Path::new(&resolved)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("lightcurve");

    let output_dir = if args.len() >= 3 { &args[2] } else { "superphot_results" };

    match villar_pso::fit_lightcurve(&resolved) {
        Ok(result) => {
            result.print_summary();

            // Plot
            match villar_pso::make_plot(&result, name, output_dir) {
                Ok(path) => eprintln!("Plot saved → {}", path),
                Err(e) => eprintln!("Plot failed: {}", e),
            }

            // CSV-friendly output
            println!("\n# CSV format: param_name,value");
            for (filt_idx, filt) in villar_pso::FILTERS.iter().enumerate() {
                for (p_idx, pname) in villar_pso::PARAM_NAMES.iter().enumerate() {
                    let idx = filt_idx * villar_pso::N_BASE + p_idx;
                    println!("{}_{},{}", pname, filt, result.phys_params[idx]);
                }
            }
            println!("reduced_chi2,{}", result.reduced_chi2);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
