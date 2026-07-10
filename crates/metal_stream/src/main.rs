//! CLI: Metal STREAM bandwidth ceiling (JSON on stdout with --json).

use metal_stream::run_stream;
use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut json = false;
    let mut n_float4: u64 = 64 * 1024 * 1024; // 1 GiB / buffer
    let mut warmup: u32 = 5;
    let mut iterations: u32 = 30;

    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json = true,
            "--n-float4" => {
                n_float4 = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--n-float4 requires integer");
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--warmup requires integer");
            }
            "--iterations" => {
                iterations = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--iterations requires integer");
            }
            "--help" | "-h" => {
                eprintln!(
                    "metal_stream — Metal STREAM bandwidth ceiling\n\
                     \n\
                     Usage: metal_stream [--json] [--n-float4 N] [--warmup W] [--iterations I]\n\
                     Default: N=67108864 float4 (1 GiB/buffer), W=5, I=30"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::from(2);
            }
        }
    }

    match run_stream(n_float4, warmup, iterations) {
        Ok(report) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("Metal STREAM bandwidth ceiling");
                println!("  method:  {}", report.method);
                println!("  device:  {}", report.device);
                println!(
                    "  buffer:  {:.2} GiB ({} float4)",
                    report.buffer_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    report.n_float4
                );
                for k in &report.kernels {
                    println!("  {:<6} {:.2} GB/s  ({} streams)", k.name, k.gbs, k.streams);
                }
                println!(
                    "  best:   {:.2} GB/s  ({})",
                    report.best_gbs, report.best_kernel
                );
                println!("  notes:  {}", report.notes);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("metal_stream error: {e}");
            ExitCode::from(1)
        }
    }
}
