//! Metal STREAM bandwidth ceiling probe.
//!
//! Kernel-grade empirical bandwidth for custom MSL work (HLD Phase 3 / #8).
//! Host is Rust + `metal` crate; GPU kernels are MSL (`shaders/stream.metal`).
//!
//! **Correctness before performance:** use [`verify_stream_kernels`] (host oracle)
//! in unit tests; [`run_stream`] measures GB/s only after kernels are proven.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct StreamKernelResult {
    pub name: String,
    pub gbs: f64,
    pub streams: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamReport {
    pub method: String,
    pub device: String,
    pub buffer_bytes: usize,
    pub n_float4: u64,
    pub warmup: u32,
    pub iterations: u32,
    pub kernels: Vec<StreamKernelResult>,
    pub best_gbs: f64,
    pub best_kernel: String,
    pub notes: String,
}

/// STREAM scalar `s` used by scale/triad (matches host fill in performance path).
pub const STREAM_SCALAR: f32 = 1.0001;

/// CPU oracles — pure host math for STREAM kernels (f32 element streams).
/// Used by unit tests and GPU correctness checks. Not performance-sensitive.
pub mod oracle {
    use super::STREAM_SCALAR;

    pub fn copy(a: &[f32], c: &mut [f32]) {
        assert_eq!(a.len(), c.len());
        c.copy_from_slice(a);
    }

    pub fn scale(c: &[f32], b: &mut [f32], s: f32) {
        assert_eq!(c.len(), b.len());
        for i in 0..c.len() {
            b[i] = s * c[i];
        }
    }

    pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        for i in 0..a.len() {
            c[i] = a[i] + b[i];
        }
    }

    pub fn triad(b: &[f32], c: &[f32], a: &mut [f32], s: f32) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        for i in 0..a.len() {
            a[i] = b[i] + s * c[i];
        }
    }

    /// Deterministic fill for element `i` of buffers a/b/c.
    pub fn fill_abc(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        let mut c = vec![0.0f32; n];
        for i in 0..n {
            a[i] = (i as f32) * 0.001 + 1.0;
            b[i] = (i as f32) * 0.002 + 2.0;
            c[i] = (i as f32) * 0.003 + 3.0;
        }
        let _ = STREAM_SCALAR;
        (a, b, c)
    }

    pub fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let mut m = 0.0f32;
        for i in 0..x.len() {
            m = m.max((x[i] - y[i]).abs());
        }
        m
    }
}

/// Run STREAM Copy / Scale / Add / Triad on the default Metal device (performance).
///
/// `n_float4` elements per buffer (each element 16 bytes). Default-sized
/// runs use 64M float4 ≈ 1 GiB per buffer.
///
/// Does **not** assert absolute GB/s — machine-dependent. Use
/// [`verify_stream_kernels`] for correctness.
#[cfg(target_os = "macos")]
pub fn run_stream(n_float4: u64, warmup: u32, iterations: u32) -> Result<StreamReport, String> {
    macos::run_stream_impl(n_float4, warmup, iterations)
}

#[cfg(not(target_os = "macos"))]
pub fn run_stream(_n_float4: u64, _warmup: u32, _iterations: u32) -> Result<StreamReport, String> {
    Err("metal_stream requires macOS + Metal".into())
}

/// Host-oracle correctness for all four MSL STREAM kernels (small buffers).
///
/// Returns `Ok(())` when GPU outputs match CPU reference within `atol`.
#[cfg(target_os = "macos")]
pub fn verify_stream_kernels(n_float4: u64, atol: f32) -> Result<(), String> {
    macos::verify_stream_kernels_impl(n_float4, atol)
}

#[cfg(not(target_os = "macos"))]
pub fn verify_stream_kernels(_n_float4: u64, _atol: f32) -> Result<(), String> {
    Err("metal_stream requires macOS + Metal".into())
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use metal::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn shader_source() -> Result<String, String> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest.join("shaders/stream.metal");
        if path.is_file() {
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))
        } else {
            Ok(include_str!("../shaders/stream.metal").to_string())
        }
    }

    fn pipeline(
        device: &Device,
        lib: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("function {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    }

    struct Context {
        device: Device,
        queue: CommandQueue,
        p_copy: ComputePipelineState,
        p_scale: ComputePipelineState,
        p_add: ComputePipelineState,
        p_triad: ComputePipelineState,
    }

    impl Context {
        fn new() -> Result<Self, String> {
            let device = Device::system_default().ok_or("no Metal device")?;
            let src = shader_source()?;
            let lib = device
                .new_library_with_source(&src, &CompileOptions::new())
                .map_err(|e| format!("compile MSL: {e}"))?;
            Ok(Self {
                p_copy: pipeline(&device, &lib, "stream_copy")?,
                p_scale: pipeline(&device, &lib, "stream_scale")?,
                p_add: pipeline(&device, &lib, "stream_add")?,
                p_triad: pipeline(&device, &lib, "stream_triad")?,
                queue: device.new_command_queue(),
                device,
            })
        }

        fn dispatch(
            &self,
            pipe: &ComputePipelineState,
            n_float4: u64,
            bind: &dyn Fn(&ComputeCommandEncoderRef),
        ) {
            let tg = MTLSize::new(256, 1, 1);
            let grid = MTLSize::new(n_float4, 1, 1);
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            bind(enc);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    fn write_f32_buffer(buf: &Buffer, data: &[f32]) {
        unsafe {
            let ptr = buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    fn read_f32_buffer(buf: &Buffer, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        unsafe {
            let ptr = buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        out
    }

    pub fn verify_stream_kernels_impl(n_float4: u64, atol: f32) -> Result<(), String> {
        if n_float4 == 0 {
            return Err("n_float4 must be > 0".into());
        }
        let ctx = Context::new()?;
        let n = (n_float4 as usize).checked_mul(4).ok_or("n overflow")?;
        let bytes = n * std::mem::size_of::<f32>();
        let opts = MTLResourceOptions::StorageModeShared;

        let (a0, b0, c0) = oracle::fill_abc(n);
        let s = STREAM_SCALAR;

        // --- copy: c = a ---
        {
            let a = ctx.device.new_buffer(bytes as u64, opts);
            let c = ctx.device.new_buffer(bytes as u64, opts);
            write_f32_buffer(&a, &a0);
            write_f32_buffer(&c, &vec![0.0f32; n]);
            ctx.dispatch(&ctx.p_copy, n_float4, &|e| {
                e.set_buffer(0, Some(&a), 0);
                e.set_buffer(1, Some(&c), 0);
            });
            let got = read_f32_buffer(&c, n);
            let mut exp = vec![0.0f32; n];
            oracle::copy(&a0, &mut exp);
            let err = oracle::max_abs_diff(&got, &exp);
            if err > atol {
                return Err(format!(
                    "stream_copy mismatch max_abs_diff={err} atol={atol}"
                ));
            }
        }

        // --- scale: b = s * c ---
        {
            let c = ctx.device.new_buffer(bytes as u64, opts);
            let b = ctx.device.new_buffer(bytes as u64, opts);
            let sbuf = ctx.device.new_buffer(4, opts);
            write_f32_buffer(&c, &c0);
            write_f32_buffer(&b, &vec![0.0f32; n]);
            write_f32_buffer(&sbuf, &[s]);
            ctx.dispatch(&ctx.p_scale, n_float4, &|e| {
                e.set_buffer(0, Some(&c), 0);
                e.set_buffer(1, Some(&b), 0);
                e.set_buffer(2, Some(&sbuf), 0);
            });
            let got = read_f32_buffer(&b, n);
            let mut exp = vec![0.0f32; n];
            oracle::scale(&c0, &mut exp, s);
            let err = oracle::max_abs_diff(&got, &exp);
            if err > atol {
                return Err(format!(
                    "stream_scale mismatch max_abs_diff={err} atol={atol}"
                ));
            }
        }

        // --- add: c = a + b ---
        {
            let a = ctx.device.new_buffer(bytes as u64, opts);
            let b = ctx.device.new_buffer(bytes as u64, opts);
            let c = ctx.device.new_buffer(bytes as u64, opts);
            write_f32_buffer(&a, &a0);
            write_f32_buffer(&b, &b0);
            write_f32_buffer(&c, &vec![0.0f32; n]);
            ctx.dispatch(&ctx.p_add, n_float4, &|e| {
                e.set_buffer(0, Some(&a), 0);
                e.set_buffer(1, Some(&b), 0);
                e.set_buffer(2, Some(&c), 0);
            });
            let got = read_f32_buffer(&c, n);
            let mut exp = vec![0.0f32; n];
            oracle::add(&a0, &b0, &mut exp);
            let err = oracle::max_abs_diff(&got, &exp);
            if err > atol {
                return Err(format!(
                    "stream_add mismatch max_abs_diff={err} atol={atol}"
                ));
            }
        }

        // --- triad: a = b + s * c ---
        {
            let b = ctx.device.new_buffer(bytes as u64, opts);
            let c = ctx.device.new_buffer(bytes as u64, opts);
            let a = ctx.device.new_buffer(bytes as u64, opts);
            let sbuf = ctx.device.new_buffer(4, opts);
            write_f32_buffer(&b, &b0);
            write_f32_buffer(&c, &c0);
            write_f32_buffer(&a, &vec![0.0f32; n]);
            write_f32_buffer(&sbuf, &[s]);
            ctx.dispatch(&ctx.p_triad, n_float4, &|e| {
                e.set_buffer(0, Some(&b), 0);
                e.set_buffer(1, Some(&c), 0);
                e.set_buffer(2, Some(&a), 0);
                e.set_buffer(3, Some(&sbuf), 0);
            });
            let got = read_f32_buffer(&a, n);
            let mut exp = vec![0.0f32; n];
            oracle::triad(&b0, &c0, &mut exp, s);
            let err = oracle::max_abs_diff(&got, &exp);
            if err > atol {
                return Err(format!(
                    "stream_triad mismatch max_abs_diff={err} atol={atol}"
                ));
            }
        }

        Ok(())
    }

    pub fn run_stream_impl(
        n_float4: u64,
        warmup: u32,
        iterations: u32,
    ) -> Result<StreamReport, String> {
        let ctx = Context::new()?;
        let device_name = ctx.device.name().to_string();

        let bytes = (n_float4 as usize)
            .checked_mul(16)
            .ok_or("buffer size overflow")?;
        let opts = MTLResourceOptions::StorageModeShared;
        let a = ctx.device.new_buffer(bytes as u64, opts);
        let b = ctx.device.new_buffer(bytes as u64, opts);
        let c = ctx.device.new_buffer(bytes as u64, opts);
        let sbuf = ctx.device.new_buffer(4, opts);

        unsafe {
            *(sbuf.contents() as *mut f32) = STREAM_SCALAR;
            let n = bytes / 4;
            let ap = a.contents() as *mut f32;
            let bp = b.contents() as *mut f32;
            let cp = c.contents() as *mut f32;
            for i in 0..n {
                *ap.add(i) = 1.0;
                *bp.add(i) = 2.0;
                *cp.add(i) = 3.0;
            }
        }

        let tg = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(n_float4, 1, 1);

        let timed = |pipe: &ComputePipelineState,
                     streams: f64,
                     bind: &dyn Fn(&ComputeCommandEncoderRef)|
         -> f64 {
            for _ in 0..warmup {
                let cmd = ctx.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                bind(enc);
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            let mut best = 0.0_f64;
            for _ in 0..iterations {
                let t0 = Instant::now();
                let cmd = ctx.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                bind(enc);
                enc.dispatch_threads(grid, tg);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let dt = t0.elapsed().as_secs_f64();
                if dt > 0.0 {
                    best = best.max((streams * bytes as f64) / dt / 1e9);
                }
            }
            best
        };

        let copy = timed(&ctx.p_copy, 2.0, &|e| {
            e.set_buffer(0, Some(&a), 0);
            e.set_buffer(1, Some(&c), 0);
        });
        let scale = timed(&ctx.p_scale, 2.0, &|e| {
            e.set_buffer(0, Some(&c), 0);
            e.set_buffer(1, Some(&b), 0);
            e.set_buffer(2, Some(&sbuf), 0);
        });
        let add = timed(&ctx.p_add, 3.0, &|e| {
            e.set_buffer(0, Some(&a), 0);
            e.set_buffer(1, Some(&b), 0);
            e.set_buffer(2, Some(&c), 0);
        });
        let triad = timed(&ctx.p_triad, 3.0, &|e| {
            e.set_buffer(0, Some(&b), 0);
            e.set_buffer(1, Some(&c), 0);
            e.set_buffer(2, Some(&a), 0);
            e.set_buffer(3, Some(&sbuf), 0);
        });

        let kernels = vec![
            StreamKernelResult {
                name: "copy".into(),
                gbs: copy,
                streams: 2.0,
            },
            StreamKernelResult {
                name: "scale".into(),
                gbs: scale,
                streams: 2.0,
            },
            StreamKernelResult {
                name: "add".into(),
                gbs: add,
                streams: 3.0,
            },
            StreamKernelResult {
                name: "triad".into(),
                gbs: triad,
                streams: 3.0,
            },
        ];
        let best = kernels
            .iter()
            .max_by(|x, y| x.gbs.partial_cmp(&y.gbs).unwrap())
            .unwrap();

        Ok(StreamReport {
            method: "rust_metal_stream_float4".into(),
            device: device_name,
            buffer_bytes: bytes,
            n_float4,
            warmup,
            iterations,
            best_gbs: best.gbs,
            best_kernel: best.name.clone(),
            kernels,
            notes: "MSL STREAM (copy/scale/add/triad) via metal-rs host; \
                    peak wall-clock GB/s after warmup. Kernel-grade ceiling for \
                    custom Metal work (HLD Phase 3). Not Apple theoretical peak. \
                    Correctness: cargo test (host oracle)."
                .into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_copy_scale_add_triad_are_consistent() {
        let n = 128;
        let (a, b, c0) = oracle::fill_abc(n);
        let s = STREAM_SCALAR;

        let mut c = vec![0.0f32; n];
        oracle::copy(&a, &mut c);
        assert_eq!(oracle::max_abs_diff(&c, &a), 0.0);

        let mut b_out = vec![0.0f32; n];
        oracle::scale(&c0, &mut b_out, s);
        for i in 0..n {
            assert!((b_out[i] - s * c0[i]).abs() < 1e-6);
        }

        let mut c_add = vec![0.0f32; n];
        oracle::add(&a, &b, &mut c_add);
        for i in 0..n {
            assert!((c_add[i] - (a[i] + b[i])).abs() < 1e-6);
        }

        let mut a_tri = vec![0.0f32; n];
        oracle::triad(&b, &c0, &mut a_tri, s);
        for i in 0..n {
            assert!((a_tri[i] - (b[i] + s * c0[i])).abs() < 1e-5);
        }
    }

    #[test]
    fn report_contract_or_skip_without_metal() {
        // Tiny performance probe — no absolute GB/s floor (machine-dependent).
        let n = 256 * 1024; // 4 MiB / buffer as float4 count... wait 256k * 16 = 4 MiB
        match run_stream(n, 1, 2) {
            Ok(r) => {
                assert_eq!(r.method, "rust_metal_stream_float4");
                assert_eq!(r.kernels.len(), 4);
                assert!(r.best_gbs > 0.0);
                assert!(r.best_gbs < 10_000.0, "absurd bandwidth: {}", r.best_gbs);
                let names: Vec<_> = r.kernels.iter().map(|k| k.name.as_str()).collect();
                assert_eq!(names, ["copy", "scale", "add", "triad"]);
                assert!(names.contains(&r.best_kernel.as_str()));
                for k in &r.kernels {
                    assert!(k.streams == 2.0 || k.streams == 3.0);
                }
            }
            Err(e) => {
                assert!(
                    e.contains("macOS") || e.contains("Metal"),
                    "unexpected error: {e}"
                );
            }
        }
    }

    #[test]
    fn msl_kernels_match_host_oracle_or_skip() {
        // Small grid: correctness, not performance. 4096 float4 = 64 KiB/buffer.
        let n_float4 = 4096u64;
        match verify_stream_kernels(n_float4, 1e-4) {
            Ok(()) => {}
            Err(e) => {
                assert!(
                    e.contains("macOS") || e.contains("Metal"),
                    "kernel correctness failed (not a platform skip): {e}"
                );
            }
        }
    }

    #[test]
    fn msl_kernels_non_multiple_of_threadgroup_or_skip() {
        // 1000 is not a multiple of 256 — dispatch_threads must still cover all elements.
        let n_float4 = 1000u64;
        match verify_stream_kernels(n_float4, 1e-4) {
            Ok(()) => {}
            Err(e) => {
                assert!(
                    e.contains("macOS") || e.contains("Metal"),
                    "kernel correctness failed (not a platform skip): {e}"
                );
            }
        }
    }
}
