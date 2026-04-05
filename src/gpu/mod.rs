use crate::{preprocess, FitResult, PreprocessedData, PsoConfig};

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
mod host_shared;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "cuda")]
use cuda::{CudaBatchData, CudaContext};
#[cfg(feature = "metal")]
use metal::{MetalBatchData, MetalContext};

/// Preprocessed data for one source, ready for accelerator packing.
pub struct SourceData {
    pub name: String,
    pub data: PreprocessedData,
}

impl AsRef<SourceData> for SourceData {
    fn as_ref(&self) -> &SourceData {
        self
    }
}

/// Available accelerator backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "metal")]
    Metal,
}

enum BackendContext {
    #[cfg(feature = "cuda")]
    Cuda(CudaContext),
    #[cfg(feature = "metal")]
    Metal(MetalContext),
}

/// Batch data uploaded for a specific backend.
pub enum GpuBatchData {
    #[cfg(feature = "cuda")]
    Cuda(CudaBatchData),
    #[cfg(feature = "metal")]
    Metal(MetalBatchData),
}

/// Generic accelerator context that dispatches to CUDA or Metal.
pub struct GpuContext {
    backend: GpuBackend,
    inner: BackendContext,
}

impl GpuContext {
    /// Create a context using the default backend for this build.
    pub fn new(device: i32) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            return Self::new_with_backend(device, GpuBackend::Cuda);
        }

        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            return Self::new_with_backend(device, GpuBackend::Metal);
        }

        #[allow(unreachable_code)]
        Err("No GPU backend is enabled at compile-time".to_string())
    }

    /// Create a context for an explicitly selected backend.
    pub fn new_with_backend(device: i32, backend: GpuBackend) -> Result<Self, String> {
        match backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda => {
                let ctx = CudaContext::new(device)?;
                Ok(Self {
                    backend,
                    inner: BackendContext::Cuda(ctx),
                })
            }
            #[cfg(feature = "metal")]
            GpuBackend::Metal => {
                let ctx = MetalContext::new(device)?;
                Ok(Self {
                    backend,
                    inner: BackendContext::Metal(ctx),
                })
            }
        }
    }

    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    pub fn set_device(&self) -> Result<(), String> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            BackendContext::Cuda(ctx) => ctx.set_device(),
            #[cfg(feature = "metal")]
            BackendContext::Metal(ctx) => ctx.set_device(),
        }
    }

    pub fn device_id(&self) -> i32 {
        match &self.inner {
            #[cfg(feature = "cuda")]
            BackendContext::Cuda(ctx) => ctx.device_id(),
            #[cfg(feature = "metal")]
            BackendContext::Metal(ctx) => ctx.device_id(),
        }
    }

    pub fn pack_batch<S: AsRef<SourceData>>(&self, sources: &[S]) -> Result<GpuBatchData, String> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            BackendContext::Cuda(_) => Ok(GpuBatchData::Cuda(CudaBatchData::new(sources)?)),
            #[cfg(feature = "metal")]
            BackendContext::Metal(ctx) => Ok(GpuBatchData::Metal(ctx.pack_batch(sources)?)),
        }
    }

    pub fn batch_pso<S: AsRef<SourceData>>(
        &self,
        data: &GpuBatchData,
        sources: &[S],
        config: &PsoConfig,
        seed: u64,
    ) -> Result<Vec<FitResult>, String> {
        #[allow(unreachable_patterns)]
        match (&self.inner, data) {
            #[cfg(feature = "cuda")]
            (BackendContext::Cuda(ctx), GpuBatchData::Cuda(batch)) => {
                ctx.batch_pso(batch, sources, config, seed)
            }
            #[cfg(feature = "metal")]
            (BackendContext::Metal(ctx), GpuBatchData::Metal(batch)) => {
                ctx.batch_pso(batch, sources, config, seed)
            }
            _ => Err("GpuBatchData backend does not match GpuContext backend".to_string()),
        }
    }

    pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
        &self,
        data: &GpuBatchData,
        sources: &[S],
        config: &PsoConfig,
    ) -> Result<Vec<FitResult>, String> {
        #[allow(unreachable_patterns)]
        match (&self.inner, data) {
            #[cfg(feature = "cuda")]
            (BackendContext::Cuda(ctx), GpuBatchData::Cuda(batch)) => {
                ctx.batch_pso_multi_seed(batch, sources, config)
            }
            #[cfg(feature = "metal")]
            (BackendContext::Metal(ctx), GpuBatchData::Metal(batch)) => {
                ctx.batch_pso_multi_seed(batch, sources, config)
            }
            _ => Err("GpuBatchData backend does not match GpuContext backend".to_string()),
        }
    }
}

/// Load all CSVs from a directory, preprocess them, and return source data.
/// Sources that fail preprocessing are skipped (with a warning on stderr).
pub fn load_sources(data_dir: &str) -> Vec<SourceData> {
    let mut entries: Vec<_> = std::fs::read_dir(data_dir)
        .expect("Cannot read data directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut sources = Vec::new();
    for entry in entries {
        let path = entry.path();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let csv_path = path.to_string_lossy().to_string();
        match preprocess(&csv_path) {
            Ok(data) => sources.push(SourceData { name, data }),
            Err(e) => eprintln!("SKIP {}: {}", name, e),
        }
    }
    sources
}

#[cfg(feature = "cuda")]
pub fn detect_gpu_count() -> usize {
    cuda::detect_gpu_count()
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
pub fn detect_gpu_count() -> usize {
    metal::detect_gpu_count()
}

pub fn detect_gpu_count_for_backend(backend: GpuBackend) -> usize {
    match backend {
        #[cfg(feature = "cuda")]
        GpuBackend::Cuda => cuda::detect_gpu_count(),
        #[cfg(feature = "metal")]
        GpuBackend::Metal => metal::detect_gpu_count(),
    }
}
