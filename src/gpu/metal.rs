//! Metal backend scaffold.
//!
//! This module is intentionally a stub for now. It provides the same shape as
//! the CUDA backend so the rest of the crate can be backend-agnostic while the
//! Metal kernel and command encoder path are implemented later.

use crate::{FitResult, PsoConfig};

use super::SourceData;

pub struct MetalBatchData;

pub struct MetalContext {
    device: i32,
}

impl MetalContext {
    pub fn new(device: i32) -> Result<Self, String> {
        Ok(Self { device })
    }

    pub fn set_device(&self) -> Result<(), String> {
        Ok(())
    }

    pub fn device_id(&self) -> i32 {
        self.device
    }

    pub fn pack_batch<S: AsRef<SourceData>>(
        &self,
        _sources: &[S],
    ) -> Result<MetalBatchData, String> {
        Err("Metal backend not implemented yet".to_string())
    }

    pub fn batch_pso<S: AsRef<SourceData>>(
        &self,
        _data: &MetalBatchData,
        _sources: &[S],
        _config: &PsoConfig,
        _seed: u64,
    ) -> Result<Vec<FitResult>, String> {
        Err("Metal backend not implemented yet".to_string())
    }

    pub fn batch_pso_multi_seed<S: AsRef<SourceData>>(
        &self,
        _data: &MetalBatchData,
        _sources: &[S],
        _config: &PsoConfig,
    ) -> Result<Vec<FitResult>, String> {
        Err("Metal backend not implemented yet".to_string())
    }
}

#[allow(dead_code)]
pub fn detect_gpu_count() -> usize {
    1
}
