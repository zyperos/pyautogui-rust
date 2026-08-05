use pyo3::prelude::*;

mod gpu;
#[cfg(target_os = "windows")]
mod gpu_engine;
pub mod tinylocate;
pub mod vision;

#[cfg(not(target_os = "windows"))]
mod dummy;
#[cfg(target_os = "windows")]
mod win;

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    gpu::register(m)?;
    tinylocate::register(m)?;
    #[cfg(target_os = "windows")]
    win::register(m)?;

    #[cfg(not(target_os = "windows"))]
    dummy::register(m)?;

    Ok(())
}
