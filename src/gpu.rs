//! Lightweight CUDA discovery used by the external TinyLocate runtime.

use pyo3::prelude::*;

#[cfg(target_os = "windows")]
const TINYLOCATE_PTX: &str = include_str!("cuda/tinylocate.ptx");

#[cfg(target_os = "windows")]
fn query() -> Option<(String, i32, i32, u64)> {
    use cudarc::driver::CudaContext;

    let context = CudaContext::new(0).ok()?;
    let name = context.name().ok()?;
    let (major, minor) = context.compute_capability().ok()?;
    let memory = u64::try_from(context.total_mem().ok()?).ok()?;
    Some((name, major, minor, memory))
}

#[cfg(target_os = "windows")]
fn self_test() -> Option<bool> {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    let context = CudaContext::new(0).ok()?;
    let stream = context.default_stream();
    let module = context.load_module(Ptx::from_src(TINYLOCATE_PTX)).ok()?;
    let function = module.load_function("tln_conv2d").ok()?;
    let input = stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).ok()?;
    let weight = stream.clone_htod(&[2.0f32]).ok()?;
    let bias = stream.clone_htod(&[1.0f32]).ok()?;
    let mut output = stream.alloc_zeros::<f32>(4).ok()?;
    let mut launch = stream.launch_builder(&function);
    launch.arg(&input);
    launch.arg(&weight);
    launch.arg(&bias);
    launch.arg(&mut output);
    let in_channels = 1i32;
    let input_height = 2i32;
    let input_width = 2i32;
    let out_channels = 1i32;
    let output_height = 2i32;
    let output_width = 2i32;
    let kernel = 1i32;
    let stride = 1i32;
    let padding = 0i32;
    let groups = 1i32;
    let activation = 0i32;
    launch.arg(&in_channels);
    launch.arg(&input_height);
    launch.arg(&input_width);
    launch.arg(&out_channels);
    launch.arg(&output_height);
    launch.arg(&output_width);
    launch.arg(&kernel);
    launch.arg(&stride);
    launch.arg(&padding);
    launch.arg(&groups);
    launch.arg(&activation);
    unsafe { launch.launch(LaunchConfig::for_num_elems(4)) }.ok()?;
    let result = stream.clone_dtoh(&output).ok()?;
    Some(result == [3.0, 5.0, 7.0, 9.0])
}

#[cfg(not(target_os = "windows"))]
fn self_test() -> Option<bool> {
    None
}

#[cfg(not(target_os = "windows"))]
fn query() -> Option<(String, i32, i32, u64)> {
    None
}

#[pyfunction]
fn tinylocate_gpu_info() -> Option<(String, i32, i32, u64)> {
    query()
}

#[pyfunction]
fn tinylocate_gpu_self_test() -> bool {
    self_test().unwrap_or(false)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(tinylocate_gpu_info, module)?)?;
    module.add_function(wrap_pyfunction!(tinylocate_gpu_self_test, module)?)?;
    Ok(())
}
