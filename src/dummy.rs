use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn platform_error() -> PyErr {
    PyOSError::new_err("the Rust native backend is currently Windows-specific")
}

#[pyfunction]
fn set_process_dpi_aware() -> PyResult<bool> {
    Ok(false)
}

#[pyfunction]
fn time_begin_period(_period: u32) -> PyResult<u32> {
    Ok(0)
}

#[pyfunction]
fn time_end_period(_period: u32) -> PyResult<u32> {
    Ok(0)
}

#[pyfunction]
fn get_cursor_pos() -> PyResult<(i32, i32)> {
    Err(platform_error())
}

#[pyfunction]
fn get_system_metrics(_index: i32) -> PyResult<i32> {
    Ok(0)
}

#[pyfunction]
fn set_cursor_pos(_x: i32, _y: i32) -> PyResult<()> {
    Err(platform_error())
}

#[pyfunction]
fn send_mouse_event(_ev: u32, _x: i32, _y: i32, _data: i32) -> PyResult<()> {
    Err(platform_error())
}

#[pyfunction]
fn send_keyboard_event(_vk: u8, _scan: u8, _flags: u32) -> PyResult<()> {
    Err(platform_error())
}

#[pyfunction]
fn vk_key_scan_a(_c: u8) -> PyResult<i16> {
    Ok(-1)
}

#[pyfunction]
fn mouse_is_swapped() -> PyResult<bool> {
    Ok(false)
}

#[pyfunction]
fn move_rel(_dx: i32, _dy: i32) -> PyResult<()> {
    Err(platform_error())
}

#[pyfunction]
fn send_inputs(_events: Vec<Bound<'_, PyTuple>>) -> PyResult<u32> {
    Ok(0)
}

#[pyfunction]
fn move_to_smooth(_x: i32, _y: i32, _duration: f64, _steps: u32) -> PyResult<()> {
    Err(platform_error())
}

#[pyfunction(signature = (region=None))]
fn capture_screen_gdi(_py: Python<'_>, region: Option<(i32, i32, i32, i32)>) -> PyResult<PyObject> {
    let _ = region;
    Err(platform_error())
}

#[pyfunction(signature = (needle_bytes, needle_w, needle_h, confidence, region=None))]
fn locate_on_screen_rust(
    needle_bytes: &[u8],
    needle_w: usize,
    needle_h: usize,
    confidence: f32,
    region: Option<(i32, i32, i32, i32)>,
) -> PyResult<Option<(i32, i32, i32, i32)>> {
    let _ = (needle_bytes, needle_w, needle_h, confidence, region);
    Err(platform_error())
}

#[pyfunction]
fn start_failsafe_hook() -> PyResult<()> {
    Ok(())
}

#[pyfunction]
fn check_failsafe_triggered() -> PyResult<bool> {
    Ok(false)
}

#[pyfunction]
fn reset_failsafe_triggered() -> PyResult<()> {
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_process_dpi_aware, module)?)?;
    module.add_function(wrap_pyfunction!(get_cursor_pos, module)?)?;
    module.add_function(wrap_pyfunction!(get_system_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(set_cursor_pos, module)?)?;
    module.add_function(wrap_pyfunction!(send_mouse_event, module)?)?;
    module.add_function(wrap_pyfunction!(send_keyboard_event, module)?)?;
    module.add_function(wrap_pyfunction!(vk_key_scan_a, module)?)?;
    module.add_function(wrap_pyfunction!(mouse_is_swapped, module)?)?;
    module.add_function(wrap_pyfunction!(move_rel, module)?)?;
    module.add_function(wrap_pyfunction!(time_begin_period, module)?)?;
    module.add_function(wrap_pyfunction!(time_end_period, module)?)?;
    module.add_function(wrap_pyfunction!(send_inputs, module)?)?;
    module.add_function(wrap_pyfunction!(move_to_smooth, module)?)?;
    module.add_function(wrap_pyfunction!(capture_screen_gdi, module)?)?;
    module.add_function(wrap_pyfunction!(start_failsafe_hook, module)?)?;
    module.add_function(wrap_pyfunction!(check_failsafe_triggered, module)?)?;
    module.add_function(wrap_pyfunction!(reset_failsafe_triggered, module)?)?;
    module.add_function(wrap_pyfunction!(locate_on_screen_rust, module)?)?;
    Ok(())
}
