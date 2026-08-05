mod capture;
mod hook;
mod input;

use std::io;
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyOSError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use windows_sys::Win32::Foundation::POINT;
use windows_sys::Win32::UI::HiDpi::{
    SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::VkKeyScanA;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetCursorPos, GetSystemMetrics, SetProcessDPIAware, SM_CXSCREEN, SM_CYSCREEN, SM_SWAPBUTTON,
};

use crate::vision;
use input::{NativeEvent, VirtualDesktop};

type LocatedVariant = (i32, i32, i32, i32, f32, usize);

#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
    fn timeEndPeriod(period: u32) -> u32;
}

fn to_py_os_error(error: io::Error) -> PyErr {
    PyOSError::new_err(error.to_string())
}

fn last_os_error(operation: &str) -> io::Error {
    let source = io::Error::last_os_error();
    io::Error::new(
        source.kind(),
        format!("{operation} failed (Windows error: {source})"),
    )
}

#[pyfunction]
fn set_process_dpi_aware() -> PyResult<bool> {
    let modern =
        unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    if modern != 0 {
        return Ok(true);
    }
    Ok(unsafe { SetProcessDPIAware() } != 0)
}

#[pyfunction]
fn time_begin_period(period: u32) -> PyResult<u32> {
    Ok(unsafe { timeBeginPeriod(period) })
}

#[pyfunction]
fn time_end_period(period: u32) -> PyResult<u32> {
    Ok(unsafe { timeEndPeriod(period) })
}

#[pyfunction]
fn get_cursor_pos() -> PyResult<(i32, i32)> {
    let mut point = POINT { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut point) } == 0 {
        Err(to_py_os_error(last_os_error("GetCursorPos")))
    } else {
        Ok((point.x, point.y))
    }
}

#[pyfunction]
fn get_system_metrics(index: i32) -> PyResult<i32> {
    Ok(unsafe { GetSystemMetrics(index) })
}

#[pyfunction]
fn set_cursor_pos(py: Python<'_>, x: i32, y: i32) -> PyResult<()> {
    py.allow_threads(move || input::send_absolute_mouse(x, y, 0, 0x0001))
        .map_err(to_py_os_error)
}

#[pyfunction]
fn send_mouse_event(py: Python<'_>, ev: u32, x: i32, y: i32, data: i32) -> PyResult<()> {
    py.allow_threads(move || input::send_absolute_mouse(x, y, data as u32, ev))
        .map_err(to_py_os_error)
}

#[pyfunction]
fn send_keyboard_event(py: Python<'_>, vk: u8, scan: u8, flags: u32) -> PyResult<()> {
    py.allow_threads(move || input::send_keyboard(vk, scan, flags))
        .map_err(to_py_os_error)
}

#[pyfunction]
fn vk_key_scan_a(c: u8) -> PyResult<i16> {
    Ok(unsafe { VkKeyScanA(c) })
}

#[pyfunction]
fn mouse_is_swapped() -> PyResult<bool> {
    Ok(unsafe { GetSystemMetrics(SM_SWAPBUTTON) } != 0)
}

#[pyfunction]
fn move_rel(py: Python<'_>, dx: i32, dy: i32) -> PyResult<()> {
    py.allow_threads(move || input::send_relative_mouse(dx, dy))
        .map_err(to_py_os_error)
}

fn require_tuple_len(tuple: &Bound<'_, PyTuple>, minimum: usize, index: usize) -> PyResult<()> {
    if tuple.len() < minimum {
        Err(PyValueError::new_err(format!(
            "event {index} requires at least {minimum} fields, got {}",
            tuple.len()
        )))
    } else {
        Ok(())
    }
}

fn parse_mouse_data(value: i64, event_index: usize) -> PyResult<u32> {
    if value < i32::MIN as i64 || value > u32::MAX as i64 {
        Err(PyOverflowError::new_err(format!(
            "mouse data in event {event_index} is outside the 32-bit range"
        )))
    } else {
        Ok(value as u32)
    }
}

fn parse_events(events: Vec<Bound<'_, PyTuple>>) -> PyResult<Vec<NativeEvent>> {
    let mut parsed = Vec::with_capacity(events.len());
    for (index, event) in events.into_iter().enumerate() {
        require_tuple_len(&event, 1, index)?;
        let event_type: u32 = event.get_item(0)?.extract()?;
        match event_type {
            0 => {
                require_tuple_len(&event, 5, index)?;
                let x = event.get_item(1)?.extract()?;
                let y = event.get_item(2)?.extract()?;
                let mouse_data = parse_mouse_data(event.get_item(3)?.extract()?, index)?;
                let flags = event.get_item(4)?.extract()?;
                parsed.push(NativeEvent::Mouse {
                    x,
                    y,
                    mouse_data,
                    flags,
                });
            }
            1 => {
                require_tuple_len(&event, 4, index)?;
                parsed.push(NativeEvent::Keyboard {
                    virtual_key: event.get_item(1)?.extract()?,
                    scan_code: event.get_item(2)?.extract()?,
                    flags: event.get_item(3)?.extract()?,
                });
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid event type {other} at event {index}"
                )));
            }
        }
    }
    Ok(parsed)
}

#[pyfunction]
fn send_inputs(py: Python<'_>, events: Vec<Bound<'_, PyTuple>>) -> PyResult<u32> {
    let parsed = parse_events(events)?;
    py.allow_threads(move || input::send_events(&parsed))
        .map_err(to_py_os_error)
}

struct TimerResolutionGuard(bool);

impl TimerResolutionGuard {
    fn one_millisecond() -> Self {
        Self(unsafe { timeBeginPeriod(1) } == 0)
    }
}

impl Drop for TimerResolutionGuard {
    fn drop(&mut self) {
        if self.0 {
            unsafe {
                timeEndPeriod(1);
            }
        }
    }
}

fn smooth_move(x: i32, y: i32, duration: f64, steps: u32) -> io::Result<()> {
    if !duration.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "movement duration must be finite",
        ));
    }
    if duration <= 0.0 || steps <= 1 {
        return input::send_absolute_mouse(x, y, 0, 0x0001);
    }

    let mut current = POINT { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut current) } == 0 {
        return Err(last_os_error("GetCursorPos"));
    }
    let desktop = VirtualDesktop::from_system()?;
    let total_duration = Duration::try_from_secs_f64(duration).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "movement duration is outside the supported timer range",
        )
    })?;
    let started = Instant::now();
    let _timer_resolution = TimerResolutionGuard::one_millisecond();

    for step in 1..=steps {
        let progress = step as f64 / steps as f64;
        let deadline = started
            .checked_add(total_duration.mul_f64(progress))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "timer deadline overflow")
            })?;
        let now = Instant::now();
        if deadline > now {
            std::thread::sleep(deadline - now);
        }

        let next_x = (current.x as f64 + (x as f64 - current.x as f64) * progress).round() as i32;
        let next_y = (current.y as f64 + (y as f64 - current.y as f64) * progress).round() as i32;
        input::send_absolute_mouse_on_desktop(next_x, next_y, 0, 0x0001, desktop)?;
    }
    Ok(())
}

#[pyfunction]
fn move_to_smooth(py: Python<'_>, x: i32, y: i32, duration: f64, steps: u32) -> PyResult<()> {
    py.allow_threads(move || smooth_move(x, y, duration, steps))
        .map_err(to_py_os_error)
}

fn default_capture_rect() -> io::Result<(i32, i32, i32, i32)> {
    let width = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let height = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    if width <= 0 || height <= 0 {
        Err(io::Error::other(format!(
            "invalid primary screen metrics: {width}x{height}"
        )))
    } else {
        Ok((0, 0, width, height))
    }
}

fn resolve_capture_rect(region: Option<(i32, i32, i32, i32)>) -> io::Result<(i32, i32, i32, i32)> {
    let rect = match region {
        Some(rect) => rect,
        None => default_capture_rect()?,
    };
    if rect.2 <= 0 || rect.3 <= 0 {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "capture dimensions must be positive, got {}x{}",
                rect.2, rect.3
            ),
        ))
    } else {
        Ok(rect)
    }
}

#[pyfunction(signature = (region=None))]
fn capture_screen_gdi(py: Python<'_>, region: Option<(i32, i32, i32, i32)>) -> PyResult<PyObject> {
    let (left, top, width, height) = resolve_capture_rect(region).map_err(to_py_os_error)?;
    let buffer = py
        .allow_threads(move || capture::capture_bgra(left, top, width, height))
        .map_err(to_py_os_error)?;
    Ok(PyBytes::new(py, &buffer).into())
}

#[pyfunction(signature = (needle_bytes, needle_w, needle_h, confidence, region=None))]
fn locate_on_screen_rust(
    py: Python<'_>,
    needle_bytes: &[u8],
    needle_w: usize,
    needle_h: usize,
    confidence: f32,
    region: Option<(i32, i32, i32, i32)>,
) -> PyResult<Option<(i32, i32, i32, i32)>> {
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(PyValueError::new_err(
            "confidence must be a finite value between 0.0 and 1.0",
        ));
    }
    let needle_len = needle_w
        .checked_mul(needle_h)
        .ok_or_else(|| PyOverflowError::new_err("template dimensions are too large"))?;
    if needle_len == 0 {
        return Err(PyValueError::new_err(
            "template width and height must be positive",
        ));
    }
    if needle_bytes.len() < needle_len {
        return Err(PyValueError::new_err(format!(
            "template buffer has {} bytes but {needle_len} are required",
            needle_bytes.len()
        )));
    }

    let (left, top, width, height) = resolve_capture_rect(region).map_err(to_py_os_error)?;
    if needle_w > width as usize || needle_h > height as usize {
        return Ok(None);
    }
    let needle = needle_bytes[..needle_len].to_vec();
    let found = py
        .allow_threads(move || -> io::Result<Option<vision::TemplateMatch>> {
            let bgra = capture::capture_bgra(left, top, width, height)?;
            let grayscale = vision::bgra_to_gray(&bgra, width as usize, height as usize)
                .ok_or_else(|| io::Error::other("captured BGRA buffer has an invalid length"))?;
            Ok(vision::find_best_hierarchical(
                &grayscale,
                width as usize,
                height as usize,
                &needle,
                needle_w,
                needle_h,
                confidence,
            ))
        })
        .map_err(to_py_os_error)?;

    found
        .map(|matched| {
            let match_x = i32::try_from(matched.x)
                .map_err(|_| PyOverflowError::new_err("matched x coordinate overflowed i32"))?;
            let match_y = i32::try_from(matched.y)
                .map_err(|_| PyOverflowError::new_err("matched y coordinate overflowed i32"))?;
            let result_x = left
                .checked_add(match_x)
                .ok_or_else(|| PyOverflowError::new_err("absolute x coordinate overflowed i32"))?;
            let result_y = top
                .checked_add(match_y)
                .ok_or_else(|| PyOverflowError::new_err("absolute y coordinate overflowed i32"))?;
            let result_width = i32::try_from(needle_w)
                .map_err(|_| PyOverflowError::new_err("template width overflowed i32"))?;
            let result_height = i32::try_from(needle_h)
                .map_err(|_| PyOverflowError::new_err("template height overflowed i32"))?;
            Ok((result_x, result_y, result_width, result_height))
        })
        .transpose()
}

#[pyfunction(signature = (variants, confidence, region=None))]
fn locate_variants_on_screen_rust(
    py: Python<'_>,
    variants: Vec<(Vec<u8>, usize, usize)>,
    confidence: f32,
    region: Option<(i32, i32, i32, i32)>,
) -> PyResult<Option<LocatedVariant>> {
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(PyValueError::new_err(
            "confidence must be a finite value between 0.0 and 1.0",
        ));
    }
    if variants.is_empty() {
        return Err(PyValueError::new_err(
            "at least one reference variant is required",
        ));
    }
    for (index, (pixels, width, height)) in variants.iter().enumerate() {
        let required = width
            .checked_mul(*height)
            .ok_or_else(|| PyOverflowError::new_err("template dimensions are too large"))?;
        if required == 0 || pixels.len() < required {
            return Err(PyValueError::new_err(format!(
                "variant {index} has invalid dimensions or a short pixel buffer"
            )));
        }
    }

    let (left, top, width, height) = resolve_capture_rect(region).map_err(to_py_os_error)?;
    let found = py
        .allow_threads(move || -> io::Result<Option<vision::VariantMatch>> {
            let bgra = capture::capture_bgra(left, top, width, height)?;
            let grayscale = vision::bgra_to_gray(&bgra, width as usize, height as usize)
                .ok_or_else(|| io::Error::other("captured BGRA buffer has an invalid length"))?;
            let borrowed = variants
                .iter()
                .map(|(pixels, variant_width, variant_height)| {
                    (&pixels[..], *variant_width, *variant_height)
                })
                .collect::<Vec<_>>();
            Ok(vision::find_best_variant(
                &grayscale,
                width as usize,
                height as usize,
                &borrowed,
                confidence,
            ))
        })
        .map_err(to_py_os_error)?;

    found
        .map(|candidate| {
            let x =
                left.checked_add(i32::try_from(candidate.matched.x).map_err(|_| {
                    PyOverflowError::new_err("matched x coordinate overflowed i32")
                })?)
                .ok_or_else(|| PyOverflowError::new_err("absolute x coordinate overflowed i32"))?;
            let y =
                top.checked_add(i32::try_from(candidate.matched.y).map_err(|_| {
                    PyOverflowError::new_err("matched y coordinate overflowed i32")
                })?)
                .ok_or_else(|| PyOverflowError::new_err("absolute y coordinate overflowed i32"))?;
            let result_width = i32::try_from(candidate.width)
                .map_err(|_| PyOverflowError::new_err("template width overflowed i32"))?;
            let result_height = i32::try_from(candidate.height)
                .map_err(|_| PyOverflowError::new_err("template height overflowed i32"))?;
            Ok((
                x,
                y,
                result_width,
                result_height,
                candidate.matched.confidence,
                candidate.variant_index,
            ))
        })
        .transpose()
}

#[pyfunction]
fn start_failsafe_hook() -> PyResult<()> {
    hook::start().map_err(to_py_os_error)
}

#[pyfunction]
fn check_failsafe_triggered() -> PyResult<bool> {
    Ok(hook::triggered())
}

#[pyfunction]
fn reset_failsafe_triggered() -> PyResult<()> {
    hook::reset();
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
    module.add_function(wrap_pyfunction!(locate_variants_on_screen_rust, module)?)?;
    Ok(())
}
