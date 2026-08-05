use std::io;
use std::mem::size_of;

use windows_sys::Win32::Foundation::SetLastError;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    MapVirtualKeyA, SendInput, INPUT, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_SCANCODE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_MOVE,
    MOUSEEVENTF_VIRTUALDESK, MOUSEINPUT,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VirtualDesktop {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

impl VirtualDesktop {
    pub fn from_system() -> io::Result<Self> {
        let desktop = Self {
            left: unsafe { GetSystemMetrics(SM_XVIRTUALSCREEN) },
            top: unsafe { GetSystemMetrics(SM_YVIRTUALSCREEN) },
            width: unsafe { GetSystemMetrics(SM_CXVIRTUALSCREEN) },
            height: unsafe { GetSystemMetrics(SM_CYVIRTUALSCREEN) },
        };
        if desktop.width <= 0 || desktop.height <= 0 {
            Err(io::Error::other(format!(
                "invalid virtual desktop metrics: left={}, top={}, width={}, height={}",
                desktop.left, desktop.top, desktop.width, desktop.height
            )))
        } else {
            Ok(desktop)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeEvent {
    Mouse {
        x: i32,
        y: i32,
        mouse_data: u32,
        flags: u32,
    },
    Keyboard {
        virtual_key: u8,
        scan_code: u8,
        flags: u32,
    },
}

/// Normalize one physical virtual-desktop coordinate to SendInput's 0..65535
/// domain. Values outside the desktop are clamped to the nearest edge.
pub(crate) fn normalize_axis(value: i32, origin: i32, extent: i32) -> io::Result<i32> {
    if extent <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "desktop extent must be positive",
        ));
    }
    if extent == 1 {
        return Ok(0);
    }

    let origin = origin as i64;
    let extent = extent as i64;
    let coordinate = (value as i64).clamp(origin, origin + extent - 1) - origin;
    Ok(((coordinate * 65_535 + (extent - 1) / 2) / (extent - 1)) as i32)
}

fn normalize_point(x: i32, y: i32, desktop: VirtualDesktop) -> io::Result<(i32, i32)> {
    Ok((
        normalize_axis(x, desktop.left, desktop.width)?,
        normalize_axis(y, desktop.top, desktop.height)?,
    ))
}

fn is_extended_key(virtual_key: u8) -> bool {
    matches!(
        virtual_key,
        0x03 | 0x21..=0x2E | 0x5B..=0x5D | 0x6F | 0x90 | 0xA3 | 0xA5
    )
}

fn keyboard_input(virtual_key: u8, scan_code: u8, flags: u32) -> io::Result<INPUT> {
    let resolved_scan_code = if scan_code == 0 {
        unsafe { MapVirtualKeyA(virtual_key as u32, 0) as u16 }
    } else {
        scan_code as u16
    };
    if resolved_scan_code == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("virtual key 0x{virtual_key:02X} has no scan-code mapping"),
        ));
    }

    let mut resolved_flags = flags | KEYEVENTF_SCANCODE;
    if is_extended_key(virtual_key) {
        resolved_flags |= KEYEVENTF_EXTENDEDKEY;
    }

    let mut input = unsafe { std::mem::zeroed::<INPUT>() };
    input.r#type = INPUT_KEYBOARD;
    input.Anonymous.ki = KEYBDINPUT {
        wVk: 0,
        wScan: resolved_scan_code,
        dwFlags: resolved_flags,
        time: 0,
        dwExtraInfo: 0,
    };
    Ok(input)
}

fn absolute_mouse_input(
    x: i32,
    y: i32,
    mouse_data: u32,
    flags: u32,
    desktop: VirtualDesktop,
) -> io::Result<INPUT> {
    let (normalized_x, normalized_y) = normalize_point(x, y, desktop)?;
    let mut input = unsafe { std::mem::zeroed::<INPUT>() };
    input.r#type = INPUT_MOUSE;
    input.Anonymous.mi = MOUSEINPUT {
        dx: normalized_x,
        dy: normalized_y,
        mouseData: mouse_data,
        dwFlags: flags | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        time: 0,
        dwExtraInfo: 0,
    };
    Ok(input)
}

fn relative_mouse_input(dx: i32, dy: i32) -> INPUT {
    let mut input = unsafe { std::mem::zeroed::<INPUT>() };
    input.r#type = INPUT_MOUSE;
    input.Anonymous.mi = MOUSEINPUT {
        dx,
        dy,
        mouseData: 0,
        dwFlags: MOUSEEVENTF_MOVE,
        time: 0,
        dwExtraInfo: 0,
    };
    input
}

fn send_built_inputs(inputs: &[INPUT]) -> io::Result<u32> {
    if inputs.is_empty() {
        return Ok(0);
    }
    let requested = u32::try_from(inputs.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "input batch contains more than u32::MAX events",
        )
    })?;

    let mut total_sent = 0u32;
    while total_sent < requested {
        let remaining = requested - total_sent;
        unsafe { SetLastError(0) };
        let sent = unsafe {
            SendInput(
                remaining,
                inputs.as_ptr().add(total_sent as usize),
                size_of::<INPUT>() as i32,
            )
        };
        if sent == 0 || sent > remaining {
            let source = io::Error::last_os_error();
            return Err(io::Error::new(
                if source.raw_os_error().unwrap_or(0) == 0 {
                    io::ErrorKind::Other
                } else {
                    source.kind()
                },
                format!(
                    "SendInput inserted {total_sent} of {requested} events (Windows error: {source})"
                ),
            ));
        }
        total_sent += sent;
    }
    Ok(total_sent)
}

pub(crate) fn send_absolute_mouse(x: i32, y: i32, mouse_data: u32, flags: u32) -> io::Result<()> {
    let desktop = VirtualDesktop::from_system()?;
    send_absolute_mouse_on_desktop(x, y, mouse_data, flags, desktop)
}

pub(crate) fn send_absolute_mouse_on_desktop(
    x: i32,
    y: i32,
    mouse_data: u32,
    flags: u32,
    desktop: VirtualDesktop,
) -> io::Result<()> {
    let input = absolute_mouse_input(x, y, mouse_data, flags, desktop)?;
    send_built_inputs(std::slice::from_ref(&input)).map(|_| ())
}

pub(crate) fn send_relative_mouse(dx: i32, dy: i32) -> io::Result<()> {
    let input = relative_mouse_input(dx, dy);
    send_built_inputs(std::slice::from_ref(&input)).map(|_| ())
}

pub(crate) fn send_keyboard(virtual_key: u8, scan_code: u8, flags: u32) -> io::Result<()> {
    let input = keyboard_input(virtual_key, scan_code, flags)?;
    send_built_inputs(std::slice::from_ref(&input)).map(|_| ())
}

pub(crate) fn send_events(events: &[NativeEvent]) -> io::Result<u32> {
    if events.is_empty() {
        return Ok(0);
    }

    let desktop = if events
        .iter()
        .any(|event| matches!(event, NativeEvent::Mouse { .. }))
    {
        Some(VirtualDesktop::from_system()?)
    } else {
        None
    };
    let mut inputs = Vec::with_capacity(events.len());
    for event in events {
        inputs.push(match *event {
            NativeEvent::Mouse {
                x,
                y,
                mouse_data,
                flags,
            } => absolute_mouse_input(
                x,
                y,
                mouse_data,
                flags,
                desktop.expect("mouse batches initialize desktop metrics"),
            )?,
            NativeEvent::Keyboard {
                virtual_key,
                scan_code,
                flags,
            } => keyboard_input(virtual_key, scan_code, flags)?,
        });
    }
    send_built_inputs(&inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_virtual_desktop_edges_and_negative_origins() {
        assert_eq!(normalize_axis(-1920, -1920, 3840).unwrap(), 0);
        assert_eq!(normalize_axis(1919, -1920, 3840).unwrap(), 65_535);
        assert_eq!(normalize_axis(-5000, -1920, 3840).unwrap(), 0);
        assert_eq!(normalize_axis(5000, -1920, 3840).unwrap(), 65_535);
    }

    #[test]
    fn normalizes_single_pixel_and_rejects_invalid_extents() {
        assert_eq!(normalize_axis(123, 123, 1).unwrap(), 0);
        assert!(normalize_axis(0, 0, 0).is_err());
        assert!(normalize_axis(0, 0, -1).is_err());
    }

    #[test]
    fn absolute_mouse_inputs_include_virtual_desktop_flag() {
        let desktop = VirtualDesktop {
            left: -100,
            top: -50,
            width: 200,
            height: 100,
        };
        let input = absolute_mouse_input(-100, -50, 0, MOUSEEVENTF_MOVE, desktop).unwrap();
        let mouse = unsafe { input.Anonymous.mi };
        assert_eq!((mouse.dx, mouse.dy), (0, 0));
        assert_ne!(mouse.dwFlags & MOUSEEVENTF_ABSOLUTE, 0);
        assert_ne!(mouse.dwFlags & MOUSEEVENTF_VIRTUALDESK, 0);
    }

    #[test]
    fn extended_key_table_covers_navigation_and_right_modifiers() {
        assert!(is_extended_key(0x25));
        assert!(is_extended_key(0xA3));
        assert!(is_extended_key(0x6F));
        assert!(!is_extended_key(b'A'));
    }
}
