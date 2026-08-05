use std::cell::RefCell;
use std::ffi::c_void;
use std::io;
use std::ptr;
use std::slice;

use windows_sys::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GetDC, ReleaseDC,
    SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, CAPTUREBLT, DIB_RGB_COLORS, HBITMAP, HDC,
    HGDIOBJ, SRCCOPY,
};

thread_local! {
    static CAPTURE_CONTEXT: RefCell<Option<CaptureContext>> = const { RefCell::new(None) };
}

struct CaptureContext {
    screen_dc: HDC,
    memory_dc: HDC,
    bitmap: HBITMAP,
    previous_object: HGDIOBJ,
    bits: *mut u8,
    width: i32,
    height: i32,
    buffer_len: usize,
}

impl CaptureContext {
    fn new(width: i32, height: i32) -> io::Result<Self> {
        let buffer_len = checked_buffer_len(width, height)?;
        let screen_dc = unsafe { GetDC(0) };
        if screen_dc == 0 {
            return Err(last_error("GetDC"));
        }

        let memory_dc = unsafe { CreateCompatibleDC(screen_dc) };
        if memory_dc == 0 {
            unsafe { ReleaseDC(0, screen_dc) };
            return Err(last_error("CreateCompatibleDC"));
        }

        let mut bitmap_info = unsafe { std::mem::zeroed::<BITMAPINFO>() };
        bitmap_info.bmiHeader = BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width,
            biHeight: -height, // top-down DIB; no post-capture row reversal
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB,
            biSizeImage: u32::try_from(buffer_len).unwrap_or(0),
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        };
        let mut raw_bits: *mut c_void = ptr::null_mut();
        let bitmap = unsafe {
            CreateDIBSection(screen_dc, &bitmap_info, DIB_RGB_COLORS, &mut raw_bits, 0, 0)
        };
        if bitmap == 0 || raw_bits.is_null() {
            unsafe {
                if bitmap != 0 {
                    DeleteObject(bitmap);
                }
                DeleteDC(memory_dc);
                ReleaseDC(0, screen_dc);
            }
            return Err(last_error("CreateDIBSection"));
        }

        let previous_object = unsafe { SelectObject(memory_dc, bitmap) };
        if previous_object == 0 {
            unsafe {
                DeleteObject(bitmap);
                DeleteDC(memory_dc);
                ReleaseDC(0, screen_dc);
            }
            return Err(last_error("SelectObject"));
        }

        Ok(Self {
            screen_dc,
            memory_dc,
            bitmap,
            previous_object,
            bits: raw_bits.cast(),
            width,
            height,
            buffer_len,
        })
    }

    fn capture(&mut self, left: i32, top: i32) -> io::Result<Vec<u8>> {
        let raster_operation = SRCCOPY | CAPTUREBLT;
        let mut copied = unsafe {
            BitBlt(
                self.memory_dc,
                0,
                0,
                self.width,
                self.height,
                self.screen_dc,
                left,
                top,
                raster_operation,
            )
        };
        // Some display drivers reject CAPTUREBLT. Retrying SRCCOPY preserves a
        // broad compatibility path without recreating any GDI resources.
        if copied == 0 {
            copied = unsafe {
                BitBlt(
                    self.memory_dc,
                    0,
                    0,
                    self.width,
                    self.height,
                    self.screen_dc,
                    left,
                    top,
                    SRCCOPY,
                )
            };
        }
        if copied == 0 {
            return Err(last_error("BitBlt"));
        }

        let pixels = unsafe { slice::from_raw_parts(self.bits, self.buffer_len) };
        Ok(pixels.to_vec())
    }
}

impl Drop for CaptureContext {
    fn drop(&mut self) {
        unsafe {
            SelectObject(self.memory_dc, self.previous_object);
            DeleteObject(self.bitmap);
            DeleteDC(self.memory_dc);
            ReleaseDC(0, self.screen_dc);
        }
    }
}

fn checked_buffer_len(width: i32, height: i32) -> io::Result<usize> {
    if width <= 0 || height <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("capture dimensions must be positive, got {width}x{height}"),
        ));
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "capture buffer is too large"))
}

fn last_error(operation: &str) -> io::Error {
    let source = io::Error::last_os_error();
    io::Error::new(
        source.kind(),
        format!("{operation} failed (Windows error: {source})"),
    )
}

pub(crate) fn capture_bgra(left: i32, top: i32, width: i32, height: i32) -> io::Result<Vec<u8>> {
    let expected_len = checked_buffer_len(width, height)?;
    CAPTURE_CONTEXT.with(|cell| {
        let mut context = cell.try_borrow_mut().map_err(|_| {
            io::Error::other("screen capture was re-entered on the same operating-system thread")
        })?;
        let needs_rebuild = context
            .as_ref()
            .map(|existing| {
                existing.width != width
                    || existing.height != height
                    || existing.buffer_len != expected_len
            })
            .unwrap_or(true);
        if needs_rebuild {
            *context = Some(CaptureContext::new(width, height)?);
        }
        context
            .as_mut()
            .expect("capture context initialized above")
            .capture(left, top)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_capture_buffer_sizes_without_overflow() {
        assert_eq!(checked_buffer_len(1920, 1080).unwrap(), 1920 * 1080 * 4);
        assert!(checked_buffer_len(0, 10).is_err());
        assert!(checked_buffer_len(10, -1).is_err());
        if usize::BITS == 32 {
            assert!(checked_buffer_len(i32::MAX, i32::MAX).is_err());
        }
    }
}
