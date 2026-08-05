use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, OnceLock};

use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageA, GetMessageA, GetSystemMetrics, SetWindowsHookExA,
    TranslateMessage, UnhookWindowsHookEx, MSG, MSLLHOOKSTRUCT, SM_CXVIRTUALSCREEN,
    SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, WH_MOUSE_LL,
};

static FAILSAFE_TRIGGERED: AtomicBool = AtomicBool::new(false);
static HOOK_START_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

unsafe extern "system" fn low_level_mouse_proc(code: i32, wparam: usize, lparam: isize) -> isize {
    if code >= 0 && lparam != 0 {
        let mouse = &*(lparam as *const MSLLHOOKSTRUCT);
        let left = GetSystemMetrics(SM_XVIRTUALSCREEN) as i64;
        let top = GetSystemMetrics(SM_YVIRTUALSCREEN) as i64;
        let width = GetSystemMetrics(SM_CXVIRTUALSCREEN) as i64;
        let height = GetSystemMetrics(SM_CYVIRTUALSCREEN) as i64;
        if width > 0 && height > 0 {
            let right = left + width - 1;
            let bottom = top + height - 1;
            let x = mouse.pt.x as i64;
            let y = mouse.pt.y as i64;
            if (x == left || x == right) && (y == top || y == bottom) {
                FAILSAFE_TRIGGERED.store(true, Ordering::Release);
            }
        }
    }
    CallNextHookEx(0, code, wparam, lparam)
}

fn initialize_hook_thread() -> Result<(), String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("pyautogui-failsafe-hook".to_owned())
        .spawn(move || unsafe {
            let hook = SetWindowsHookExA(WH_MOUSE_LL, Some(low_level_mouse_proc), 0, 0);
            if hook == 0 {
                let error = io::Error::last_os_error();
                let _ = sender.send(Err(format!(
                    "SetWindowsHookExA failed (Windows error: {error})"
                )));
                return;
            }
            if sender.send(Ok(())).is_err() {
                UnhookWindowsHookEx(hook);
                return;
            }

            let mut message = std::mem::zeroed::<MSG>();
            while GetMessageA(&mut message, 0, 0, 0) > 0 {
                TranslateMessage(&message);
                DispatchMessageA(&message);
            }
            UnhookWindowsHookEx(hook);
        })
        .map_err(|error| format!("failed to spawn failsafe hook thread: {error}"))?;

    receiver
        .recv()
        .map_err(|error| format!("failsafe hook thread exited during startup: {error}"))?
}

pub(crate) fn start() -> io::Result<()> {
    match HOOK_START_RESULT.get_or_init(initialize_hook_thread) {
        Ok(()) => Ok(()),
        Err(message) => Err(io::Error::other(message.clone())),
    }
}

pub(crate) fn triggered() -> bool {
    FAILSAFE_TRIGGERED.load(Ordering::Acquire)
}

pub(crate) fn reset() {
    FAILSAFE_TRIGGERED.store(false, Ordering::Release);
}
