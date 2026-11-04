//! A viewer ahead of the game for the CPU and the GPU. Each headset refresh the viewer copies the
//! newest pair into the headset's images and hands them in before the runtime's deadline; on the
//! same GPU as the game, its copy waits behind whatever the game has queued, and a copy that waits
//! too long misses the refresh (the runtime then shows every second one for a moment: a stutter
//! however steady the game's frames). So the viewer's process, its frame thread and its device ask
//! to be scheduled before the game's. Each step says whether Windows allowed it (the highest GPU
//! classes need an administrator).

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::ID3D11Device;
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread, HIGH_PRIORITY_CLASS, SetPriorityClass, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
use windows::core::{Interface, s};

/// `D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH`: above every normal process's GPU work, below realtime.
const GPU_CLASS_HIGH: i32 = 4;
/// `D3DKMT_SCHEDULINGPRIORITYCLASS_BELOW_NORMAL`: the game's own class while a viewer shows it.
const GPU_CLASS_BELOW_NORMAL: i32 = 1;
/// The most a device's GPU thread priority goes (-7 to 7).
const GPU_THREAD_MOST: i32 = 7;

type SetSchedulingClassFn = unsafe extern "system" fn(HANDLE, i32) -> i32;

/// Raises the viewer's process and the calling (frame) thread for the CPU, and the process and
/// `device` for the GPU; what was allowed, to say.
pub fn raise(device: &ID3D11Device) -> String {
    let mut said = Vec::new();
    // SAFETY: the current process's and thread's pseudo handles.
    let cpu = unsafe { SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS).is_ok() };
    said.push(format!("process {}", if cpu { "high" } else { "refused" }));
    // SAFETY: as above.
    let thread = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST).is_ok() };
    said.push(format!("frame thread {}", if thread { "highest" } else { "refused" }));
    said.push(format!("GPU class {}", gpu_class(GPU_CLASS_HIGH, "high")));
    let gpu_thread = device.cast::<IDXGIDevice>().ok().is_some_and(|dxgi| {
        // SAFETY: a call on the device's own interface.
        unsafe { dxgi.SetGPUThreadPriority(GPU_THREAD_MOST).is_ok() }
    });
    said.push(format!("device {}", if gpu_thread { "+7" } else { "refused" }));
    said.join(", ")
}

/// The game's side of [`raise`]: its process's GPU work scheduled below normal, once a process.
/// The viewer's high class alone did not keep its copy ahead: the game is the foreground window
/// (Windows favours its GPU work) and its OpenGL work goes in long stretches, so the viewer's
/// copy waited behind several of the game's frames (a refresh missed, SteamVR halving the rate).
/// Two classes apart, the viewer's few hundred microseconds go first at every switch; the game
/// still has the whole GPU whenever the viewer has nothing queued. Lowering needs no
/// administrator. What Windows said, to log.
pub fn yield_to_viewer() -> &'static str {
    static SAID: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    SAID.get_or_init(|| gpu_class(GPU_CLASS_BELOW_NORMAL, "below normal"))
}

/// The process's GPU scheduling class set to `class` (gdi32's own call, looked up: not every
/// Windows has it); `name` when allowed.
fn gpu_class(class: i32, name: &'static str) -> &'static str {
    // SAFETY: a system library by name, and an export of it by name.
    let call = unsafe { LoadLibraryA(s!("gdi32.dll")).ok().and_then(|gdi| GetProcAddress(gdi, s!("D3DKMTSetProcessSchedulingPriorityClass"))) };
    let Some(call) = call else { return "not offered" };
    // SAFETY: the export's documented signature (process, class) -> NTSTATUS.
    let set: SetSchedulingClassFn = unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, SetSchedulingClassFn>(call) };
    // SAFETY: the current process's pseudo handle and a class the call takes.
    let status = unsafe { set(GetCurrentProcess(), class) };
    if status >= 0 { name } else { "refused" }
}
