//! NVIDIA DLSS on a D3D11 device of the loop's own, for a game that renders both eyes each frame
//! smaller than the eye size and leaves the upscaling to Monaka:
//!
//! 1. each eye's part of the captured frame and of its depth is copied into render-size textures;
//! 2. its motion vectors are written from that depth and the eye's camera now and at its previous
//!    frame ([`monaka_warp::MotionPass`]: the camera's motion only);
//!    With them, a mask of where pixels moved on their own (an enemy walking, the gun in a moving
//!    hand: what lands on nothing that far away in the eye's previous depth), which DLSS takes from
//!    the current frame rather than its history ("bias current colour"): with the camera's motion
//!    alone, its history would ghost around them;
//! 3. the eye's own DLSS feature (one history per eye) upscales it to the eye size;
//! 4. with [`Settings::sharpness`], AMD's CAS sharpens it ([`monaka_warp::SharpenPass`]), as DLSS's
//!    own sharpening no longer does.
//!
//! The projection each eye is drawn with is jittered by [`jitter`] (the same sequence for both
//! eyes); the frame's [`EyeFrame::jitter`] says by how much. NGX is NVIDIA's runtime in the driver;
//! the DLSS model itself (`nvngx_dlss.dll`) is looked for in [`Settings::dll_dirs`], then beside
//! the game's executable.

mod ngx;

use monaka_channel::d3d;
use monaka_warp::{EyeView, MotionPass, Moved, SharpenPass};
use std::ffi::{CStr, c_void};
use std::path::PathBuf;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_BOX, ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    ID3D11UnorderedAccessView,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R16G16_FLOAT,
    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R32_FLOAT,
};
use windows::core::Interface;

/// The project id NGX is opened with (a custom engine has no NVIDIA application id).
const PROJECT_ID: &CStr = c"8f4d6c2a-31b7-4e0d-9a5c-6e2b1f7d40a9";

/// How much smaller than the eye size each eye is rendered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    /// The eye size itself: anti-aliasing only.
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl Quality {
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.trim() {
            "dlaa" => Self::Dlaa,
            "quality" => Self::Quality,
            "balanced" => Self::Balanced,
            "performance" => Self::Performance,
            "ultra_performance" => Self::UltraPerformance,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Dlaa => "DLAA",
            Self::Quality => "Quality",
            Self::Balanced => "Balanced",
            Self::Performance => "Performance",
            Self::UltraPerformance => "Ultra Performance",
        }
    }

    /// The render size over the eye size, per side (NVIDIA's ratios).
    pub fn scale(self) -> f32 {
        match self {
            Self::Dlaa => 1.0,
            Self::Quality => 2.0 / 3.0,
            Self::Balanced => 0.58,
            Self::Performance => 0.5,
            Self::UltraPerformance => 1.0 / 3.0,
        }
    }

    fn value(self) -> i32 {
        match self {
            Self::Dlaa => ngx::PERF_DLAA,
            Self::Quality => ngx::PERF_MAX_QUALITY,
            Self::Balanced => ngx::PERF_BALANCED,
            Self::Performance => ngx::PERF_MAX_PERF,
            Self::UltraPerformance => ngx::PERF_ULTRA_PERFORMANCE,
        }
    }
}

/// The size each eye is rendered at for an eye of `display` at `quality` (even sides).
pub fn render_size(display: (u32, u32), quality: Quality) -> (u32, u32) {
    let side = |n: u32| ((n as f32 * quality.scale() / 2.0).round() as u32 * 2).max(2);
    (side(display.0), side(display.1))
}

/// The jitter phases for `quality`: 8 times the pixel ratio, 8 to 72.
pub fn phases(quality: Quality) -> u32 {
    let ratio = 1.0 / quality.scale();
    ((8.0 * ratio * ratio).ceil() as u32).clamp(8, 72)
}

/// The jitter at `phase` (render pixels, x right, y down): the Halton (2, 3) point, centred.
pub fn jitter(phase: u32) -> [f32; 2] {
    let halton = |mut index: u32, base: u32| {
        let (mut fraction, mut result) = (1.0f32, 0.0f32);
        while index > 0 {
            fraction /= base as f32;
            result += fraction * (index % base) as f32;
            index /= base;
        }
        result
    };
    [halton(phase + 1, 2) - 0.5, halton(phase + 1, 3) - 0.5]
}

#[derive(Debug)]
pub enum Error {
    /// NGX or DLSS is not available here (no NVIDIA RTX GPU, an old driver, no model DLL).
    Unavailable(String),
    Failed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(why) => write!(f, "DLSS unavailable: {why}"),
            Self::Failed(why) => write!(f, "DLSS: {why}"),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

fn ngx_call(what: &str, result: ngx::NgxResult) -> Result<()> {
    if ngx::failed(result) { Err(Error::Failed(format!("{what}: {result:#x}"))) } else { Ok(()) }
}

fn d3d_call<T>(what: &str, result: windows::core::Result<T>) -> Result<T> {
    result.map_err(|e| Error::Failed(format!("{what}: {e}")))
}

/// Settings for [`Dlss11::new`].
#[derive(Clone, Debug)]
pub struct Settings {
    /// The size each eye is rendered at.
    pub render: (u32, u32),
    /// The size each eye is upscaled to.
    pub display: (u32, u32),
    pub quality: Quality,
    /// A folder NGX may write its logs and cache into.
    pub data_dir: PathBuf,
    /// Where to look for `nvngx_dlss.dll`, first to last.
    pub dll_dirs: Vec<PathBuf>,
    /// 1 or -1: whether DLSS is told the content shift or its opposite.
    pub jitter_sign: f32,
    /// CAS sharpening 0..1 after DLSS, or none.
    pub sharpness: Option<f32>,
}

/// One eye's frame.
pub struct EyeFrame<'a> {
    /// The finished image holding the eye (RGBA8 or BGRA8, any of the family) and the eye's part of
    /// it (render size).
    pub color: &'a ID3D11Texture2D,
    pub color_box: D3D11_BOX,
    /// The window depth (R32, 0..1) the eye was drawn with, and the eye's part of it (render size).
    pub depth: &'a ID3D11Texture2D,
    pub depth_box: D3D11_BOX,
    /// The camera and the unjittered projection the depth was rendered with.
    pub view: EyeView,
    /// The pixel offset the projection was jittered by (x right, y down), from [`jitter`].
    pub jitter: [f32; 2],
    pub frame_time_ms: f32,
    /// Drop the eye's history (a cut).
    pub reset: bool,
}

struct Eye {
    handle: *mut ngx::Handle,
    output: (ID3D11Texture2D, ID3D11UnorderedAccessView),
    /// With sharpening: what DLSS writes, and a view the sharpening reads it by.
    upscaled: Option<(ID3D11Texture2D, ID3D11ShaderResourceView)>,
    previous: Option<EyeView>,
    /// The eye's depth as last written (render size), for the moved mask.
    previous_depth: (ID3D11Texture2D, ID3D11ShaderResourceView),
}

pub struct Dlss11 {
    device: ID3D11Device,
    parameters: *mut ngx::Parameter,
    motion: MotionPass,
    sharpen: Option<SharpenPass>,
    settings: Settings,
    /// The render-size inputs, shared by both eyes (written and read in turn on one context).
    color: Option<(DXGI_FORMAT, ID3D11Texture2D)>,
    depth_in: ID3D11Texture2D,
    depth: (ID3D11Texture2D, ID3D11UnorderedAccessView),
    motion_vectors: (ID3D11Texture2D, ID3D11UnorderedAccessView),
    /// Where pixels moved on their own (render size, shared by both eyes like the inputs).
    moved: (ID3D11Texture2D, ID3D11UnorderedAccessView),
    eyes: Vec<Eye>,
}

// SAFETY: used under its owner's lock on the game's render thread.
unsafe impl Send for Dlss11 {}

fn texture(device: &ID3D11Device, (width, height): (u32, u32), format: DXGI_FORMAT, bind: u32) -> Result<ID3D11Texture2D> {
    let desc = windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC { BindFlags: bind, ..d3d::simple_texture_desc(width, height, format) };
    let mut made = None;
    // SAFETY: creates a texture of our own on our own device.
    unsafe { d3d_call("texture", device.CreateTexture2D(&desc, None, Some(&mut made)))? };
    made.ok_or_else(|| Error::Failed("texture".into()))
}

fn with_uav(device: &ID3D11Device, size: (u32, u32), format: DXGI_FORMAT) -> Result<(ID3D11Texture2D, ID3D11UnorderedAccessView)> {
    let made = texture(device, size, format, (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0) as u32)?;
    let mut view = None;
    // SAFETY: a view of our own texture, made with unordered access.
    unsafe { d3d_call("unordered access view", device.CreateUnorderedAccessView(&made, None, Some(&mut view)))? };
    Ok((made, view.ok_or_else(|| Error::Failed("unordered access view".into()))?))
}

/// A texture DLSS writes and the sharpening reads, with its shader view.
fn with_srv(device: &ID3D11Device, size: (u32, u32), format: DXGI_FORMAT) -> Result<(ID3D11Texture2D, ID3D11ShaderResourceView)> {
    let made = texture(device, size, format, (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_UNORDERED_ACCESS.0) as u32)?;
    let mut view = None;
    // SAFETY: a view of our own texture, made with shader-resource binding.
    unsafe { d3d_call("shader view", device.CreateShaderResourceView(&made, None, Some(&mut view)))? };
    Ok((made, view.ok_or_else(|| Error::Failed("shader view".into()))?))
}

/// The plain (not sRGB) format of a colour format's family, as DLSS reads it: an sRGB back buffer's
/// bytes as stored.
fn plain(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => DXGI_FORMAT_B8G8R8A8_UNORM,
        other => other,
    }
}

fn wide(path: &std::path::Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

impl Dlss11 {
    /// Opens NGX on `device`, checks DLSS is there and makes each eye's feature.
    pub fn new(device: &ID3D11Device, context: &ID3D11DeviceContext, settings: Settings) -> Result<Self> {
        let _ = std::fs::create_dir_all(&settings.data_dir);
        let data = wide(&settings.data_dir);
        let dirs: Vec<Vec<u16>> = settings.dll_dirs.iter().map(|d| wide(d)).collect();
        let dir_pointers: Vec<*const u16> = dirs.iter().map(|d| d.as_ptr()).collect();
        let info = ngx::FeatureCommonInfo {
            path_list: ngx::PathListInfo { path: dir_pointers.as_ptr(), length: dir_pointers.len() as u32 },
            internal: std::ptr::null_mut(),
            logging: ngx::LoggingInfo { callback: None, minimum_level: 0, disable_other_sinks: false },
        };
        let raw = device.as_raw();
        // SAFETY: NGX's documented initialisation on our live device, with strings and a path list
        // that outlive the call.
        let opened = unsafe { ngx::NVSDK_NGX_D3D11_Init_with_ProjectID(PROJECT_ID.as_ptr(), ngx::ENGINE_TYPE_CUSTOM, c"1.0".as_ptr(), data.as_ptr(), raw, &info, ngx::VERSION_API) };
        if ngx::failed(opened) {
            return Err(Error::Unavailable(format!("NGX did not open ({opened:#x}): an NVIDIA RTX GPU and a recent driver are needed")));
        }
        let mut parameters = std::ptr::null_mut();
        // SAFETY: NGX is open on this device; it hands back its own parameter block.
        let got = unsafe { ngx::NVSDK_NGX_D3D11_GetCapabilityParameters(&mut parameters) };
        let shut = |why: Error| {
            // SAFETY: closes what was opened above.
            unsafe {
                if !parameters.is_null() {
                    ngx::NVSDK_NGX_D3D11_DestroyParameters(parameters);
                }
                ngx::NVSDK_NGX_D3D11_Shutdown1(raw);
            }
            why
        };
        if ngx::failed(got) || parameters.is_null() {
            return Err(shut(Error::Unavailable(format!("no NGX parameters ({got:#x})"))));
        }
        let get = |name: &CStr| {
            let mut value = 0;
            // SAFETY: a query of NGX's own parameter block.
            let result = unsafe { ngx::NVSDK_NGX_Parameter_GetI(parameters, name.as_ptr(), &mut value) };
            (!ngx::failed(result)).then_some(value)
        };
        if get(c"SuperSampling.NeedsUpdatedDriver").unwrap_or(0) != 0 {
            return Err(shut(Error::Unavailable("the NVIDIA driver is too old for DLSS".into())));
        }
        if get(c"SuperSampling.Available").unwrap_or(0) == 0 {
            return Err(shut(Error::Unavailable("DLSS is not available on this GPU, or nvngx_dlss.dll was not found".into())));
        }
        let made = Self::make(device, context, parameters, settings);
        made.map_err(shut)
    }

    fn make(device: &ID3D11Device, context: &ID3D11DeviceContext, parameters: *mut ngx::Parameter, settings: Settings) -> Result<Self> {
        let (render, display) = (settings.render, settings.display);
        let depth_in = texture(device, render, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_SHADER_RESOURCE.0 as u32)?;
        let depth = with_uav(device, render, DXGI_FORMAT_R32_FLOAT)?;
        let motion_vectors = with_uav(device, render, DXGI_FORMAT_R16G16_FLOAT)?;
        let moved = with_uav(device, render, DXGI_FORMAT_R8_UNORM)?;
        let sharpen = match settings.sharpness {
            Some(_) => Some(SharpenPass::new(device).map_err(|e| Error::Failed(format!("sharpening: {e}")))?),
            None => None,
        };
        let motion = MotionPass::new(device).map_err(|e| Error::Failed(format!("motion pass: {e}")))?;
        let mut this = Self { device: device.clone(), parameters, motion, sharpen, settings, color: None, depth_in, depth, motion_vectors, moved, eyes: Vec::new() };
        for _ in 0..2 {
            let output = with_uav(device, display, DXGI_FORMAT_R8G8B8A8_UNORM)?;
            let upscaled = match this.sharpen {
                Some(_) => Some(with_srv(device, display, DXGI_FORMAT_R8G8B8A8_UNORM)?),
                None => None,
            };
            let handle = this.create_feature(context)?;
            let previous_depth = with_srv(device, render, DXGI_FORMAT_R32_FLOAT)?;
            this.eyes.push(Eye { handle, output, upscaled, previous: None, previous_depth });
        }
        Ok(this)
    }

    fn create_feature(&self, context: &ID3D11DeviceContext) -> Result<*mut ngx::Handle> {
        let (p, s) = (self.parameters, &self.settings);
        let mut handle = std::ptr::null_mut();
        // SAFETY: NGX_D3D11_CREATE_DLSS_EXT's own sequence on NGX's parameter block and our context.
        unsafe {
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"CreationNodeMask".as_ptr(), 1);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"VisibilityNodeMask".as_ptr(), 1);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"Width".as_ptr(), s.render.0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"Height".as_ptr(), s.render.1);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"OutWidth".as_ptr(), s.display.0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"OutHeight".as_ptr(), s.display.1);
            ngx::NVSDK_NGX_Parameter_SetI(p, c"PerfQualityValue".as_ptr(), s.quality.value());
            ngx::NVSDK_NGX_Parameter_SetI(p, c"DLSS.Feature.Create.Flags".as_ptr(), ngx::DLSS_MV_LOW_RES);
            ngx::NVSDK_NGX_Parameter_SetI(p, c"DLSS.Enable.Output.Subrects".as_ptr(), 0);
            ngx_call("creating the DLSS feature", ngx::NVSDK_NGX_D3D11_CreateFeature(context.as_raw(), ngx::FEATURE_SUPER_SAMPLING, p, &mut handle))?;
        }
        if handle.is_null() { Err(Error::Failed("no DLSS feature".into())) } else { Ok(handle) }
    }

    pub fn render(&self) -> (u32, u32) {
        self.settings.render
    }

    pub fn display(&self) -> (u32, u32) {
        self.settings.display
    }

    /// Upscales eye `eye`'s frame on `context` (our device's); returns the eye's output (display
    /// size, RGBA8 UNORM), ready for anything later on that context.
    pub fn upscale(&mut self, context: &ID3D11DeviceContext, eye: usize, frame: &EyeFrame<'_>) -> Result<ID3D11Texture2D> {
        let render = self.settings.render;
        let size = |b: &D3D11_BOX| (b.right - b.left, b.bottom - b.top);
        if size(&frame.color_box) != render || size(&frame.depth_box) != render {
            return Err(Error::Failed(format!("a {:?} eye for a {}x{} upscaler", size(&frame.color_box), render.0, render.1)));
        }
        let format = plain(d3d::texture_desc(frame.color).Format);
        if self.color.as_ref().is_none_or(|(made, _)| *made != format) {
            self.color = Some((format, texture(&self.device, render, format, D3D11_BIND_SHADER_RESOURCE.0 as u32)?));
        }
        let color = &self.color.as_ref().expect("made above").1;
        // SAFETY: copies between textures of one family on our own context.
        unsafe {
            context.CopySubresourceRegion(color, 0, 0, 0, 0, frame.color, 0, Some(&frame.color_box));
            context.CopySubresourceRegion(&self.depth_in, 0, 0, 0, 0, frame.depth, 0, Some(&frame.depth_box));
        }
        let state = &mut self.eyes[eye];
        let previous = if frame.reset { None } else { state.previous };
        let moved = Moved { previous_depth: previous.map(|_| &state.previous_depth.1), mask: &self.moved.1 };
        self.motion
            .run_moved(context, &self.depth_in, &frame.view, previous.as_ref(), None, &self.motion_vectors.1, &self.depth.1, Some(moved), render)
            .map_err(|e| Error::Failed(format!("motion vectors: {e}")))?;
        // SAFETY: copies between two of our own same-shaped R32 textures on our context.
        unsafe { context.CopyResource(&state.previous_depth.0, &self.depth.0) };
        let p = self.parameters;
        let sign = self.settings.jitter_sign;
        let resource = |t: &ID3D11Texture2D| t.as_raw() as *mut c_void;
        // SAFETY: NGX_D3D11_EVALUATE_DLSS_EXT's own sequence, with our live textures, on our context.
        unsafe {
            ngx::NVSDK_NGX_Parameter_SetD3d11Resource(p, c"Color".as_ptr(), resource(color));
            let written = state.upscaled.as_ref().map_or(&state.output.0, |(t, _)| t);
            ngx::NVSDK_NGX_Parameter_SetD3d11Resource(p, c"Output".as_ptr(), resource(written));
            ngx::NVSDK_NGX_Parameter_SetD3d11Resource(p, c"Depth".as_ptr(), resource(&self.depth.0));
            ngx::NVSDK_NGX_Parameter_SetD3d11Resource(p, c"MotionVectors".as_ptr(), resource(&self.motion_vectors.0));
            ngx::NVSDK_NGX_Parameter_SetD3d11Resource(p, c"DLSS.Input.Bias.Current.Color.Mask".as_ptr(), resource(&self.moved.0));
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.Bias.Current.Color.SubrectBase.X".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.Bias.Current.Color.SubrectBase.Y".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"Jitter.Offset.X".as_ptr(), sign * frame.jitter[0]);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"Jitter.Offset.Y".as_ptr(), sign * frame.jitter[1]);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"Sharpness".as_ptr(), 0.0);
            ngx::NVSDK_NGX_Parameter_SetI(p, c"Reset".as_ptr(), i32::from(previous.is_none()));
            ngx::NVSDK_NGX_Parameter_SetF(p, c"MV.Scale.X".as_ptr(), render.0 as f32);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"MV.Scale.Y".as_ptr(), render.1 as f32);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.Color.Subrect.Base.X".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.Color.Subrect.Base.Y".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.Depth.Subrect.Base.X".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.Depth.Subrect.Base.Y".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.MV.SubrectBase.X".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Input.MV.SubrectBase.Y".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Output.Subrect.Base.X".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Output.Subrect.Base.Y".as_ptr(), 0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Render.Subrect.Dimensions.Width".as_ptr(), render.0);
            ngx::NVSDK_NGX_Parameter_SetUI(p, c"DLSS.Render.Subrect.Dimensions.Height".as_ptr(), render.1);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"DLSS.Pre.Exposure".as_ptr(), 1.0);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"DLSS.Exposure.Scale".as_ptr(), 1.0);
            ngx::NVSDK_NGX_Parameter_SetF(p, c"FrameTimeDeltaInMsec".as_ptr(), frame.frame_time_ms);
            ngx_call("evaluating DLSS", ngx::NVSDK_NGX_D3D11_EvaluateFeature_C(context.as_raw(), state.handle, p, std::ptr::null()))?;
        }
        if let (Some(sharpen), Some((_, view)), Some(sharpness)) = (&self.sharpen, &state.upscaled, self.settings.sharpness) {
            sharpen.run(context, view, &state.output.1, self.settings.display, sharpness);
        }
        state.previous = Some(frame.view);
        Ok(state.output.0.clone())
    }
}

impl Drop for Dlss11 {
    fn drop(&mut self) {
        // SAFETY: releases what `new` made, on the device it was made on.
        unsafe {
            for eye in &self.eyes {
                ngx::NVSDK_NGX_D3D11_ReleaseFeature(eye.handle);
            }
            ngx::NVSDK_NGX_D3D11_DestroyParameters(self.parameters);
            ngx::NVSDK_NGX_D3D11_Shutdown1(self.device.as_raw());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_renders_two_thirds_of_each_side_at_even_sizes() {
        assert_eq!(render_size((2688, 2688), Quality::Quality), (1792, 1792));
        assert_eq!(render_size((2688, 2688), Quality::Dlaa), (2688, 2688));
        assert_eq!(render_size((2688, 2688), Quality::Performance), (1344, 1344));
        let (w, h) = render_size((2017, 1999), Quality::Balanced);
        assert!(w % 2 == 0 && h % 2 == 0);
    }

    #[test]
    fn jitter_stays_within_a_pixel_and_phases_follow_the_ratio() {
        assert!((0..72).map(jitter).all(|[x, y]| x.abs() <= 0.5 && y.abs() <= 0.5));
        assert_eq!(phases(Quality::Dlaa), 8);
        assert_eq!(phases(Quality::Quality), 18);
        assert_eq!(phases(Quality::UltraPerformance), 72);
    }

    /// On an NVIDIA RTX GPU, with `MONAKA_DLSS_DLL_DIR` naming a folder holding `nvngx_dlss.dll`:
    /// a grey frame upscales to a grey eye, twice per eye.
    #[test]
    #[ignore = "needs an NVIDIA RTX GPU and MONAKA_DLSS_DLL_DIR"]
    fn a_grey_frame_upscales_grey_on_this_gpu() {
        use windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC;
        let dir = PathBuf::from(std::env::var("MONAKA_DLSS_DLL_DIR").expect("MONAKA_DLSS_DLL_DIR"));
        let (device, context, _) = d3d::create_device(None).expect("device");
        let display = (512, 512);
        let render = render_size(display, Quality::Quality);
        let settings = Settings { render, display, quality: Quality::Quality, data_dir: std::env::temp_dir().join("monaka_dlss_test"), dll_dirs: vec![dir], jitter_sign: 1.0, sharpness: Some(0.5) };
        let mut dlss = Dlss11::new(&device, &context, settings).expect("DLSS");
        let frame = (render.0 * 2, render.1);
        let colour_desc = D3D11_TEXTURE2D_DESC { BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32, ..d3d::simple_texture_desc(frame.0, frame.1, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) };
        let colour = d3d::create_texture_with(&device, &colour_desc, &[128u8, 128, 128, 255].repeat((frame.0 * frame.1) as usize)).expect("colour");
        let depth_desc = D3D11_TEXTURE2D_DESC { BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32, ..d3d::simple_texture_desc(frame.0, frame.1, DXGI_FORMAT_R32_FLOAT) };
        let depth = d3d::create_texture_with(&device, &depth_desc, &0.5f32.to_le_bytes().repeat((frame.0 * frame.1) as usize)).expect("depth");
        let view = EyeView { camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], projection: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, -1.0, 0.0] };
        for n in 0..4u32 {
            let eye = (n % 2) as usize;
            let part = D3D11_BOX { left: eye as u32 * render.0, top: 0, front: 0, right: (eye as u32 + 1) * render.0, bottom: render.1, back: 1 };
            let frame = EyeFrame { color: &colour, color_box: part, depth: &depth, depth_box: part, view, jitter: jitter(n), frame_time_ms: 11.0, reset: n < 2 };
            let out = dlss.upscale(&context, eye, &frame).expect("upscale");
            let desc = d3d::texture_desc(&out);
            assert_eq!((desc.Width, desc.Height), display);
            let pixel = d3d::read_pixel(&device, &context, &out, 256, 256).expect("read");
            assert!(pixel[..3].iter().all(|&c| (118..=138).contains(&c)), "{pixel:?}");
        }
    }

    #[test]
    fn options_name_each_quality() {
        for name in ["dlaa", "quality", "balanced", "performance", "ultra_performance"] {
            assert!(Quality::parse(name).is_some(), "{name}");
        }
        assert_eq!(Quality::parse("ultra"), None);
    }
}
