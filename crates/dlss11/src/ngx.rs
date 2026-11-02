//! NVIDIA's NGX C API, the part DLSS on D3D11 needs (`nvsdk_ngx.h`, `nvsdk_ngx_params.h`,
//! `nvsdk_ngx_defs.h`, SDK API version 1.5); the create and evaluate helpers of
//! `nvsdk_ngx_helpers.h` are [`crate::Dlss11`]'s own.

#![allow(non_snake_case)]

use std::ffi::{c_char, c_void};

/// `NVSDK_NGX_Result`: success is 1, failures are `0xBAD0_0000` and up.
pub type NgxResult = u32;

pub fn failed(result: NgxResult) -> bool {
    result & 0xFFF0_0000 == 0xBAD0_0000
}

/// `NVSDK_NGX_Parameter` and `NVSDK_NGX_Handle`: opaque.
#[repr(C)]
pub struct Parameter {
    _private: [u8; 0],
}
#[repr(C)]
pub struct Handle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PathListInfo {
    pub path: *const *const u16,
    pub length: u32,
}

#[repr(C)]
pub struct LoggingInfo {
    pub callback: Option<unsafe extern "C" fn(*const c_char, i32, i32)>,
    pub minimum_level: i32,
    pub disable_other_sinks: bool,
}

#[repr(C)]
pub struct FeatureCommonInfo {
    pub path_list: PathListInfo,
    pub internal: *mut c_void,
    pub logging: LoggingInfo,
}

pub const VERSION_API: i32 = 0x15;
pub const ENGINE_TYPE_CUSTOM: i32 = 0;
pub const FEATURE_SUPER_SAMPLING: i32 = 1;

/// `NVSDK_NGX_PerfQuality_Value`.
pub const PERF_MAX_PERF: i32 = 0;
pub const PERF_BALANCED: i32 = 1;
pub const PERF_MAX_QUALITY: i32 = 2;
pub const PERF_ULTRA_PERFORMANCE: i32 = 3;
pub const PERF_DLAA: i32 = 5;

/// `NVSDK_NGX_DLSS_Feature_Flags`: motion vectors at the render size.
pub const DLSS_MV_LOW_RES: i32 = 1 << 1;

unsafe extern "C" {
    pub fn NVSDK_NGX_D3D11_Init_with_ProjectID(
        project_id: *const c_char,
        engine_type: i32,
        engine_version: *const c_char,
        data_path: *const u16,
        device: *mut c_void,
        info: *const FeatureCommonInfo,
        version: i32,
    ) -> NgxResult;
    pub fn NVSDK_NGX_D3D11_Shutdown1(device: *mut c_void) -> NgxResult;
    pub fn NVSDK_NGX_D3D11_GetCapabilityParameters(out: *mut *mut Parameter) -> NgxResult;
    pub fn NVSDK_NGX_D3D11_DestroyParameters(parameters: *mut Parameter) -> NgxResult;
    pub fn NVSDK_NGX_D3D11_CreateFeature(context: *mut c_void, feature: i32, parameters: *mut Parameter, out: *mut *mut Handle) -> NgxResult;
    pub fn NVSDK_NGX_D3D11_EvaluateFeature_C(context: *mut c_void, handle: *const Handle, parameters: *const Parameter, callback: *const c_void) -> NgxResult;
    pub fn NVSDK_NGX_D3D11_ReleaseFeature(handle: *mut Handle) -> NgxResult;

    pub fn NVSDK_NGX_Parameter_SetUI(parameters: *mut Parameter, name: *const c_char, value: u32);
    pub fn NVSDK_NGX_Parameter_SetI(parameters: *mut Parameter, name: *const c_char, value: i32);
    pub fn NVSDK_NGX_Parameter_SetF(parameters: *mut Parameter, name: *const c_char, value: f32);
    pub fn NVSDK_NGX_Parameter_SetD3d11Resource(parameters: *mut Parameter, name: *const c_char, value: *mut c_void);
    pub fn NVSDK_NGX_Parameter_GetI(parameters: *mut Parameter, name: *const c_char, out: *mut i32) -> NgxResult;
}
