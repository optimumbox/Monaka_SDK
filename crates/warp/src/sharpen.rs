//! Contrast-adaptive sharpening (`sharpen.hlsl`, AMD's CAS) of an upscaled image: what DLSS's own
//! sharpening did before NVIDIA's newer models dropped it.

use crate::{ConstantBuffer, Result, SavedCompute, compute_shader};
use windows::Win32::Graphics::Direct3D11::*;
use windows::core::s;

pub const SHARPEN_SOURCE: &str = include_str!("sharpen.hlsl");

/// The layout of the HLSL `cbuffer Sharpen`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Constants {
    peak: f32,
    pad: [f32; 3],
    width: u32,
    height: u32,
    pad2: [u32; 2],
}

pub struct SharpenPass {
    shader: ID3D11ComputeShader,
    constants: ConstantBuffer<Constants>,
}

/// CAS's peak for `sharpness` 0..1: -1 / lerp(8, 5, sharpness).
fn peak(sharpness: f32) -> f32 {
    -1.0 / (8.0 + (5.0 - 8.0) * sharpness.clamp(0.0, 1.0))
}

impl SharpenPass {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        Ok(Self { shader: compute_shader(device, SHARPEN_SOURCE, "sharpen.hlsl", s!("Main"))?, constants: ConstantBuffer::new(device)? })
    }

    /// Writes `source` (a view of an RGBA image) sharpened by `sharpness` (0..1) into `target`
    /// (another image of the same `width` x `height`).
    pub fn run(&self, context: &ID3D11DeviceContext, source: &ID3D11ShaderResourceView, target: &ID3D11UnorderedAccessView, (width, height): (u32, u32), sharpness: f32) {
        let saved = SavedCompute::take(context);
        self.constants.update(context, &Constants { peak: peak(sharpness), width, height, ..Default::default() });
        self.constants.bind(context);
        // SAFETY: runs our shader on the context with our own resources; its compute bindings go
        // back after.
        unsafe {
            context.CSSetShader(&self.shader, None);
            context.CSSetShaderResources(0, Some(&[Some(source.clone())]));
            context.CSSetUnorderedAccessViews(0, 1, Some([Some(target.clone())].as_ptr()), None);
            context.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            context.CSSetUnorderedAccessViews(0, 1, Some([None].as_ptr()), None);
            context.CSSetShaderResources(0, Some(&[None]));
        }
        saved.restore(context);
    }
}

#[cfg(test)]
#[test]
fn the_peak_runs_from_an_eighth_to_a_fifth() {
    assert!((peak(0.0) + 0.125).abs() < 1e-6);
    assert!((peak(1.0) + 0.2).abs() < 1e-6);
    assert!((peak(7.0) + 0.2).abs() < 1e-6);
}
