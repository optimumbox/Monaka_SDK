//! Motion vectors for temporal upscaling (`motion.hlsl`): from a frame's depth and the camera it
//! was rendered from against the same eye's previous camera, plus the game's own object motion
//! when it writes one. The game crate gives both cameras in canonical form with the projections the
//! depth buffer really uses (unjittered).

use crate::{ConstantBuffer, Error, Readable, Result, SavedCompute, compute_shader, depth_formats, readable, shader_view};
use monaka_channel::d3d;
use monaka_core::camera::{Mat34, Mat44, compose, position, rigid_inverse};
use monaka_core::depth::reprojection;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};
use windows::core::{Interface, s};

pub const MOTION_SOURCE: &str = include_str!("motion.hlsl");

/// The layout of the HLSL `cbuffer Motion`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Constants {
    m: [f32; 16],
    width: u32,
    height: u32,
    depth_width: u32,
    depth_height: u32,
    object_scale: [f32; 2],
    has_object: u32,
    flags: u32,
    linear: [f32; 4],
    /// The depth a point the hands' reach away has, and which way is nearer (1 larger, -1 smaller;
    /// 0: no reach), then padding.
    held: [f32; 4],
    unproject: [f32; 16],
    items: [[f32; 16]; 2],
    item_at: [[f32; 4]; 2],
}

/// An item in a hand (the gun, with the hand and arm about it): where it was (canonical, metres,
/// item-to-world) this frame and the eye's previous frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldItem {
    pub now: Mat34,
    pub before: Mat34,
}

const IDENTITY34: Mat34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
const IDENTITY44: Mat44 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// The world now to `previous`'s clip space for what moved with `item`; none if singular.
fn item_reprojection(item: &HeldItem, previous: &EyeView) -> Option<Mat44> {
    let moved = compose(&item.before, &rigid_inverse(&item.now));
    reprojection(&IDENTITY44, &moved, &previous.projection, &previous.camera)
}

/// One eye's camera for a frame: canonical camera-to-world and the depth buffer's projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeView {
    pub camera: Mat34,
    pub projection: Mat44,
}

/// The game's object motion: its texture (R16G16_FLOAT, render size) and the scale to UV units.
/// An R32G32B32A32_FLOAT texture is tested object motion (`motion.hlsl`): z the particles' share
/// of the pixel, w the window depth the motion was drawn at.
pub struct ObjectMotion<'a> {
    pub texture: &'a ID3D11Texture2D,
    pub scale: [f32; 2],
}

/// Where pixels moved on their own (`motion.hlsl`'s mask): the eye's previous depth as the pass
/// wrote it (R32_FLOAT, render size; none: the eye's first frame) and the mask written (a
/// single-channel UNORM or float view, render size).
pub struct Moved<'a> {
    pub previous_depth: Option<&'a ID3D11ShaderResourceView>,
    pub mask: &'a ID3D11UnorderedAccessView,
    /// Within this many metres of the eye everything counts as moved on its own (0: none): the
    /// hands, the arms and what they hold move with the controllers, not the world, so the camera's
    /// motion is wrong for all of them, and the depth test alone misses a gun shifted across its
    /// own surface (its depth at the old place the same).
    pub held_reach: f32,
    /// The hands' items (left, right): within the reach, what is nearer each moves with it rather
    /// than the camera. None of either: all within the reach is masked.
    pub items: [Option<HeldItem>; 2],
}

/// The depth at `reach` metres before an eye drawn with `projection`, and which way is nearer
/// (`motion.hlsl`'s Held); none without a reach or a perspective projection.
fn held_depth(projection: &Mat44, reach: f32) -> [f32; 4] {
    let p = projection;
    match reach > 0.0 && p[11].is_finite() && p[11] != 0.0 {
        true => [-p[10] + p[11] / reach, p[11].signum(), 0.0, 0.0],
        false => [0.0; 4],
    }
}

pub struct MotionPass {
    device: ID3D11Device,
    shader: ID3D11ComputeShader,
    constants: ConstantBuffer<Constants>,
    depth: Option<Readable>,
    /// A view of the game's object-motion texture, kept while it is the same texture, and whether
    /// it is tested object motion.
    object: Option<(usize, ID3D11ShaderResourceView, bool)>,
}

impl MotionPass {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            shader: compute_shader(device, MOTION_SOURCE, "motion.hlsl", s!("Main"))?,
            constants: ConstantBuffer::new(device)?,
            depth: None,
            object: None,
        })
    }

    fn object_view(&mut self, texture: &ID3D11Texture2D) -> Option<(ID3D11ShaderResourceView, bool)> {
        let key = texture.as_raw() as usize;
        if let Some((known, view, tested)) = &self.object
            && *known == key
        {
            return Some((view.clone(), *tested));
        }
        let desc = d3d::texture_desc(texture);
        if desc.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
            return None;
        }
        let tested = desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT;
        // The game's live texture allows shader reads (checked above).
        let view = shader_view(&self.device, texture, if tested { DXGI_FORMAT_R32G32B32A32_FLOAT } else { DXGI_FORMAT_R16G16_FLOAT }).ok()?;
        self.object = Some((key, view.clone(), tested));
        Some((view, tested))
    }

    /// Writes this frame's motion vectors (`motion`, an R16G16_FLOAT view) and depth (`depth_out`,
    /// R32_FLOAT) at `width` x `height` from the game's `depth` texture. Without a `previous` view
    /// (the eye's first frame) the camera adds no motion.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        context: &ID3D11DeviceContext,
        depth: &ID3D11Texture2D,
        current: &EyeView,
        previous: Option<&EyeView>,
        object: Option<ObjectMotion<'_>>,
        motion: &ID3D11UnorderedAccessView,
        depth_out: &ID3D11UnorderedAccessView,
        size: (u32, u32),
    ) -> Result<()> {
        self.run_moved(context, depth, current, previous, object, motion, depth_out, None, size)
    }

    /// [`Self::run`], and with `moved` the mask of where pixels moved on their own.
    #[allow(clippy::too_many_arguments)]
    pub fn run_moved(
        &mut self,
        context: &ID3D11DeviceContext,
        depth: &ID3D11Texture2D,
        current: &EyeView,
        previous: Option<&EyeView>,
        object: Option<ObjectMotion<'_>>,
        motion: &ID3D11UnorderedAccessView,
        depth_out: &ID3D11UnorderedAccessView,
        moved: Option<Moved<'_>>,
        (width, height): (u32, u32),
    ) -> Result<()> {
        let depth_desc = d3d::texture_desc(depth);
        if depth_desc.SampleDesc.Count != 1 {
            return Err(Error::Unsupported("multisampled depth"));
        }
        let copy = readable(&self.device, self.depth.take(), &depth_desc, depth_formats(depth_desc.Format).ok_or(Error::Unsupported("depth format"))?)?;
        let m = match previous {
            Some(previous) => reprojection(&current.projection, &current.camera, &previous.projection, &previous.camera).ok_or(Error::Unsupported("singular camera"))?,
            None => reprojection(&current.projection, &current.camera, &current.projection, &current.camera).ok_or(Error::Unsupported("singular camera"))?,
        };
        let object_view = object.as_ref().and_then(|o| self.object_view(o.texture));
        let previous_depth = moved.as_ref().and_then(|m| m.previous_depth).filter(|_| previous.is_some()).cloned();
        let tested = object_view.as_ref().is_some_and(|(_, tested)| *tested);
        let object_view = object_view.map(|(view, _)| view);
        let flags = u32::from(moved.is_some()) | (u32::from(previous_depth.is_some()) << 1) | (u32::from(tested) << 2);
        let p = previous.map_or(current.projection, |v| v.projection);
        let mut items = [[0.0; 16]; 2];
        let mut item_at = [[0.0; 4]; 2];
        let unproject = reprojection(&current.projection, &current.camera, &IDENTITY44, &IDENTITY34);
        if let (Some(moved), Some(previous), Some(_)) = (moved.as_ref().filter(|m| m.held_reach > 0.0), previous, unproject) {
            for side in 0..2 {
                if let Some(item) = moved.items[side]
                    && let Some(m) = item_reprojection(&item, previous)
                    && m.iter().all(|v| v.is_finite())
                {
                    items[side] = m;
                    let at = position(&item.now);
                    item_at[side] = [at[0], at[1], at[2], 1.0];
                }
            }
        }
        let constants = Constants {
            m,
            width,
            height,
            depth_width: depth_desc.Width,
            depth_height: depth_desc.Height,
            object_scale: object.as_ref().map_or([0.0; 2], |o| o.scale),
            has_object: u32::from(object_view.is_some()),
            flags,
            linear: [p[10], p[11], p[14], p[15]],
            held: moved.as_ref().map_or([0.0; 4], |m| held_depth(&current.projection, m.held_reach)),
            unproject: unproject.unwrap_or([0.0; 16]),
            items,
            item_at,
        };
        let saved = SavedCompute::take(context);
        // SAFETY: copies the game's depth into our same-shaped typeless copy.
        unsafe { context.CopyResource(&copy.texture, depth) };
        self.constants.update(context, &constants);
        self.constants.bind(context);
        // SAFETY: runs our shader on the game's context with our own resources; the game's compute
        // bindings go back after.
        unsafe {
            context.CSSetShader(&self.shader, None);
            context.CSSetShaderResources(0, Some(&[Some(copy.view.clone()), object_view, previous_depth]));
            let mask = moved.as_ref().map(|m| m.mask.clone());
            context.CSSetUnorderedAccessViews(0, 3, Some([Some(motion.clone()), Some(depth_out.clone()), mask].as_ptr()), None);
            context.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            context.CSSetUnorderedAccessViews(0, 3, Some([None, None, None].as_ptr()), None);
            context.CSSetShaderResources(0, Some(&[None, None, None]));
        }
        saved.restore(context);
        self.depth = Some(copy);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
