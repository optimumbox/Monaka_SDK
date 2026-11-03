//! AFW on D3D11: the GPU work of [`crate::afw`] (which eye is real, the gaps' rules, what goes out
//! at each present: the shared core) over the fence channel. Every presented frame is one
//! real eye, and the other eye of the same moment is that frame reprojected with its depth
//! (`monaka_warp::Reprojector`, the shader D3D12's [`crate::afw::d3d12`] runs too). What the real eye
//! could not see is filled from the other eye's own previous real frame, which saw it from the
//! right place, so the gaps get rendered pixels instead of stretched ones. A pair goes out every
//! frame (alternate-eye alone publishes every second frame, a frame apart).
//!
//! The game crate captures each frame's world image (before the HUD) and depth with the cameras
//! it was rendered from ([`Afw11::capture`]), and publishes at present ([`Afw11::publish`]).
//!
//! With frame generation ([`Afw11::frame_generation`], `between`) in-between pairs are added: each
//! eye's frame halfway between the last pair and this one, made by FSR frame generation on a D3D12
//! device of its own, goes out at once and the real pair half a frame later (`Afw11::due`).
//!
//! With the depth component on ([`crate::depth_layer`]) each warped pair carries both eyes' depth
//! for the headset: the real eye's depth plane and the warped eye's depth from the warp.

mod between;
pub mod ready;

pub use crate::afw::EyeCameras;
pub use between::FrameGeneration;
use between::{Between, PairIn, Waiting};
pub use ready::frame_warping;
use crate::afw::{Counts, History, Out, Presents, Rules, Shot, in_eye_order};
use crate::alternate::Step;
use crate::driver::Pending;
use crate::publish::{FLAT, Tally, copy_into, record_first};
use monaka_channel::d3d;
use monaka_channel::fence11::{Depth, EyeDepth, LazyProducer11};
use monaka_channel::pose::HeadSource;
use monaka_channel::ChannelName;
use monaka_contract::protocol::{DepthParams, HeadPose};
use monaka_producer::log;
use monaka_warp::{Frame, HudLayer, Reprojector};
use std::time::Instant;
use windows::Win32::Graphics::Direct3D11::{D3D11_BOX, D3D11_TEXTURE2D_DESC, ID3D11Device, ID3D11DeviceContext, ID3D11RenderTargetView, ID3D11Texture2D};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_TYPELESS;

/// Dying Light 1's rules: warped pairs only ([`Rules::WARPED_ONLY`]).
pub const RULES: Rules = Rules::WARPED_ONLY;

/// The present a pair goes out at: the loop's head source, the HUD's place in each eye (x, y,
/// width, height in pixels), whether it goes flat, its label, its number.
/// Both eyes' depth with a pair (left, right; R32 at one size), and how it maps to distance.
#[derive(Clone)]
pub(crate) struct PairDepth {
    textures: [ID3D11Texture2D; 2],
    params: DepthParams,
}

impl PairDepth {
    fn channel(&self) -> Depth<'_> {
        Depth { eyes: self.textures.each_ref().map(|texture| EyeDepth { texture, rect: None }), params: self.params }
    }
}

pub struct At<'a> {
    pub head: &'a HeadSource,
    pub hud_rects: [[f32; 4]; 2],
    pub flat: bool,
    pub label: Option<Pending>,
    pub present: u64,
}

pub struct Afw11 {
    reprojector: Reprojector,
    /// Each eye's latest real frame: copies of its world image and depth (how it was rendered and
    /// when are the core's [`History`]).
    frames: [Option<Frame>; 2],
    history: History,
    presents: Presents,
    /// This frame's real eye, once captured.
    latest: Option<Shot>,
    /// The head pose each eye's kept frame was drawn for (its held camera, [`held_camera`]).
    frame_poses: [Option<HeadPose>; 2],
    held_logged: u32,
    writer: LazyProducer11,
    shown: Option<ID3D11Texture2D>,
    /// Each eye's last frame as presented, for a plain pair (its pose and present are the core's).
    plain: [Option<ID3D11Texture2D>; 2],
    /// The HUD as its own layer, laid over both eyes; begun by this frame's first HUD draw.
    hud: Option<HudLayer>,
    hud_begun: bool,
    hud_views: Option<[ID3D11RenderTargetView; 2]>,
    counts: Counts,
    /// Frames captured per eye.
    captured: [u64; 2],
    tally: Tally,
    /// With a debug view: the farthest a fill's camera stood from its warp's (`take_fill_apart`).
    fill_apart: f32,
    /// With a debug view: the fills logged as standing apart from their warp's camera.
    gaps_logged: u32,
    /// Frame generation's in-between pairs, when asked for ([`Afw11::frame_generation`]).
    between: Option<Between>,
    /// Warped pairs still to be written beside the log ([`Afw11::snapshots`]).
    snapshots: Snapshots,
}

/// Warped pairs written beside the log: how many are left, when the next is due, the last's number.
#[derive(Default)]
struct Snapshots {
    left: u32,
    at: Option<Instant>,
    index: u32,
}

/// The camera a frame drawn from `camera` for the head pose `then` is drawn from for its held
/// things, seen from `now_camera` at the head pose `now`: the same place in the play space (where the
/// tracked head puts the drawn view, [`monaka_core::camera::tracking_space`]), the play space where
/// it is now. None without a valid head pose now.
fn held_camera(camera: &monaka_core::camera::Mat34, then: &HeadPose, now_camera: &monaka_core::camera::Mat34, now: &HeadPose) -> Option<monaka_core::camera::Mat34> {
    use monaka_core::camera::{compose, rigid_inverse, tracking_space};
    if now.valid == 0 || then.valid == 0 {
        return None;
    }
    let space_then = tracking_space(camera, then.orientation, then.position);
    let space_now = tracking_space(now_camera, now.orientation, now.position);
    let held = compose(&space_now, &compose(&rigid_inverse(&space_then), camera));
    held.iter().all(|v| v.is_finite()).then_some(held)
}

/// The first snapshot twice this long after the first warped pair, then one every this long.
const SNAPSHOT_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

impl Snapshots {
    /// The number of the snapshot due now, if one is.
    fn due(&mut self) -> Option<u32> {
        if self.left == 0 {
            return None;
        }
        let now = Instant::now();
        if now < *self.at.get_or_insert(now + 2 * SNAPSHOT_EVERY) {
            return None;
        }
        self.left -= 1;
        self.at = Some(now + SNAPSHOT_EVERY);
        self.index += 1;
        Some(self.index)
    }
}

/// Snapshot `index`: the warped pair `images` (in eye order, eye `warped` the warped one) and the
/// warped eye tinted by where each pixel came from, read now and written beside the log on a
/// thread of its own. The tinted view is drawn into the reprojector's output, so it is drawn
/// plainly again after for anything still holding it.
#[allow(clippy::too_many_arguments)]
fn snapshot(device: &ID3D11Device, context: &ID3D11DeviceContext, reprojector: &mut Reprojector, index: u32, images: &[ID3D11Texture2D; 2], warped: usize, target: &monaka_warp::View, primary: &Frame, fill: Option<&Frame>) {
    let Some(session) = monaka_producer::Session::get() else { return };
    let side = |eye: usize| ["left", "right"][eye.min(1)];
    let mut read = Vec::new();
    for (eye, image) in images.iter().enumerate() {
        let kind = if eye == warped { "warped" } else { "real" };
        if let Ok(image) = d3d::read_image(device, context, image) {
            read.push((session.path(&format!("afw-{index}-{}-{kind}.bmp", side(eye))), image));
        }
    }
    let sources = std::mem::replace(&mut reprojector.sources, 1);
    if let Ok(tinted) = reprojector.run(context, target, primary, fill).cloned()
        && let Ok(image) = d3d::read_image(device, context, &tinted)
    {
        read.push((session.path(&format!("afw-{index}-{}-sources.bmp", side(warped))), image));
    }
    reprojector.sources = sources;
    let _ = reprojector.run(context, target, primary, fill);
    log!("frame warping: snapshot {index} written beside the log (afw-{index}-*.bmp)");
    std::thread::spawn(move || {
        for (path, (width, height, rgba)) in read {
            let bgra: Vec<u8> = rgba.as_chunks::<4>().0.iter().flat_map(|p| [p[2], p[1], p[0], 255]).collect();
            let _ = std::fs::write(path, monaka_core::bmp::bgra(width, height, &bgra));
        }
    });
}

impl Afw11 {
    pub fn new(device: &ID3D11Device, channel: ChannelName) -> Result<Self, monaka_warp::Error> {
        Self::with_rules(device, channel, RULES)
    }

    /// With `rules` of the game's own in place of [`RULES`] (an OpenGL game's frames go through
    /// the same-frame loop: `crate::same_frame::gl`).
    pub fn with_rules(device: &ID3D11Device, channel: ChannelName, rules: Rules) -> Result<Self, monaka_warp::Error> {
        let mut reprojector = Reprojector::new(device)?;
        reprojector.fill_first = rules.fill_first;
        reprojector.held_reach = f32::from(rules.held_reach_cm) / 100.0;
        Ok(Self {
            reprojector,
            frames: [None, None],
            history: History::new(rules),
            presents: Presents::new(rules),
            latest: None,
            writer: LazyProducer11::new(channel),
            shown: None,
            plain: [None, None],
            hud: None,
            hud_begun: false,
            hud_views: None,
            counts: Counts::default(),
            captured: [0; 2],
            snapshots: Snapshots::default(),
            frame_poses: [None, None],
            held_logged: 0,
            tally: Tally::NEW,
            fill_apart: 0.0,
            gaps_logged: 0,
            between: None,
        })
    }

    /// Adds in-between pairs by FSR frame generation (`between`): each warped pair's generated pair
    /// goes out at once, the real pair half a frame later, when `Afw11::due` says (or before the
    /// next present's pair). Without frame generation available the pairs go out as before.
    pub fn frame_generation(&mut self, settings: FrameGeneration) {
        self.between = Some(Between::new(settings));
    }

    /// When the real pair waiting behind its generated pair goes out ([`Afw11::send_due`]). The
    /// thread that owns the device's context sends it (the GL path's warp thread): a pacing thread
    /// of its own, as D3D12 has, would publish on a context another thread is using.
    pub(crate) fn due(&self) -> Option<Instant> {
        self.between.as_ref()?.waiting.as_ref().map(|w| w.at)
    }

    /// Publishes the real pair waiting behind its generated pair, if one waits. Whether it went out.
    pub(crate) fn send_due(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, head: &HeadSource) -> bool {
        let Some(Waiting { images, pose, depth, .. }) = self.between.as_mut().and_then(|b| b.waiting.take()) else { return false };
        let sent = self.send(device, context, [&images[0], &images[1]], [pose; 2], depth.as_ref(), head, |_| {});
        if sent {
            crate::afw::Counters::add(&ready::COUNTERS.real);
        }
        sent
    }

    /// Takes the HUD as a layer laid over both eyes ([`Afw11::hud_targets`]) instead of leaving
    /// it in the real eye's image only.
    pub fn enable_hud(&mut self, device: &ID3D11Device) -> Result<(), monaka_warp::Error> {
        self.hud = Some(HudLayer::new(device)?);
        Ok(())
    }

    /// Shows a debug view in the warped eye instead of its image (`monaka_warp::Reprojector::sources`).
    pub fn show_sources(&mut self, view: u32) {
        self.reprojector.sources = view;
    }

    /// Writes `count` warped pairs beside the log, one every [`SNAPSHOT_EVERY`]: each eye as sent
    /// and the warped eye tinted by where each pixel came from (`monaka_warp::Reprojector::sources`),
    /// to see the warp's faults without a headset. Each costs the warp thread a moment.
    pub fn snapshots(&mut self, count: u32) {
        if self.snapshots.index == 0 {
            self.snapshots.left = count;
        }
    }

    /// With a debug view: the farthest a fill's camera stood from the camera it was warped to since
    /// the last call (world units; 0 with a still head).
    pub fn take_fill_apart(&mut self) -> Option<f32> {
        (self.reprojector.sources != 0).then(|| std::mem::take(&mut self.fill_apart))
    }

    /// Counts one failure of `what`, logging the first.
    fn failed(&mut self, what: &str, e: impl std::fmt::Display) {
        count_failure(&mut self.counts, what, e);
    }

    /// The black and white targets this frame's HUD draws go into (each draw into both), begun
    /// and cleared at the frame's first HUD draw. `None` without a HUD layer.
    pub fn hud_targets(&mut self, context: &ID3D11DeviceContext, width: u32, height: u32) -> Option<[ID3D11RenderTargetView; 2]> {
        if self.hud_begun
            && let Some(views) = &self.hud_views
        {
            return Some(views.clone());
        }
        let layer = self.hud.as_mut()?;
        match layer.begin(context, width, height) {
            Ok(targets) => {
                self.hud_begun = true;
                self.hud_views = Some(targets.clone());
                Some(targets)
            }
            Err(e) => {
                self.failed("HUD layer unavailable", e);
                None
            }
        }
    }

    /// The warps so far, and each eye's frames captured.
    pub fn counts(&self) -> (Counts, [u64; 2]) {
        (self.counts, self.captured)
    }

    /// The periodic report's line ([`crate::afw::report_line`]) for this backend's warps.
    pub fn report_line(&self) -> String {
        crate::afw::report_line(self.captured, self.counts, &ready::COUNTERS, 0, self.between.as_ref().map(|b| b.skipped))
    }

    /// Pairs published, and pairs the channel had no room for.
    pub fn published(&self) -> (u64, u64) {
        (self.tally.pairs, self.tally.failures)
    }

    /// Pairs published with each eye's depth.
    pub fn with_depth(&self) -> u64 {
        self.writer.pairs_with_depth()
    }

    /// The frame of `shot` is complete before its HUD: keep its world image `color` and `depth` as
    /// its eye's latest real frame.
    pub fn capture(&mut self, context: &ID3D11DeviceContext, shot: &Shot, color: &ID3D11Texture2D, depth: &ID3D11Texture2D) {
        self.capture_jittered(context, shot, color, depth, [0.0; 2]);
    }

    /// [`Afw11::capture`] of an upscaled frame: `color` unjittered at its full size, `depth` at the
    /// render size from the projection shifted by `jitter` (render pixels, x right, y down).
    pub fn capture_jittered(&mut self, context: &ID3D11DeviceContext, shot: &Shot, color: &ID3D11Texture2D, depth: &ID3D11Texture2D, jitter: [f32; 2]) {
        let eye = shot.eye;
        self.history.begin();
        let reuse = self.frames[eye].take();
        match self.reprojector.capture(context, color, depth, shot.rendered.projection, shot.rendered.camera, reuse) {
            Ok(mut frame) => {
                frame.jitter = jitter;
                self.frames[eye] = Some(frame);
                self.frame_poses[eye] = Some(shot.pose).filter(|p| p.valid != 0);
                self.history.keep(shot);
                self.latest = Some(*shot);
                self.captured[eye] += 1;
            }
            Err(e) => {
                self.failed("frame warping capture failed", e);
                self.history.lose(eye);
                self.latest = None;
            }
        }
    }

    /// At present `at`: publishes what the core says goes out of this frame (`final_image`, the
    /// finished back buffer): its real eye and the other eye reprojected from it, the HUD layer laid
    /// over each at its place when one was drawn this frame. `report` adjusts the pose record.
    pub fn publish(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, final_image: &ID3D11Texture2D, at: &At, report: impl Fn(&mut HeadPose)) -> bool {
        // A real pair still waiting goes out first, so pairs never go out of order.
        self.send_due(device, context, at.head);
        let hud_drawn = std::mem::take(&mut self.hud_begun);
        let shot = self.latest.take();
        let out = self.presents.out(at.flat, at.label.is_some(), shot.is_some(), Instant::now());
        if out.forgets() {
            self.history.forget();
        }
        match (out, shot, at.label) {
            (Out::Warped, Some(shot), _) => self.publish_warped(device, context, final_image, &shot, at, hud_drawn, report),
            (Out::Flat, ..) => {
                crate::afw::Counters::add(&ready::COUNTERS.flat);
                let whole = whole_frame(final_image);
                let Some(image) = self.shown_image(device, context, final_image, hud_drawn.then_some((0, whole))) else { return false };
                self.send(device, context, [&image, &image], FLAT, None, at.head, |_| {})
            }
            (Out::BothEyes, ..) => {
                let pose = at.head.current().unwrap_or_default();
                let whole = whole_frame(final_image);
                let Some(image) = self.shown_image(device, context, final_image, hud_drawn.then_some((0, whole))) else { return false };
                self.send(device, context, [&image, &image], [pose; 2], None, at.head, report)
            }
            (Out::Plain, _, Some(label)) => self.publish_plain(device, context, final_image, label, at, hud_drawn, report),
            (_, shot, _) => {
                if shot.is_none() {
                    crate::afw::Counters::add(&ready::COUNTERS.no_inputs);
                }
                false
            }
        }
    }

    /// A frame the game drew with both eyes side by side (its own stereo layout, as in a menu):
    /// `image`'s left and right halves go out as the pair without a pose record, so the viewer
    /// shows it on its screen as the same-frame loop's menus are, not warped, both eyes' frames
    /// forgotten first as for any frame not one eye's.
    pub fn publish_halves(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, image: &ID3D11Texture2D, at: &At) -> bool {
        self.send_due(device, context, at.head);
        self.hud_begun = false;
        self.latest = None;
        self.history.forget();
        self.presents.forget();
        let Some(shown) = self.shown_image(device, context, image, None) else { return false };
        let desc = d3d::texture_desc(&shown);
        let (width, height) = (desc.Width / 2, desc.Height);
        let Some(writer) = self.writer.get(device, width, height, desc.Format).ready(|note| log!("{note} (alternate-eye with depth)")) else { return false };
        let halves = [0, width].map(|left| D3D11_BOX { left, top: 0, front: 0, right: left + width, bottom: height, back: 1 });
        let published = writer.publish(context, &shown, halves, monaka_channel::tick());
        match self.tally.count(published) {
            Ok(step) => step == Step::Published,
            Err(e) => {
                self.failed("publish failed", e);
                false
            }
        }
    }

    /// The warped pair: the real eye's finished image and the other eye reprojected from its frame,
    /// the gaps from the other eye's own latest real frame while the rules let it fill.
    #[allow(clippy::too_many_arguments)]
    fn publish_warped(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, final_image: &ID3D11Texture2D, shot: &Shot, at: &At, hud_drawn: bool, report: impl Fn(&mut HeadPose)) -> bool {
        let (eye, other) = (shot.eye, shot.other_eye());
        let desc = d3d::texture_desc(final_image);
        let warp = self.history.warp(shot);
        let Some(primary) = &self.frames[eye] else { return false };
        let fill = warp.fill.and(self.frames[other].as_ref());
        if let Some(fill) = fill.filter(|_| self.reprojector.sources != 0) {
            let (a, b) = (&fill.camera, &warp.target.camera);
            let gap = [a[3] - b[3], a[7] - b[7], a[11] - b[11]];
            let apart = gap.iter().map(|v| v * v).sum::<f32>().sqrt();
            self.fill_apart = self.fill_apart.max(apart);
            // The first gaps over half a millimetre, in the target camera's axes (right, up, back).
            if apart > 0.0005 && self.gaps_logged < 60 {
                self.gaps_logged += 1;
                let along = |k: usize| (0..3).map(|i| gap[i] * b[i * 4 + k]).sum::<f32>() * 1000.0;
                log!("frame warping: eye {other}'s fill {:.1} mm from its target: right {:+.1} up {:+.1} back {:+.1} mm", apart * 1000.0, along(0), along(1), along(2));
            }
        }
        let target = warp.target.view(desc.Width, desc.Height);
        self.reprojector.fill_held_camera = fill.zip(self.frame_poses[other]).and_then(|(fill, then)| held_camera(&fill.camera, &then, &warp.target.camera, &shot.pose));
        if let Some(held) = self.reprojector.fill_held_camera.filter(|_| self.held_logged < 8)
            && let Some(fill) = fill
        {
            let moved = [held[3] - fill.camera[3], held[7] - fill.camera[7], held[11] - fill.camera[11]];
            let moved = moved.iter().map(|v| v * v).sum::<f32>().sqrt();
            if moved > 0.01 {
                self.held_logged += 1;
                log!("frame warping: the play space moved {:.1} cm since eye {other}'s last frame: its held things drawn from there", moved * 100.0);
            }
        }
        // With frame generation or the headset's depth, the warped eye's depth too, at the real
        // eye's depth size.
        let real_depth = (primary.depth().clone(), primary.jitter);
        let framegen = self.between.as_ref().is_some_and(Between::on);
        let headset_params = crate::depth_layer::on().then(|| shot.rendered.depth_params(crate::depth_layer::units_per_metre())).flatten();
        let mut drawn = None;
        if framegen || headset_params.is_some() {
            let depth = d3d::texture_desc(&real_depth.0);
            match self.reprojector.run_with_depth(context, &target, primary, fill, (depth.Width, depth.Height)) {
                Ok((image, depth)) => drawn = Some(Ok((image.clone(), Some(depth.clone())))),
                Err(e) => {
                    if let Some(between) = self.between.as_mut().filter(|_| framegen) {
                        between.give_up(format!("the warped eye's depth: {e}"));
                    } else {
                        count_failure(&mut self.counts, "the warped eye's depth", e);
                    }
                }
            }
        }
        let drawn = drawn.unwrap_or_else(|| self.reprojector.run(context, &target, primary, fill).map(|image| (image.clone(), None)));
        let (synthesized, warped_depth) = match drawn {
            Ok(drawn) => drawn,
            Err(e) => {
                self.failed("frame warping reprojection failed", e);
                return false;
            }
        };
        // Both eyes' depth for the headset: the real eye's depth plane beside the warped eye's.
        let pair_depth = match (headset_params, &warped_depth) {
            (Some(params), Some(warped)) => match self.reprojector.plain_depth(context, primary) {
                Ok(real) => Some(PairDepth { textures: in_eye_order(eye, real.clone(), warped.clone()), params }),
                Err(e) => {
                    count_failure(&mut self.counts, "the real eye's depth", e);
                    None
                }
            },
            _ => None,
        };
        self.counts.pairs += 1;
        if fill.is_some() {
            self.counts.filled += 1;
        }
        let Some(shown) = self.shown_image(device, context, final_image, None) else { return false };
        let mut images = in_eye_order(eye, shown, synthesized);
        if hud_drawn {
            for (slot, image) in images.iter_mut().enumerate() {
                if let Some(with_hud) = self.with_hud(context, image, slot, at.hud_rects[slot]) {
                    *image = with_hud;
                }
            }
        }
        if let Some(index) = self.snapshots.due()
            && let Some(primary) = &self.frames[eye]
        {
            let fill = warp.fill.and(self.frames[other].as_ref());
            snapshot(device, context, &mut self.reprojector, index, &images, other, &target, primary, fill);
        }
        // The pose record as it always went: the report applied to the frame's pose, valid or not.
        let mut pose = shot.pose;
        report(&mut pose);
        if let (Some(between), Some(warped_depth)) = (self.between.as_mut(), warped_depth.as_ref()) {
            let due = between.interval.due(Instant::now());
            let pair = PairIn {
                shot,
                present: at.present,
                images: [&images[0], &images[1]],
                depths: in_eye_order(eye, &real_depth.0, warped_depth),
                jitters: in_eye_order(eye, real_depth.1, [0.0; 2]),
            };
            if let Some(made) = between.pair(device, context, &mut self.presents, &pair) {
                // The generated pair now, the real pair at its time.
                // Halfway between the last pair and this one: this (the newer) pair's depth.
                between.waiting = Some(Waiting { at: due, images: made.kept, pose, depth: pair_depth.clone() });
                let mut halfway = made.halfway;
                report(&mut halfway);
                let sent = self.send(device, context, [&made.generated[0], &made.generated[1]], [halfway; 2], pair_depth.as_ref(), at.head, |_| {});
                if sent {
                    crate::afw::Counters::add(&ready::COUNTERS.generated);
                }
                // A pair goes out either way: the real one at its time.
                return true;
            }
        }
        self.send(device, context, [&images[0], &images[1]], [pose; 2], pair_depth.as_ref(), at.head, |_| {})
    }

    /// A frame not warped: its finished image as its eye's newest, beside the other eye's newest
    /// when that is the frame just before.
    #[allow(clippy::too_many_arguments)]
    fn publish_plain(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, final_image: &ID3D11Texture2D, label: Pending, at: &At, hud_drawn: bool, report: impl Fn(&mut HeadPose)) -> bool {
        let eye = label.eye.min(1);
        self.presents.lose_plain(eye);
        let Some(image) = self.shown_image(device, context, final_image, hud_drawn.then_some((eye, at.hud_rects[eye]))) else { return false };
        let desc = d3d::texture_desc(&image);
        let mut slot = self.plain[eye].take();
        let Some(copy) = copy_into(&mut slot, device, context, &image, &d3d::private_copy_desc(&desc)).cloned() else { return false };
        self.plain[eye] = Some(copy.clone());
        let Some(other_pose) = self.presents.plain(eye, label.pose, at.present) else { return false };
        let Some(other) = self.plain[1 - eye].clone() else { return false };
        crate::afw::Counters::add(&ready::COUNTERS.plain);
        self.send(device, context, in_eye_order(eye, &copy, &other), in_eye_order(eye, label.pose, other_pose), None, at.head, report)
    }

    /// The finished image copied so the game can reuse its back buffer (typeless, so an sRGB back
    /// buffer is read as stored), with the HUD layer laid over it as eye `hud.0` at `hud.1` when
    /// asked.
    fn shown_image(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, final_image: &ID3D11Texture2D, hud: Option<(usize, [f32; 4])>) -> Option<ID3D11Texture2D> {
        let desc = d3d::texture_desc(final_image);
        let copy = D3D11_TEXTURE2D_DESC { Format: DXGI_FORMAT_R8G8B8A8_TYPELESS, ..d3d::private_copy_desc(&desc) };
        let shown = copy_into(&mut self.shown, device, context, final_image, &copy).cloned()?;
        match hud {
            Some((eye, rect)) => Some(self.with_hud(context, &shown, eye, rect).unwrap_or(shown)),
            None => Some(shown),
        }
    }

    /// `image` with this frame's HUD layer laid over it as eye `eye` at `rect`.
    fn with_hud(&mut self, context: &ID3D11DeviceContext, image: &ID3D11Texture2D, eye: usize, rect: [f32; 4]) -> Option<ID3D11Texture2D> {
        let result = self.hud.as_mut()?.composite(context, eye, image, rect).cloned();
        result.map_err(|e| self.failed("HUD composite failed", e)).ok()
    }

    /// Publishes `images` (left, right) with `poses` (`report` adjusting each eye that has one) and
    /// each eye's `depth` for the headset.
    #[allow(clippy::too_many_arguments)]
    fn send(&mut self, device: &ID3D11Device, context: &ID3D11DeviceContext, images: [&ID3D11Texture2D; 2], poses: [HeadPose; 2], depth: Option<&PairDepth>, head: &HeadSource, report: impl Fn(&mut HeadPose)) -> bool {
        let desc = d3d::texture_desc(images[0]);
        // A size the viewer did not open the channel at is refused (logged once), not failed.
        let Some(writer) = self.writer.get(device, desc.Width, desc.Height, desc.Format).ready(|note| log!("{note} (alternate-eye with depth)")) else { return false };
        record_first(head, writer.next_sequence(), poses, report);
        let published = writer.publish_whole_with_depth(context, images, depth.map(PairDepth::channel), monaka_channel::tick());
        match self.tally.count(published) {
            Ok(step) => step == Step::Published,
            Err(e) => {
                self.failed("publish failed", e);
                false
            }
        }
    }
}

/// Counts one failure of `what` in `counts`, logging the first.
fn count_failure(counts: &mut Counts, what: &str, e: impl std::fmt::Display) {
    if counts.failures == 0 {
        log!("{what}: {e}");
    }
    counts.failures += 1;
}

/// The whole of `image` as a HUD place.
fn whole_frame(image: &ID3D11Texture2D) -> [f32; 4] {
    let desc = d3d::texture_desc(image);
    [0.0, 0.0, desc.Width as f32, desc.Height as f32]
}
