use super::*;
use monaka_channel::d3d::{create_device, create_texture_with, read_image, read_pixel};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R32_FLOAT};

fn texture(device: &ID3D11Device, w: u32, h: u32, format: DXGI_FORMAT, bytes: &[u8]) -> ID3D11Texture2D {
    create_texture_with(device, &D3D11_TEXTURE2D_DESC { BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32, ..d3d::simple_texture_desc(w, h, format) }, bytes).unwrap()
}

#[test]
fn shaders_compile() {
    for entry in [s!("Clear"), s!("Splat"), s!("Paint"), s!("Resolve"), s!("ResolveDepth")] {
        crate::compile(REPROJECT_SOURCE, "reproject.hlsl", entry).unwrap();
    }
}

#[test]
fn other_eye_from_this_frame_and_gaps_from_its_own_last_frame() {
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 8u32);
    // A standard-depth projection with P00 = P11 = 1 and a wall 2 m away everywhere.
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    projection[0] = 1.0;
    projection[5] = 1.0;
    projection[10] = f / (n - f);
    projection[11] = n * f / (n - f);
    projection[14] = -1.0;
    let wall = -projection[10] + projection[11] / 2.0;
    let depth: Vec<u8> = (0..w * h).flat_map(|_| wall.to_le_bytes()).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    // This frame: the left camera, red = 4 * column. The right eye's last frame: plain green.
    let columns: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| [(x * 4) as u8, 0, 0, 255])).collect();
    let green: Vec<u8> = (0..w * h).flat_map(|_| [0, 255, 0, 255]).collect();
    let left = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &columns);
    let last_right = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &green);
    let primary = reprojector.capture(&context, &left, &depth, projection, camera(-0.5), None).unwrap();
    let fill = reprojector.capture(&context, &last_right, &depth, projection, camera(0.5), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.5), mapping, width: w, height: h };

    // Cameras 1 m apart, wall 2 m away: the right eye sees each column 16 pixels further left.
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    assert_eq!(read_pixel(&device, &context, &image, 20, 3).unwrap(), [36 * 4, 0, 0, 255]);
    // The right edge was out of the left camera's view: its own last frame fills it.
    assert_eq!(read_pixel(&device, &context, &image, 55, 3).unwrap(), [0, 255, 0, 170]);
    // Without a fill frame, the gap continues its neighbour instead.
    let image = reprojector.run(&context, &view, &primary, None).unwrap().clone();
    assert_eq!(read_pixel(&device, &context, &image, 55, 3).unwrap(), [63 * 4, 0, 0, 85]);
}

#[test]
fn a_surface_drawn_larger_keeps_its_own_colour_in_its_cracks() {
    // The wall 2 m away seen from 1 m nearer is drawn twice its size: every other target pixel has no
    // source pixel. Those are cracks in the wall, not gaps beside something nearer: the wall's own
    // colour (as it is, or across the crack: alpha 255 or 85), never the fill's (170).
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 16u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let wall = -projection[10] + projection[11] / 2.0;
    let depth: Vec<u8> = (0..w * h).flat_map(|_| wall.to_le_bytes()).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let camera = |z: f32| -> Mat34 { [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z] };
    let red: Vec<u8> = (0..w * h).flat_map(|_| [255, 0, 0, 255]).collect();
    let green: Vec<u8> = (0..w * h).flat_map(|_| [0, 255, 0, 255]).collect();
    let this = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &red);
    let last = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &green);
    let primary = reprojector.capture(&context, &this, &depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &last, &depth, projection, camera(-1.0), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(-1.0), mapping, width: w, height: h };
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let pixel = read_pixel(&device, &context, &image, x, y).unwrap();
            assert!(pixel[..3] == [255, 0, 0] && pixel[3] != 170, "pixel {x}, {y}: {pixel:?}");
        }
    }
}

#[test]
fn a_gap_beside_something_near_shows_what_is_behind_it_not_a_near_fill() {
    // A box 0.5 m away in front of a wall 2 m away; the eye 0.1 m to the right sees the box 4.8
    // pixels further left than the wall, a gap opening right of it. The fill frame is all near
    // (green, at the box's depth: a near thing that moved since, as a held bow): the gap shows the wall,
    // continued from beside it (alpha 85).
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 8u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (box_columns, wall, near) = (24..36u32, [255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let depth: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| at(if box_columns.contains(&x) { 0.5 } else { 2.0 }).to_le_bytes())).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let colours: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| if box_columns.contains(&x) { near } else { wall })).collect();
    let this = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colours);
    let near_depth: Vec<u8> = (0..w * h).flat_map(|_| at(0.5).to_le_bytes()).collect();
    let near_depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &near_depth);
    let green: Vec<u8> = (0..w * h).flat_map(|_| [0, 255, 0, 255]).collect();
    let last = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &green);
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &this, &depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &last, &near_depth, projection, camera(0.1), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.1), mapping, width: w, height: h };
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    // The box: source columns 24..36 land 6.4 pixels left, 17..29; the wall 1.6 left; the gap between.
    assert_eq!(read_pixel(&device, &context, &image, 22, 3).unwrap(), near);
    for x in 30..33 {
        assert_eq!(read_pixel(&device, &context, &image, x, 3).unwrap(), [wall[0], wall[1], wall[2], 85], "column {x}");
    }
    // The fill showing something 1 m away there (a bar between the box and the wall, which the
    // real eye could not see behind the box): clearly behind the box, so the fill (alpha 170).
    let bar_depth: Vec<u8> = (0..w * h).flat_map(|_| at(1.0).to_le_bytes()).collect();
    let bar_depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &bar_depth);
    let fill = reprojector.capture(&context, &last, &bar_depth, projection, camera(0.1), None).unwrap();
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    for x in 30..33 {
        assert_eq!(read_pixel(&device, &context, &image, x, 3).unwrap(), [0, 255, 0, 170], "column {x}: the bar from the fill");
    }
}

#[test]
fn a_gap_within_the_hands_reach_never_takes_the_eyes_own_frame() {
    // The box 0.5 m away in front of the wall 2 m away, as above; the eye's own last frame shows
    // something 0.8 m away there (green): the real eye could not see it behind the box, so it fills
    // the gap. Within the hands' reach (0.9 m) that is a held gun where it was a frame ago: the gap
    // shows the wall behind it instead (alpha 85).
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 8u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (box_columns, wall, near) = (24..36u32, [255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let depth: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| at(if box_columns.contains(&x) { 0.5 } else { 2.0 }).to_le_bytes())).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let colours: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| if box_columns.contains(&x) { near } else { wall })).collect();
    let this = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colours);
    let held_depth: Vec<u8> = (0..w * h).flat_map(|_| at(0.8).to_le_bytes()).collect();
    let held_depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &held_depth);
    let green: Vec<u8> = (0..w * h).flat_map(|_| [0, 255, 0, 255]).collect();
    let last = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &green);
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &this, &depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &last, &held_depth, projection, camera(0.1), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.1), mapping, width: w, height: h };
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    for x in 30..32 {
        assert_eq!(read_pixel(&device, &context, &image, x, 3).unwrap(), [0, 255, 0, 170], "column {x}: without a reach, the fill");
    }
    reprojector.held_reach = 0.9;
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    for x in 30..33 {
        assert_eq!(read_pixel(&device, &context, &image, x, 3).unwrap(), [wall[0], wall[1], wall[2], 85], "column {x}: the wall behind, not the held thing");
    }
    // Past the reach (a bar 1 m away, as above), the fill as before.
    let bar_depth: Vec<u8> = (0..w * h).flat_map(|_| at(1.0).to_le_bytes()).collect();
    let bar_depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &bar_depth);
    let fill = reprojector.capture(&context, &last, &bar_depth, projection, camera(0.1), None).unwrap();
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    assert_eq!(read_pixel(&device, &context, &image, 31, 3).unwrap(), [0, 255, 0, 170], "the bar past the reach, from the fill");
}

#[test]
fn held_things_in_the_eyes_own_frame_move_with_the_play_space() {
    // As above, the eye's own last frame showing something 0.8 m away (green) everywhere, but drawn
    // 0.1 m to the left of where the eye is now: the stick moved the player since. In the world the
    // green thing would have moved 0.1 m against the eye; held, it moved with it. With the held
    // camera (where the last frame's camera would be in today's play space, here the eye itself),
    // it lands where it was in that frame and fills the gap beside the box.
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 8u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (box_columns, wall, near) = (24..36u32, [255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let depth: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| at(if box_columns.contains(&x) { 0.5 } else { 2.0 }).to_le_bytes())).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let colours: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| if box_columns.contains(&x) { near } else { wall })).collect();
    let this = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colours);
    let held_depth: Vec<u8> = (0..w * h).flat_map(|_| at(0.8).to_le_bytes()).collect();
    let held_depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &held_depth);
    let green: Vec<u8> = (0..w * h).flat_map(|_| [0, 255, 0, 255]).collect();
    let last = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &green);
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &this, &depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &last, &held_depth, projection, camera(0.0), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.1), mapping, width: w, height: h };
    reprojector.held_reach = 0.9;
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    for x in 30..33 {
        assert_eq!(read_pixel(&device, &context, &image, x, 3).unwrap(), [wall[0], wall[1], wall[2], 85], "column {x}: without the held camera, the wall behind");
    }
    reprojector.fill_held_camera = Some(camera(0.1));
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    for x in 30..32 {
        assert_eq!(read_pixel(&device, &context, &image, x, 3).unwrap(), [0, 255, 0, 170], "column {x}: the held thing, moved with the play space");
    }
}

#[test]
fn a_hole_the_real_eye_could_not_see_in_a_surface_comes_from_the_eyes_own_frame() {
    // A bar 1 m away in front of a wall 4 m away; in the real eye something 0.25 m away hides the
    // bar's middle rows, but not the wall just left of it. The other eye 0.25 m to the right sees
    // the wall shift 4 pixels, the bar 16: that wall slides into the bar's place in those rows (the
    // wall seen through the bar, in the hider's shape). Its own last frame (from where it is) shows
    // the bar there, and the bar is above and below: the bar, from that frame (alpha 170). Where that
    // frame shows something near (a hand that has moved since) with nothing of it around, the warp.
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (128u32, 32u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (wall, bar, hider) = ([255u8, 0, 0, 255], [0u8, 255, 0, 255], [0u8, 0, 255, 255]);
    let hidden = |x: u32, y: u32| (12..20).contains(&y) && (60..90).contains(&x);
    let real = |x: u32, y: u32| if hidden(x, y) { (hider, 0.25) } else if (60..72).contains(&x) { (bar, 1.0) } else { (wall, 4.0) };
    // The other eye's own frame: the bar where it sees it (16 pixels left), and a near blob at 96..104, 2..8.
    let own = |x: u32, y: u32| if (96..104).contains(&x) && (2..8).contains(&y) { (hider, 0.25) } else if (44..56).contains(&x) { (bar, 1.0) } else { (wall, 4.0) };
    let images = |of: &dyn Fn(u32, u32) -> ([u8; 4], f32)| {
        let colour: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| of(x, y).0)).collect();
        let depth: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| at(of(x, y).1).to_le_bytes())).collect();
        (texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colour), texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth))
    };
    let ((real_colour, real_depth), (own_colour, own_depth)) = (images(&real), images(&own));
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &real_colour, &real_depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &own_colour, &own_depth, projection, camera(0.25), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.25), mapping, width: w, height: h };
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    let pixel = |x: u32, y: u32| read_pixel(&device, &context, &image, x, y).unwrap();
    assert_eq!(pixel(50, 5), [0, 255, 0, 255], "the bar, warped");
    for y in 12..20 {
        assert_eq!(pixel(50, y), [0, 255, 0, 170], "row {y}: the bar from the eye's own frame, not the wall");
    }
    assert_eq!(pixel(100, 5), [255, 0, 0, 255], "the wall, warped: the own frame's near blob has nothing of it around");
}

#[test]
fn a_bar_crossing_a_gap_comes_from_the_eyes_own_frame() {
    // A bar 1 m away in front of a wall 4 m away; in the real eye something 0.25 m away hides a block
    // around the bar's middle rows, bar and wall alike. The other eye 0.25 m to the right sees a gap
    // there with only wall left and right of it (what made it has moved off 64 pixels): the bar runs
    // through the gap, above and below it in the warp, so the bar from its own frame (alpha 170), as
    // is the wall beside it there.
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (128u32, 32u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (wall, bar, hider) = ([255u8, 0, 0, 255], [0u8, 255, 0, 255], [0u8, 0, 255, 255]);
    let real = |x: u32, y: u32| if (12..20).contains(&y) && (40..90).contains(&x) { (hider, 0.25) } else if (60..72).contains(&x) { (bar, 1.0) } else { (wall, 4.0) };
    let own = |x: u32, _: u32| if (44..56).contains(&x) { (bar, 1.0) } else { (wall, 4.0) };
    let images = |of: &dyn Fn(u32, u32) -> ([u8; 4], f32)| {
        let colour: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| of(x, y).0)).collect();
        let depth: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| at(of(x, y).1).to_le_bytes())).collect();
        (texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colour), texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth))
    };
    let ((real_colour, real_depth), (own_colour, own_depth)) = (images(&real), images(&own));
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &real_colour, &real_depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &own_colour, &own_depth, projection, camera(0.25), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.25), mapping, width: w, height: h };
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    let pixel = |x: u32, y: u32| read_pixel(&device, &context, &image, x, y).unwrap();
    for y in 13..19 {
        assert_eq!(pixel(50, y), [0, 255, 0, 170], "row {y}: the bar through the gap");
        assert_eq!(pixel(70, y), [255, 0, 0, 170], "row {y}: the wall in the gap");
    }
}

#[test]
fn what_a_large_near_thing_hid_comes_from_the_eyes_own_frame_and_what_moved_does_not() {
    // As above, but what hides the bar in the real eye is large (a hand holding a flashlight, rows
    // 10..118 of 128): no bar within reach above or below the gap. The real eye is asked instead: its
    // own frame's bar is hidden there in the real eye (something nearer at its place), so the bar.
    // A near blob in the own frame (a hand that has moved since) where the real eye sees the wall,
    // past where the blob would be: the warp's wall.
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (128u32, 128u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (wall, bar, hider) = ([255u8, 0, 0, 255], [0u8, 255, 0, 255], [0u8, 0, 255, 255]);
    let real = |x: u32, y: u32| if (10..118).contains(&y) && (40..90).contains(&x) { (hider, 0.25) } else if (60..72).contains(&x) { (bar, 1.0) } else { (wall, 4.0) };
    let own = |x: u32, y: u32| if (20..26).contains(&x) && (120..126).contains(&y) { (hider, 0.25) } else if (44..56).contains(&x) { (bar, 1.0) } else { (wall, 4.0) };
    let images = |of: &dyn Fn(u32, u32) -> ([u8; 4], f32)| {
        let colour: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| of(x, y).0)).collect();
        let depth: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| at(of(x, y).1).to_le_bytes())).collect();
        (texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colour), texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth))
    };
    let ((real_colour, real_depth), (own_colour, own_depth)) = (images(&real), images(&own));
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &real_colour, &real_depth, projection, camera(0.0), None).unwrap();
    let fill = reprojector.capture(&context, &own_colour, &own_depth, projection, camera(0.25), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.25), mapping, width: w, height: h };
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    let pixel = |x: u32, y: u32| read_pixel(&device, &context, &image, x, y).unwrap();
    for y in [20, 64, 100] {
        assert_eq!(pixel(50, y), [0, 255, 0, 170], "row {y}: the bar behind the large near thing");
    }
    assert_eq!(pixel(22, 122), [255, 0, 0, 255], "the wall: the own frame's blob has moved");
}

#[test]
fn a_hole_in_a_near_thing_that_moved_shows_the_thing_not_what_was_behind_it() {
    // A held gun 0.5 m away fills the real eye; the other eye is 0.375 m nearer, so the gun is drawn
    // four times its size, landed pixels 4 apart: holes in it too wide to be cracks. That eye's own
    // last frame shows the wall 2 m away everywhere (the gun was elsewhere then): the holes keep the
    // gun's colour, not the wall seen through it. Where that frame showed the gun too, its own frame
    // (alpha 170; a shade apart from the gun to tell it, as alike as one surface is).
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 64u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (gun, wall, own_gun) = ([0u8, 0, 255, 255], [255u8, 0, 0, 255], [0u8, 20, 255, 255]);
    let images = |colour: [u8; 4], distance: f32| {
        let colours: Vec<u8> = (0..w * h).flat_map(|_| colour).collect();
        let depth: Vec<u8> = (0..w * h).flat_map(|_| at(distance).to_le_bytes()).collect();
        (texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colours), texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth))
    };
    let camera = |z: f32| -> Mat34 { [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z] };
    let (gun_colour, gun_depth) = images(gun, 0.5);
    let primary = reprojector.capture(&context, &gun_colour, &gun_depth, projection, camera(0.0), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(-0.375), mapping, width: w, height: h };
    let (wall_colour, wall_depth) = images(wall, 1.625);
    let fill = reprojector.capture(&context, &wall_colour, &wall_depth, projection, camera(-0.375), None).unwrap();
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    let (_, _, pixels) = read_image(&device, &context, &image).unwrap();
    for y in 8..h - 8 {
        for x in 8..w - 8 {
            let pixel = &pixels[((y * w + x) * 4) as usize..][..4];
            assert!(pixel[..3] == gun[..3], "pixel {x}, {y}: {pixel:?}, the wall through the gun");
        }
    }
    let (own_colour, own_depth) = images(own_gun, 0.125);
    let fill = reprojector.capture(&context, &own_colour, &own_depth, projection, camera(-0.375), None).unwrap();
    let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
    let (_, _, pixels) = read_image(&device, &context, &image).unwrap();
    let own = (8..h - 8).flat_map(|y| (8..w - 8).map(move |x| (x, y))).filter(|&(x, y)| pixels[((y * w + x) * 4) as usize..][..4] == [0, 20, 255, 170]).count();
    assert!(own > 0, "the own frame's gun nowhere in the holes");
}

#[test]
fn the_cameras_far_from_the_origin_land_where_they_do_near_it() {
    // The same eyes and wall, the whole scene moved thousands of units away, as a game world's
    // cameras are: the warped image is the one made near the origin.
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h, near) = (64u32, 8u32, 0.1f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[11], projection[14]) = (1.0, 1.0, near, -1.0);
    let depth: Vec<u8> = (0..w * h).flat_map(|_| (near / 2.0).to_le_bytes()).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let columns: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| [(x * 4) as u8, 0, 0, 255])).collect();
    let left = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &columns);
    let mapping = DepthMapping::from_projection(&projection, near).unwrap();
    let mut at = |offset: f32| {
        let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, offset + x, 0.0, 1.0, 0.0, -offset, 0.0, 0.0, 1.0, offset] };
        let primary = reprojector.capture(&context, &left, &depth, projection, camera(-0.5), None).unwrap();
        let view = View { projection, camera: camera(0.5), mapping, width: w, height: h };
        let image = reprojector.run(&context, &view, &primary, None).unwrap().clone();
        (0..w).map(|x| read_pixel(&device, &context, &image, x, 3).unwrap()).collect::<Vec<_>>()
    };
    assert_eq!(at(4000.0), at(0.0));
}

#[test]
fn a_moving_things_old_place_beside_it_is_not_shown_from_the_eyes_own_frame() {
    // A box 1.8 m away before a wall 2 m away, warped into an eye 1.5 m to the right (wide, so the
    // gap right of the box is 11 pixels: no crack): the box lands 106.7 pixels left (columns
    // 13..43), the wall 96 (from column 55). That eye's own last frame saw the box where it was
    // then, over the gap (a character walking): the real eye sees the wall at that place, within
    // the depth test's eighth of the box, but red, not blue, so the gap shows the wall behind it
    // (alpha 85). Where that frame saw the wall, the wall it saw (alpha 170).
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (256u32, 8u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (wall, thing) = ([255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let scene = |columns: std::ops::Range<u32>| {
        let depth: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| at(if columns.contains(&x) { 1.8 } else { 2.0 }).to_le_bytes())).collect();
        let colours: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| if columns.contains(&x) { thing } else { wall })).collect();
        (texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colours), texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth))
    };
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let (colour, depth) = scene(120..151);
    let primary = reprojector.capture(&context, &colour, &depth, projection, camera(0.0), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(1.5), mapping, width: w, height: h };
    let plain = reprojector.run(&context, &view, &primary, None).unwrap().clone();
    assert_eq!(read_pixel(&device, &context, &plain, 30, 3).unwrap(), thing, "the box where it landed");
    let mut gap = |fill_columns: std::ops::Range<u32>| {
        let (colour, depth) = scene(fill_columns);
        let fill = reprojector.capture(&context, &colour, &depth, projection, camera(1.5), None).unwrap();
        let image = reprojector.run(&context, &view, &primary, Some(&fill)).unwrap().clone();
        (47..52).map(|x| read_pixel(&device, &context, &image, x, 3).unwrap()).collect::<Vec<_>>()
    };
    let old_place = gap(44..56);
    assert!(old_place.iter().all(|p| *p == [255, 0, 0, 85]), "the box's old place: {old_place:?}");
    let uncovered = gap(0..0);
    assert!(uncovered.iter().all(|p| *p == [255, 0, 0, 170]), "the wall the eye's own frame saw: {uncovered:?}");
}

#[test]
fn the_warped_eye_gets_its_own_depth_beside_its_image() {
    // A box 0.5 m away in front of a wall 2 m away, warped into an eye 0.1 m to the right: the box
    // lands 6.4 pixels left (columns 17..29), the wall 1.6. The warped eye's depth says the same at
    // any size: the box's depth where the box landed, the wall's elsewhere (the gap right of the box
    // takes the wall behind it), and the image is the one `run` draws.
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let (w, h) = (64u32, 8u32);
    let (n, f) = (0.1f32, 100.0f32);
    let mut projection = [0.0f32; 16];
    (projection[0], projection[5], projection[10], projection[11], projection[14]) = (1.0, 1.0, f / (n - f), n * f / (n - f), -1.0);
    let at = |distance: f32| -projection[10] + projection[11] / distance;
    let (box_columns, wall, near) = (24..36u32, [255u8, 0, 0, 255], [0u8, 0, 255, 255]);
    let depth: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| at(if box_columns.contains(&x) { 0.5 } else { 2.0 }).to_le_bytes())).collect();
    let depth = texture(&device, w, h, DXGI_FORMAT_R32_FLOAT, &depth);
    let colours: Vec<u8> = (0..h).flat_map(|_| (0..w).flat_map(|x| if box_columns.contains(&x) { near } else { wall })).collect();
    let this = texture(&device, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, &colours);
    let camera = |x: f32| -> Mat34 { [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] };
    let primary = reprojector.capture(&context, &this, &depth, projection, camera(0.0), None).unwrap();
    let mapping = DepthMapping::from_projection(&projection, n).unwrap();
    let view = View { projection, camera: camera(0.1), mapping, width: w, height: h };
    let drawn: Vec<[u8; 4]> = {
        let image = reprojector.run(&context, &view, &primary, None).unwrap().clone();
        (0..w).map(|x| read_pixel(&device, &context, &image, x, 3).unwrap()).collect()
    };
    let depth_at = |texture: &ID3D11Texture2D, x: u32, y: u32| {
        let (width, _, texels) = read_image(&device, &context, texture).unwrap();
        let i = ((y * width + x) * 4) as usize;
        f32::from_le_bytes([texels[i], texels[i + 1], texels[i + 2], texels[i + 3]])
    };
    let close = |got: f32, want: f32, what: &str| assert!((got - want).abs() < 1e-4, "{what}: depth {got}, expected {want}");
    let (image, warped_depth) = reprojector.run_with_depth(&context, &view, &primary, None, (w, h)).unwrap();
    let (image, warped_depth) = (image.clone(), warped_depth.clone());
    let with_depth: Vec<[u8; 4]> = (0..w).map(|x| read_pixel(&device, &context, &image, x, 3).unwrap()).collect();
    assert_eq!(with_depth, drawn, "the image is the one run draws");
    assert_eq!(drawn[22], near);
    close(depth_at(&warped_depth, 22, 3), at(0.5), "the box");
    close(depth_at(&warped_depth, 5, 3), at(2.0), "the wall");
    close(depth_at(&warped_depth, 31, 3), at(2.0), "the gap beside the box: the wall behind it");
    // At half the size: each depth texel takes the image pixel under its centre.
    let (_, half) = reprojector.run_with_depth(&context, &view, &primary, None, (w / 2, h / 2)).unwrap();
    let half = half.clone();
    assert_eq!(d3d::texture_desc(&half).Width, w / 2);
    close(depth_at(&half, 11, 1), at(0.5), "the box at half size (image column 23)");
    close(depth_at(&half, 2, 1), at(2.0), "the wall at half size (image column 5)");
}

#[test]
#[ignore = "needs a GPU"]
fn a_depth_stencil_frame_gives_its_depth_plane_as_r32() {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    let (device, context, _) = create_device(None).unwrap();
    let mut reprojector = Reprojector::new(&device).unwrap();
    let mut desc = d3d::simple_texture_desc(16, 8, DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
    desc.BindFlags = D3D11_BIND_DEPTH_STENCIL.0 as u32;
    let depth = d3d::create_texture(&device, &desc).unwrap();
    let mut view = None;
    // SAFETY: a view of our own texture into a local, and a clear through it.
    unsafe {
        device.CreateDepthStencilView(&depth, None, Some(&mut view)).unwrap();
        context.ClearDepthStencilView(view.as_ref().unwrap(), D3D11_CLEAR_DEPTH.0 | D3D11_CLEAR_STENCIL.0, 0.375, 7);
    }
    let color = texture(&device, 16, 8, DXGI_FORMAT_R8G8B8A8_UNORM, &[0; 16 * 8 * 4]);
    let camera: Mat34 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let frame = reprojector.capture(&context, &color, &depth, [0.0; 16], camera, None).unwrap();
    let plain = reprojector.plain_depth(&context, &frame).unwrap().clone();
    let plain_desc = d3d::texture_desc(&plain);
    assert_eq!((plain_desc.Width, plain_desc.Height, plain_desc.Format), (16, 8, DXGI_FORMAT_R32_FLOAT));
    let (_, _, texels) = read_image(&device, &context, &plain).unwrap();
    assert!(texels.chunks(4).all(|t| f32::from_le_bytes(t.try_into().unwrap()) == 0.375));
}
