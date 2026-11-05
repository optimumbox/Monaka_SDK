// Reprojection: a rendered frame (colour at its own size, depth at the render size, from its own
// camera and projection) redrawn from another camera, the same on D3D11 (`reproject.rs`) and D3D12
// (`reproject12.rs`); each defines IMAGE, the type its images are written as, before this source.
// Each source colour pixel takes its depth at the same place of the depth image (UV scaled onto the
// part of it that lines up with the image, from its origin, moved by the jitter the depth was
// rendered with): the nearest of the four depth texels around that place, so a colour pixel on the
// blended edge of a near thing (the colour is finer than the depth) moves with that thing and not
// with what is behind it, which would draw its outline where the thing was. It goes through one
// matrix (target projection * target view * source camera * inverse source projection) to its
// target pixel; the nearest wins a depth test (Splat keys, then Paint the winners). A key is the
// nearness and, below it, how close the pixel landed to the target pixel's centre, so of pixels at
// one depth exactly one wins, the same one every frame. Two layers are made this way: the primary (the
// real frame of this moment, from the other eye) and the fill (this eye's own previous real frame).
// Whether a fill pixel may be trusted is asked of the primary's eye itself (Seen): the fill's point,
// taken back into the primary's frame (Back, at the primary's depth area PrimaryDepth), is either
// hidden there (that eye sees something nearer at its place: it could not see the point, so the
// fill alone knows it), seen past (that eye sees farther there: the point is no longer there, a hand
// or what it holds that has moved since the fill's frame), or outside that eye's view.
// Resolve takes the primary, or the fill where the fill is nearer and hidden from the primary's eye
// (what that eye saw beside a surface it could not see slid into the surface's place: the far wall
// seen through a bar where an arrow hid the bar), then the primary's own surface across a crack in
// it (Crack), then the primary's surface across a hole in a near thing drawn much larger, where the
// fill only knows what was behind it before it moved there (Hole: a held gun swung, which the fill
// would show the wall through), then the fill where the primary left a gap unless the fill's point
// was seen past, then
// the surface behind the gap, the farther of its horizontal neighbours (Behind). With FillFirst, a
// pixel where the fill shows the same surface as the primary (their
// nearness agrees) takes the fill: this eye's own last real frame, as the eye rendered it, so a real
// frame and a warped one alternate without a blink; the primary stands where something moved or was
// uncovered. ResolveDepth does the same with the keys alone, at the output depth's own size, and
// writes the target's depth in its own convention (depth = (nearness - NearB) / NearA).
// The output's alpha says where each pixel came from, in thirds (a 2-bit alpha holds it too): 1 the
// primary, 2/3 the fill, 1/3 the primary's own neighbour (across a crack or behind a gap), 0
// nothing warped. Where nothing warped covers a pixel, it shows this eye's own last real frame as
// that eye rendered it, unwarped, at the same place (HeldEye: the held eye's previous image, alpha
// 0 still), black only without one; ResolveDepth writes that frame's depth there (the farthest
// without one). The headset ignores alpha; captures read it. The resolves read the primary's own
// colour at Color (t0), to tell a fill point gone since its frame (Gone), and draw the primary's
// colour from there at the sub-pixel place each target pixel shows (PrimaryColour), not the nearest
// pixel Paint wrote.
cbuffer Reproject : register(b0) {
    row_major float4x4 M;
    uint W, H, SW, SH;
    uint DW, DH, OW, OH;  // the source depth's area that lines up with the image, the output depth's
    float NearA, NearB;   // target nearness = NearA * depth + NearB
    float2 Jitter;        // the source depth's jitter, in its pixels (the content moved by it)
    uint2 DepthOrigin;    // where the area starts in the source depth and in the output depth
    uint FillFirst;       // the fill where it shows the primary's surface (see above)
    uint Sources;         // a debug view instead of the image (see Shown)
    row_major float4x4 Back;  // resolves: the target's clip -> the primary's (Seen)
    uint4 PrimaryDepth;       // resolves: the primary's depth area in Depth (origin, size; size 0: none)
    float HeldNearness;       // resolves: the fill is never taken nearer than this (0: none; see Held)
    uint3 Pad;
    uint4 HeldArea;           // resolves: the held eye's own last frame's depth area in HeldDepth (origin, size; size 0: none)
};
#ifdef HELD_MOTION
// The fill's held pixels (nearer than HeldNearness) land by MHeld, from where its camera would have
// been had the play space then been where it is now (the hands move with it, not the world):
// HeldMoves in the fill's splats, HeldTrusted in the resolves (see Held).
cbuffer HeldMotion : register(b1) {
    row_major float4x4 MHeld;
    uint HeldMoves;
    uint HeldTrusted;
    uint2 HeldPad;
};
#else
static const uint HeldMoves = 0;
static const uint HeldTrusted = 0;
#endif
Texture2D<float4> Color : register(t0);
Texture2D<float> Depth : register(t1);
Texture2D<float4> Primary : register(t2);
Texture2D<float4> Fill : register(t3);
Texture2D<uint> PrimaryKeys : register(t4);
Texture2D<uint> FillKeys : register(t5);
Texture2D<float4> HeldColor : register(t6);  // resolves: this eye's own last real frame, unwarped (the fill's source)
Texture2D<float> HeldDepth : register(t7);   // resolves: its depth
RWTexture2D<uint> Keys : register(u0);
RWTexture2D<IMAGE> Image : register(u1);
RWTexture2D<IMAGE> Out : register(u2);
RWTexture2D<float> OutDepth : register(u3);

// A key: 22 bits of nearness over 10 bits of closeness to the target pixel's centre, plus one.
static const float NEARNESS_STEPS = 4194302.0;
static const uint CLOSENESS_BITS = 10;
static const float CLOSENESS_STEPS = 1023.0;

// Of the four depth texels around `at` (in texels of the area at DepthOrigin of DW x DH), the
// nearest: a blended edge pixel goes with the near surface.
float NearestDepth(float2 at) {
    int2 base = int2(floor(at - 0.5));
    float best = 0; float nearest = -1;
    [unroll] for (int j = 0; j < 2; ++j) {
        [unroll] for (int i = 0; i < 2; ++i) {
            int2 texel = clamp(base + int2(i, j), int2(0, 0), int2(DW, DH) - 1) + int2(DepthOrigin);
            float d = Depth.Load(int3(texel, 0));
            float n = NearA * d + NearB;
            if (n > nearest) { nearest = n; best = d; }
        }
    }
    return best;
}

[numthreads(8, 8, 1)] void Clear(uint3 id : SV_DispatchThreadID) {
    if (id.x < W && id.y < H) { Keys[id.xy] = 0; Image[id.xy] = float4(0, 0, 0, 0); }
}

// Where source pixel `p` lands in the target, and its key (larger is nearer; 0 means none).
bool Land(uint2 p, out uint2 target, out uint key) {
    target = uint2(0, 0); key = 0;
    float2 at = (float2(p) + 0.5) * float2(DW, DH) / float2(SW, SH) + Jitter;
    float d = NearestDepth(at);
    float2 ndc = float2((p.x + 0.5) / SW * 2 - 1, 1 - (p.y + 0.5) / SH * 2);
    float4 clip = mul(M, float4(ndc, d, 1));
#ifdef HELD_MOTION
    if (HeldMoves && HeldNearness > 0 && NearA * d + NearB > HeldNearness) clip = mul(MHeld, float4(ndc, d, 1));
#endif
    if (!(clip.w > 0)) return false;
    float3 t = clip.xyz / clip.w;
    float2 pixel = float2((t.x + 1) * 0.5 * W, (1 - t.y) * 0.5 * H);
    if (!(pixel.x >= 0 && pixel.y >= 0 && pixel.x < W && pixel.y < H)) return false;
    float nearness = saturate(NearA * t.z + NearB);
    target = uint2(pixel);
    float2 off = pixel - (floor(pixel) + 0.5);
    uint closeness = uint((1 - saturate(dot(off, off) * 2)) * CLOSENESS_STEPS);
    key = ((uint(nearness * NEARNESS_STEPS) << CLOSENESS_BITS) | closeness) + 1;
    return true;
}

[numthreads(8, 8, 1)] void Splat(uint3 id : SV_DispatchThreadID) {
    if (id.x >= SW || id.y >= SH) return;
    uint2 target; uint key;
    if (Land(id.xy, target, key)) InterlockedMax(Keys[target], key);
}

[numthreads(8, 8, 1)] void Paint(uint3 id : SV_DispatchThreadID) {
    if (id.x >= SW || id.y >= SH) return;
    uint2 target; uint key;
    if (Land(id.xy, target, key) && Keys[target] == key) Image[target] = float4(Color.Load(int3(id.xy, 0)).rgb, 1);
}

// A key's nearness (0 for none).
uint Nearness(uint key) { return key ? (key - 1) >> CLOSENESS_BITS : 0; }

// Whether two keys show one surface: their nearness within a fiftieth.
bool SameSurface(uint a, uint b) {
    uint na = Nearness(a), nb = Nearness(b);
    uint far = min(na, nb), near = max(na, nb);
    return near - far <= near / 50;
}

// The surface behind a gap at target pixel `p`: of the first pixels the primary saw left and right of
// it within 64 pixels, the farther. Its column (-1: none) and its key.
int Behind(uint2 p, out uint key) {
    key = 0; int column = -1;
    [unroll] for (int side = -1; side <= 1; side += 2) {
        [loop] for (int r = 1; r <= 64; ++r) {
            int x = int(p.x) + side * r;
            if (x < 0 || x >= int(W)) break;
            uint k = PrimaryKeys.Load(int3(x, p.y, 0));
            if (k) { if (column < 0 || Nearness(k) < Nearness(key)) { key = k; column = x; } break; }
        }
    }
    return column;
}

bool Inside(int2 p) { return p.x >= 0 && p.y >= 0 && p.x < int(W) && p.y < int(H); }

// What the primary's eye saw of the fill's point at target pixel `p` (key `fill`), taken back into
// the primary's frame (see the top): HIDDEN, something nearer at its place; PAST, something farther
// (the point is not there any more); OUTSIDE, out of that eye's view or no primary depth; SAME, the
// point itself. The primary's depth is in the target's convention (both eyes share depth terms).
static const uint HIDDEN = 0, PAST = 1, OUTSIDE = 2, SAME = 3;
// SeenAt also says where in the primary's frame the point lies (its NDC in `at_ndc`), and whether
// that eye sees anything there in front of it at all (`in_front`: nearer by a sixty-fourth).
uint SeenAt(uint2 p, uint fill, out float2 at_ndc, out bool in_front) {
    at_ndc = float2(0, 0); in_front = false;
    if (PrimaryDepth.z == 0 || PrimaryDepth.w == 0) return OUTSIDE;
    float here = Nearness(fill) / NEARNESS_STEPS;
    float2 ndc = float2((p.x + 0.5) / W * 2 - 1, 1 - (p.y + 0.5) / H * 2);
    float4 s = mul(Back, float4(ndc, (here - NearB) / NearA, 1));
    if (!(s.w > 0)) return OUTSIDE;
    s /= s.w;
    at_ndc = s.xy;
    if (any(abs(s.xy) >= 1)) return OUTSIDE;
    int2 texel = clamp(int2(float2((s.x + 1) * 0.5, (1 - s.y) * 0.5) * float2(PrimaryDepth.zw)), int2(0, 0), int2(PrimaryDepth.zw) - 1) + int2(PrimaryDepth.xy);
    float seen = NearA * Depth.Load(int3(texel, 0)) + NearB, at = NearA * s.z + NearB;
    in_front = seen > at + at / 64;
    if (seen > at + at / 8) return HIDDEN;
    if (seen + at / 8 < at) return PAST;
    return SAME;
}
uint Seen(uint2 p, uint fill) { float2 at; bool in_front; return SeenAt(p, fill, at, in_front); }

// How far apart in colour (the largest channel) the fill's point and the primary eye's view of it
// may be and still be one thing.
static const float RECOLOURED = 0.25;

// Whether the fill's point at `p` (key `fill`), which the primary's eye sees at about its depth
// (SAME) with nothing in front of it, is shown there in another colour by every primary pixel
// around it: the point has gone
// since the fill's frame (where a walking character or the swung gun was 22 ms ago, a wall a hand's
// breadth behind it matching its depth). The depth alone lets the old place through as a sliver
// beside the thing, on alternate eyes each frame: a flicker. Where that eye sees something in
// front of the point (the thing beside the gap, a little nearer than the wall it uncovers) the
// colours differ as they should: never then, nor without the primary's colour (t0 unbound).
bool Gone(uint2 p, uint fill) {
    float2 at; bool in_front;
    if (SeenAt(p, fill, at, in_front) != SAME || in_front) return false;
    uint cw, ch;
    Color.GetDimensions(cw, ch);
    if (cw == 0 || ch == 0) return false;
    int2 c = int2(float2((at.x + 1) * 0.5, (1 - at.y) * 0.5) * float2(cw, ch));
    float3 mine = Fill.Load(int3(p, 0)).rgb;
    [loop] for (int y = -1; y <= 1; ++y) {
        [loop] for (int x = -1; x <= 1; ++x) {
            int2 q = clamp(c + int2(x, y), int2(0, 0), int2(cw, ch) - 1);
            float3 d = abs(Color.Load(int3(q, 0)).rgb - mine);
            if (max(d.r, max(d.g, d.b)) <= RECOLOURED) return false;
        }
    }
    return true;
}

// Whether the fill's colour at target pixel `p` is within RECOLOURED of the primary's there or at
// one of its neighbours (pixels the primary landed on): one thing seen by both, its shading as the
// eye itself saw it, rather than something that moved and left its depth behind.
bool Agrees(uint2 p) {
    float3 mine = Fill.Load(int3(p, 0)).rgb;
    [loop] for (int y = -1; y <= 1; ++y) {
        [loop] for (int x = -1; x <= 1; ++x) {
            int2 q = int2(p) + int2(x, y);
            if (!Inside(q) || !PrimaryKeys.Load(int3(q, 0))) continue;
            float3 d = abs(Primary.Load(int3(q, 0)).rgb - mine);
            if (max(d.r, max(d.g, d.b)) <= RECOLOURED) return true;
        }
    }
    return false;
}

// A crack: a target pixel no source pixel landed on inside one of the primary's surfaces (a surface
// drawn larger than in its source, nearer or turned toward this eye, leaves such pixels in a grid).
// Told from a gap beside a nearer surface by primary pixels at about the same depth on both sides of
// it within `reach` pixels, along a row, a column or a diagonal; a gap has the nearer surface on one
// side only. The nearest such pixel, or (-1, -1).
int2 CrackWithin(uint2 p, int reach) {
    const int2 dirs[4] = { int2(1, 0), int2(0, 1), int2(1, 1), int2(1, -1) };
    uint best_near = 0; int2 best = int2(-1, -1);
    [unroll] for (int d = 0; d < 4; ++d) {
        uint ka = 0, kb = 0; int2 a = int2(-1, -1), b = int2(-1, -1);
        [loop] for (int r = 1; r <= reach; ++r) {
            int2 q = int2(p) + dirs[d] * r, s = int2(p) - dirs[d] * r;
            if (!ka && Inside(q)) { ka = PrimaryKeys.Load(int3(q, 0)); a = q; }
            if (!kb && Inside(s)) { kb = PrimaryKeys.Load(int3(s, 0)); b = s; }
        }
        uint na = Nearness(ka), nb = Nearness(kb);
        uint nearer = max(na, nb), farther = min(na, nb);
        if (ka && kb && farther >= nearer - nearer / 5 && nearer > best_near) { best_near = nearer; best = na >= nb ? a : b; }
    }
    return best;
}

// How far a crack in a held thing (below) is looked across, each way.
static const int HELD_CRACK_REACH = 8;

// A crack (above) within two pixels anywhere; within the hands' reach (HeldNearness) up to
// HELD_CRACK_REACH: a held gun a hand's length from the eyes is drawn much larger in the other eye,
// its landed pixels several apart, and a gap with that surface on both sides is inside it (not
// beside it), so it takes the surface rather than what is behind (lines drawn across the gun).
int2 Crack(uint2 p) {
    int2 crack = CrackWithin(p, 2);
    if (crack.x >= 0 || HeldNearness <= 0) return crack;
    crack = CrackWithin(p, HELD_CRACK_REACH);
    if (crack.x >= 0 && Nearness(PrimaryKeys.Load(int3(crack, 0))) / NEARNESS_STEPS > HeldNearness) return crack;
    return int2(-1, -1);
}

// How far a hole (below) is looked across, each way.
static const int HOLE_REACH = 8;

// The nearest pixel of `keys` within HOLE_REACH of `p` at nearness `near` or nearer (less a fifth):
// whether that image shows something about as near as `near` around `p`.
bool NearAround(Texture2D<uint> keys, uint2 p, uint near) {
    [loop] for (int y = -HOLE_REACH; y <= HOLE_REACH; y += 2) {
        [loop] for (int x = -HOLE_REACH; x <= HOLE_REACH; x += 2) {
            int2 q = int2(p) + int2(x, y);
            if (Inside(q) && Nearness(keys.Load(int3(q, 0))) >= near - near / 5) return true;
        }
    }
    return false;
}

// A hole: a gap at target pixel `p` with primary pixels all round it within HOLE_REACH (in each of
// the four quadrants about it), a near surface drawn so much larger than in its source (a held gun
// close to the eyes) that its landed pixels are too far apart for a crack, where the fill (key
// `fill`) shows something clearly behind all of them and nothing as near anywhere around: what that
// frame saw before the near thing moved there, which would show through it. Of the quadrants'
// closest pixels, the farthest (the surface behind the gap, should it lie between near things);
// (-1, -1) for none, and its key in `key`.
int2 Hole(uint2 p, uint fill, out uint key) {
    key = 0;
    if (!fill) return int2(-1, -1);
    int2 closest[4] = { int2(-1, -1), int2(-1, -1), int2(-1, -1), int2(-1, -1) };
    int reach[4] = { 1 << 30, 1 << 30, 1 << 30, 1 << 30 };
    uint keys[4] = { 0, 0, 0, 0 };
    [loop] for (int y = -HOLE_REACH; y <= HOLE_REACH; ++y) {
        [loop] for (int x = -HOLE_REACH; x <= HOLE_REACH; ++x) {
            int2 q = int2(p) + int2(x, y);
            if ((x == 0 && y == 0) || !Inside(q)) continue;
            uint k = PrimaryKeys.Load(int3(q, 0));
            if (!k) continue;
            // Quadrants turning about p, each with one half-axis.
            uint quadrant = x > 0 && y >= 0 ? 0 : x <= 0 && y > 0 ? 1 : x < 0 && y <= 0 ? 2 : 3;
            int d = x * x + y * y;
            if (d < reach[quadrant]) { reach[quadrant] = d; closest[quadrant] = q; keys[quadrant] = k; }
        }
    }
    uint farthest = 0xffffffff; int2 best = int2(-1, -1);
    [unroll] for (int i = 0; i < 4; ++i) {
        if (!keys[i]) return int2(-1, -1);
        if (Nearness(keys[i]) < farthest) { farthest = Nearness(keys[i]); best = closest[i]; key = keys[i]; }
    }
    if (Nearness(fill) + farthest / 8 >= farthest || NearAround(FillKeys, p, farthest)) { key = 0; return int2(-1, -1); }
    return best;
}

// What a pixel shows, its colour `color` and key `key` taken from `from` (0 the primary, 1 the fill,
// 2 a neighbour of the primary's, 3 nothing). Sources 1 tints it by where it came from: the primary
// as it is, the fill green, a neighbour solid blue (no game colour matches it exactly), nothing
// magenta. Sources 2 shows its nearness as grey (white near, a square root so far things still
// differ), the colour gone. Sources 3 shows the fill alone and 4 the primary alone (what the other
// left, magenta).
// 4 is the held eye's own last frame unwarped where nothing warped landed (HeldEye): alpha 0 as
// nothing, tinted yellow by Sources 1 (by half, the frame still seen through it).
float4 Shown(float4 color, uint key, uint from) {
    float source = from >= 3 ? 0.0 : (3 - from) / 3.0;  // the alpha (see the top)
    if (Sources == 1) {
        const float3 tints[5] = { float3(0, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 0, 1), float3(1, 1, 0) };
        const float amounts[5] = { 0.0, 0.6, 1.0, 1.0, 0.6 };
        return float4(lerp(color.rgb, tints[from], amounts[from]), source);
    }
    if (Sources == 2) return float4(sqrt(key ? ((key - 1) >> CLOSENESS_BITS) / NEARNESS_STEPS : 0).xxx, source);
    return float4(color.rgb, source);
}

// Whether the fill's pixel `fill` is within the hands' reach (HeldNearness) and not to be taken: the
// hands and what they hold move with the play space and not with the world, so unless they landed
// by its motion (HeldTrusted) the eye's own last frame shows them where they were. A gap there then
// shows what is behind it instead.
bool Held(uint fill) {
    return !HeldTrusted && HeldNearness > 0 && Nearness(fill) / NEARNESS_STEPS > HeldNearness;
}

// Whether the fill's pixel `fill` is a held thing moved with the play space (HeldMoves) though not
// trusted everywhere (Held): taken only inside the primary's own held surface (HeldInside).
bool HeldMoved(uint fill) {
    return HeldMoves && HeldNearness > 0 && fill && Nearness(fill) / NEARNESS_STEPS > HeldNearness;
}

// How far along each axis the primary's held surface is looked for about a gap (HeldInside).
// A hand's length from the eyes their views of an arm or gun lie well over a hundred pixels apart,
// so its sides seen by one eye only can be tens of pixels wide.
static const int HELD_INSIDE_REACH = 64;

// Whether a gap at `p` lies inside the primary's held things: a held pixel (within the hands'
// reach, at any nearness: an arm runs from near to far, a hand lies behind its gun) first met both
// left and right of it, or both above and below, within HELD_INSIDE_REACH (empty pixels passed
// over, a farther one ending the look).
bool Enclosed(uint2 p) {
    bool found[4] = { false, false, false, false };
    const int2 dirs[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
    [unroll] for (int d = 0; d < 4; ++d) {
        [loop] for (int r = 1; r <= HELD_INSIDE_REACH; ++r) {
            int2 q = int2(p) + dirs[d] * r;
            if (!Inside(q)) break;
            uint k = PrimaryKeys.Load(int3(q, 0));
            if (!k) continue;
            found[d] = Nearness(k) / NEARNESS_STEPS > HeldNearness;
            break;
        }
    }
    return (found[0] && found[1]) || (found[2] && found[3]);
}

// Whether the gap at `p` takes the fill's held pixel `fill` (HeldMoved): inside the primary's held
// surface, where the eye's own last frame saw a side of a held gun this eye's warp cannot (a hand's
// length away, its sides are seen by one eye only): its real surface rather than lines of its
// neighbours drawn across. Outside it the fill's held thing stays out (a ghost of the gun where the
// hand was a frame ago).
// Outside it, too, where the primary's eye does not see the point gone (not PAST): a gun raised to
// the eyes to aim is seen down its barrel by one eye and from the side by the other, its side to
// that eye only (the warp leaves hundreds of pixels there, beyond any reach), which the eye's own
// frame alone shows. Where the hand moved since, that eye sees what was behind it: never then.
bool HeldInside(uint2 p, uint fill) {
    if (!HeldMoved(fill)) return false;
    return Enclosed(p) ? !Gone(p, fill) : Seen(p, fill) != PAST;
}

// Whether the fill's pixel at `p` (key `fill`) goes out over the primary's (key `primary`): the
// same surface with FillFirst, or a clearly nearer one the primary's eye could not see (hidden).
// A held thing of the fill (HeldMoved, not trusted everywhere) goes over a clearly farther primary
// pixel that the primary's eye does not see gone: the side of a gun aimed down, which that eye sees
// edge-on, lands in the warp as pixels far apart with what is behind showing between them.
// With FillFirst the same surface takes the fill only where its colour agrees with the primary
// around it (Agrees): a walking character's last place keeps its depth but not its colour.
bool FillOver(uint2 p, uint fill, uint primary) {
    if (!fill) return false;
    uint nf = Nearness(fill), np = Nearness(primary);
    if (Held(fill)) return HeldMoved(fill) && np + nf / 8 < nf && Seen(p, fill) != PAST;
    if (FillFirst && SameSurface(fill, primary) && Agrees(p)) return true;
    return np + nf / 8 < nf && Seen(p, fill) == HIDDEN;
}

// Whether the fill's pixel at `p` (key `fill`) fills the gap there: anything but a point the
// primary's eye saw past (gone since).
bool FillsGap(uint2 p, uint fill) {
    return fill && !Held(fill) && Seen(p, fill) != PAST && !Gone(p, fill);
}

// `image` at `at` (its texels, continuous), filtered linearly between the four texels around it.
float4 Bilinear(Texture2D<float4> image, float2 at) {
    uint w, h;
    image.GetDimensions(w, h);
    float2 f = at - 0.5;
    int2 b = int2(floor(f)), last = int2(w, h) - 1;
    float2 k = f - float2(b);
    float4 c00 = image.Load(int3(clamp(b, int2(0, 0), last), 0));
    float4 c10 = image.Load(int3(clamp(b + int2(1, 0), int2(0, 0), last), 0));
    float4 c01 = image.Load(int3(clamp(b + int2(0, 1), int2(0, 0), last), 0));
    float4 c11 = image.Load(int3(clamp(b + int2(1, 1), int2(0, 0), last), 0));
    return lerp(lerp(c00, c10, k.x), lerp(c01, c11, k.x), k.y);
}

// What target pixel `p` shows where nothing warped covers it: this eye's own last real frame
// (HeldColor, the held eye's previous image) as that eye rendered it, at the same place of its own
// image, rather than black. False without that frame (t6 unbound or empty).
bool HeldEye(uint2 p, out float4 color) {
    color = float4(0, 0, 0, 1);
    uint cw, ch;
    HeldColor.GetDimensions(cw, ch);
    if (cw == 0 || ch == 0) return false;
    color = float4(Bilinear(HeldColor, (float2(p) + 0.5) / float2(W, H) * float2(cw, ch)).rgb, 1);
    return true;
}

// That frame's depth at the same place (its depth area HeldArea), beside HeldEye's colour; false
// without it.
bool HeldEyeDepth(uint2 p, out float depth) {
    depth = 0;
    if (HeldArea.z == 0 || HeldArea.w == 0) return false;
    int2 texel = clamp(int2((float2(p) + 0.5) / float2(W, H) * float2(HeldArea.zw)), int2(0, 0), int2(HeldArea.zw) - 1) + int2(HeldArea.xy);
    depth = HeldDepth.Load(int3(texel, 0));
    return true;
}

// Sub-pixel colour. Paint writes the nearest source pixel's colour at the target pixel it landed in,
// up to half a pixel from where that target pixel's centre truly is in the source: the warped eye's
// edges crawl against the real eye's. The resolves (whose M, SW, SH and Jitter are the primary's
// splat's, its depth area PrimaryDepth) find the source place each target pixel shows: from the
// winner's depth taken back (Back), then REFINE_STEPS fixed-point steps s += t - landed(s) with the
// depth at s; its colour is read between the source pixels there (Bilinear). Across a depth edge
// (the four depth texels at s, or the point found, apart by more than REFINE_EDGE of their nearness)
// the nearest pixel stays: a blend there would mix the near thing and what is behind it.
static const int REFINE_STEPS = 3;
static const float REFINE_EDGE = 0.04;

// The primary's depth at `at` (texels of its depth area, continuous), filtered linearly, and
// whether the four texels around it are one surface (their nearness within REFINE_EDGE).
bool SmoothDepth(float2 at, out float depth) {
    float2 f = at - 0.5;
    int2 b = int2(floor(f));
    float2 k = f - float2(b);
    float d[4];
    float lo = 1e30, hi = -1e30;
    [unroll] for (int j = 0; j < 2; ++j) {
        [unroll] for (int i = 0; i < 2; ++i) {
            int2 texel = clamp(b + int2(i, j), int2(0, 0), int2(PrimaryDepth.zw) - 1) + int2(PrimaryDepth.xy);
            float v = Depth.Load(int3(texel, 0));
            d[j * 2 + i] = v;
            float n = NearA * v + NearB;
            lo = min(lo, n); hi = max(hi, n);
        }
    }
    depth = lerp(lerp(d[0], d[1], k.x), lerp(d[2], d[3], k.x), k.y);
    return hi - lo <= abs(hi) * REFINE_EDGE;
}

// Where the primary's point at source place `s` (its colour pixels, continuous) lands in the target
// (`pixel`, continuous), and its target nearness, through M with the depth there; false off one
// surface.
bool Forward(float2 s, out float2 pixel, out float nearness) {
    pixel = float2(0, 0); nearness = 0;
    float d;
    if (!SmoothDepth(s * float2(PrimaryDepth.zw) / float2(SW, SH) + Jitter, d)) return false;
    float2 ndc = float2(s.x / SW * 2 - 1, 1 - s.y / SH * 2);
    float4 clip = mul(M, float4(ndc, d, 1));
    if (!(clip.w > 0)) return false;
    float3 t = clip.xyz / clip.w;
    pixel = float2((t.x + 1) * 0.5 * W, (1 - t.y) * 0.5 * H);
    nearness = NearA * t.z + NearB;
    return true;
}

// The source place target pixel `p` (the primary's key `key` there) shows, refined (see above);
// false where it may not be: the nearest pixel then.
bool Refined(uint2 p, uint key, out float2 s) {
    s = float2(0, 0);
    if (PrimaryDepth.z == 0 || PrimaryDepth.w == 0 || SW == 0 || SH == 0) return false;
    float here = Nearness(key) / NEARNESS_STEPS;
    float2 t = float2(p) + 0.5;
    float2 ndc = float2(t.x / W * 2 - 1, 1 - t.y / H * 2);
    float4 back = mul(Back, float4(ndc, (here - NearB) / NearA, 1));
    if (!(back.w > 0)) return false;
    back /= back.w;
    s = float2((back.x + 1) * 0.5 * SW, (1 - back.y) * 0.5 * SH);
    float2 scale = float2(SW, SH) / float2(W, H), pixel;
    float nearness;
    [unroll] for (int i = 0; i < REFINE_STEPS; ++i) {
        if (!Forward(s, pixel, nearness)) return false;
        s += (t - pixel) * scale;
    }
    if (!Forward(s, pixel, nearness)) return false;
    if (any(abs(t - pixel) > 0.5) || abs(nearness - here) > abs(here) * REFINE_EDGE) return false;
    return all(s >= 0) && all(s <= float2(SW, SH));
}

// The primary's colour at target pixel `p` (its key `key`): read at the refined source place, or
// the nearest pixel Paint wrote.
float4 PrimaryColour(uint2 p, uint key) {
    uint cw, ch;
    Color.GetDimensions(cw, ch);
    float2 s;
    if (cw == SW && ch == SH && Refined(p, key, s)) return float4(Bilinear(Color, s).rgb, 1);
    return Primary.Load(int3(p, 0));
}

[numthreads(8, 8, 1)] void Resolve(uint3 id : SV_DispatchThreadID) {
    if (id.x >= W || id.y >= H) return;
    if (Sources == 3) {
        Out[id.xy] = FillKeys.Load(int3(id.xy, 0)) ? Fill.Load(int3(id.xy, 0)) : float4(1, 0, 1, 1);
        return;
    }
    if (Sources == 4) {
        Out[id.xy] = PrimaryKeys.Load(int3(id.xy, 0)) ? Primary.Load(int3(id.xy, 0)) : float4(1, 0, 1, 1);
        return;
    }
    uint primary = PrimaryKeys.Load(int3(id.xy, 0));
    if (primary) {
        uint fill = FillKeys.Load(int3(id.xy, 0));
        Out[id.xy] = FillOver(id.xy, fill, primary) ? Shown(Fill.Load(int3(id.xy, 0)), fill, 1) : Shown(PrimaryColour(id.xy, primary), primary, 0);
        return;
    }
    uint fill = FillKeys.Load(int3(id.xy, 0));
    if (HeldInside(id.xy, fill)) { Out[id.xy] = Shown(Fill.Load(int3(id.xy, 0)), fill, 1); return; }
    int2 crack = Crack(id.xy);
    if (crack.x >= 0) { Out[id.xy] = Shown(Primary.Load(int3(crack, 0)), PrimaryKeys.Load(int3(crack, 0)), 2); return; }
    uint hole_key; int2 hole = Hole(id.xy, fill, hole_key);
    if (hole.x >= 0) { Out[id.xy] = Shown(Primary.Load(int3(hole, 0)), hole_key, 2); return; }
    if (FillsGap(id.xy, fill)) { Out[id.xy] = Shown(Fill.Load(int3(id.xy, 0)), fill, 1); return; }
    uint behind; int column = Behind(id.xy, behind);
    if (column >= 0) { Out[id.xy] = Shown(Primary.Load(int3(column, id.y, 0)), behind, 2); return; }
    float4 held;
    Out[id.xy] = HeldEye(id.xy, held) ? Shown(held, 0, 4) : Shown(float4(0, 0, 0, 1), 0, 3);
}

[numthreads(8, 8, 1)] void ResolveDepth(uint3 id : SV_DispatchThreadID) {
    if (id.x >= OW || id.y >= OH) return;
    uint2 p = uint2(min((id.x * 2 + 1) * W / (2 * OW), W - 1), min((id.y * 2 + 1) * H / (2 * OH), H - 1));
    uint key = PrimaryKeys.Load(int3(p, 0));
    uint fill = FillKeys.Load(int3(p, 0));
    if (key && FillOver(p, fill, key)) key = fill;
    if (!key && HeldInside(p, fill)) key = fill;
    if (!key) {
        int2 crack = Crack(p);
        if (crack.x >= 0) key = PrimaryKeys.Load(int3(crack, 0));
    }
    if (!key) {
        uint hole_key; int2 hole = Hole(p, fill, hole_key);
        if (hole.x >= 0) key = hole_key;
    }
    if (!key) {
        uint behind; int column = Behind(p, behind);
        key = FillsGap(p, fill) ? fill : column >= 0 ? behind : 0;
    }
    // Nothing warped: the held eye's own last frame's depth where its colour shows (HeldEye), else
    // the farthest.
    float held;
    if (!key) { OutDepth[id.xy + DepthOrigin] = HeldEyeDepth(p, held) ? held : -NearB / NearA; return; }
    float nearness = ((key - 1) >> CLOSENESS_BITS) / NEARNESS_STEPS;
    OutDepth[id.xy + DepthOrigin] = (nearness - NearB) / NearA;
}
