// Motion vectors for temporal upscaling, from depth: each pixel goes through one matrix (previous
// projection * previous view * this camera * inverse of this projection, all of one eye) to where
// it was in that eye's previous frame. A far depth of 0 (reversed infinite) unprojects to a
// direction, so the sky moves with the rotation only. A game's own object motion (what moved in
// the world, the camera left out) is added scaled to UV units. Out: UV-space vectors from this
// frame to the previous one (y down), and the depth as R32 for the upscaler.
// With a mask asked for (Flags bit 0), where each pixel moved on its own: its point taken to the
// eye's previous frame by the camera alone lands on nothing that far away there (the previous
// depth, Flags bit 1, compared as distance along the view, the nearest of 3x3 texels), so the
// camera's motion vector is wrong for it (an enemy walking, the gun in a moving hand). 1 there,
// 0 where the camera's motion explains it; an upscaler then takes such pixels from this frame
// rather than smearing its history over them. Only what is nearer now than what was there: a
// surface revealed from behind something (the far side of every edge while walking) is the
// upscaler's own disocclusion to handle, and marking it showed a tenth to a fifth of each eye at
// the render size whenever the player walked. Everything within the hands' reach (Held) is marked
// too: a gun moved across its own surface keeps its depth at the old place, so the test misses it.
// With the hands' items given (where each was this frame and the eye's previous one), what is within
// reach moves with the nearer item instead of the camera: its motion vector follows the hand, the
// depth test checks it like any other, and an upscaler keeps its history (the gun's sights sharp).
cbuffer Motion : register(b0) {
    row_major float4x4 M;
    uint W, H;                // render size (the vectors' and depth output's)
    uint DW, DH;              // the depth texture's size
    float2 ObjectScale;       // object motion to UV units (0: none)
    uint HasObject;
    uint Flags;               // bit 0: write the mask; bit 1: the previous depth is there
    float4 Linear;            // the previous projection's z row terms: P[2][2], P[2][3], P[3][2], P[3][3]
    float4 Held;              // the depth the hands' reach away, and which way is nearer (y: 1, -1; 0 none)
    row_major float4x4 Unproject; // this frame's NDC and depth to the world
    row_major float4x4 Item0;     // the world now to the eye's previous clip space, moved with each hand's item
    row_major float4x4 Item1;
    float4 ItemAt0, ItemAt1;      // where each hand's item is now (w: 1; 0 none)
};
// How far from an item's origin (its grip) what moves with it reaches, metres.
static const float ItemReach = 0.7;
Texture2D<float> Depth : register(t0);
Texture2D<float2> Object : register(t1);
RWTexture2D<float2> Motion : register(u0);
RWTexture2D<float> DepthOut : register(u1);
Texture2D<float> PreviousDepth : register(t2);
RWTexture2D<float> Moved : register(u2);

// Distance along the view of a depth `d` of the previous projection (huge for the far plane).
float Along(float d) {
    float below = d * Linear.z - Linear.x;
    if (abs(below) < 1e-9) return 1e9;
    return min(abs((Linear.y - d * Linear.w) / below), 1e9);
}

[numthreads(8, 8, 1)] void Main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= W || id.y >= H) return;
    float d = Depth.Load(int3(id.x * DW / W, id.y * DH / H, 0));
    float2 ndc = float2((id.x + 0.5) / W * 2 - 1, 1 - (id.y + 0.5) / H * 2);
    float4 previous = mul(M, float4(ndc, d, 1));
    bool held = Held.y != 0 && Held.y * (d - Held.x) > 0;
    bool carried = false;
    if (held && (ItemAt0.w > 0 || ItemAt1.w > 0)) {
        float4 world = mul(Unproject, float4(ndc, d, 1));
        if (abs(world.w) > 1e-12) {
            float3 at = world.xyz / world.w;
            float far0 = ItemAt0.w > 0 ? distance(at, ItemAt0.xyz) : 1e30;
            float far1 = ItemAt1.w > 0 ? distance(at, ItemAt1.xyz) : 1e30;
            // An item's own reach: past it (a free hand far from the gun) the pixel stays masked.
            if (min(far0, far1) < ItemReach) {
                previous = mul(far0 <= far1 ? Item0 : Item1, float4(at, 1));
                carried = true;
            }
        }
    }
    float2 mv = float2(0, 0);
    if (previous.w > 1e-6) {
        float2 p = previous.xy / previous.w;
        mv = float2(p.x - ndc.x, ndc.y - p.y) * 0.5;
    }
    if (HasObject) mv += Object.Load(int3(id.xy, 0)) * ObjectScale;
    Motion[id.xy] = mv;
    DepthOut[id.xy] = d;
    if (Flags & 1) {
        float moved = 0;
        if ((Flags & 2) && previous.w > 1e-6) {
            float2 p = previous.xy / previous.w;
            if (all(abs(p) < 1)) {
                float expect = Along(previous.z / previous.w);
                int2 at = int2((p.x * 0.5 + 0.5) * W, (0.5 - p.y * 0.5) * H);
                float best = 1e9;
                bool nearer = false;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int2 q = clamp(at + int2(dx, dy), int2(0, 0), int2(W - 1, H - 1));
                        float was = Along(PreviousDepth.Load(int3(q, 0)));
                        float off = (was > 1e8 && expect > 1e8) ? 0 : abs(was - expect) / max(min(was, expect), 1e-6);
                        if (off < best) { best = off; nearer = was > expect; }
                    }
                }
                moved = nearer ? saturate((best - 0.04) / 0.08) : 0;
            }
        }
        // Within the hands' reach without the items' motion: moved with the controllers, whatever
        // the depth test says.
        if (held && !carried) moved = 1;
        Moved[id.xy] = moved;
    }
}
