// AMD FidelityFX CAS (MIT): contrast-adaptive sharpening, the no-scaling path of ffx_cas.h with
// exact reciprocals. Works on the stored (display-encoded) values, as the upscaler wrote them.

cbuffer Sharpen : register(b0) {
    float Peak;   // -1 / lerp(8, 5, sharpness)
    float3 Padding;
    uint2 Size;
    uint2 Padding2;
};

Texture2D<float4> Source : register(t0);
RWTexture2D<float4> Target : register(u0);

float3 Tap(int2 p) { return Source.Load(int3(clamp(p, int2(0, 0), int2(Size) - 1), 0)).rgb; }

[numthreads(8, 8, 1)]
void Main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Size.x || id.y >= Size.y) return;
    const int2 p = int2(id.xy);
    const float3 a = Tap(p + int2(-1, -1)), b = Tap(p + int2(0, -1)), c = Tap(p + int2(1, -1));
    const float3 d = Tap(p + int2(-1, 0));
    const float4 centre = Source.Load(int3(p, 0));
    const float3 e = centre.rgb;
    const float3 f = Tap(p + int2(1, 0));
    const float3 g = Tap(p + int2(-1, 1)), h = Tap(p + int2(0, 1)), i = Tap(p + int2(1, 1));
    // The cross's minimum and maximum, plus the whole 3x3's (soft, as CAS does).
    float3 mn = min(min(min(d, e), min(f, b)), h);
    const float3 mn2 = min(min(mn, a), min(min(c, g), i));
    mn += mn2;
    float3 mx = max(max(max(d, e), max(f, b)), h);
    const float3 mx2 = max(max(mx, a), max(max(c, g), i));
    mx += mx2;
    // Less sharpening where the neighbourhood already spans the range.
    float3 amp = saturate(min(mn, 2.0 - mx) / max(mx, 1e-5));
    amp = sqrt(amp);
    const float3 w = amp * Peak;
    const float3 weight = 1.0 / (1.0 + 4.0 * w);
    Target[id.xy] = float4(saturate((b * w + d * w + f * w + h * w + e) * weight), centre.a);
}
