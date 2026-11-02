NVIDIA's DLSS SDK files the crate builds and ships with, copied here from the DLSS SDK
(https://github.com/NVIDIA/DLSS, SDK API 1.5) and not kept in git:

- `nvsdk_ngx_s.lib`: NGX, static C runtime (release builds, `+crt-static`);
- `nvsdk_ngx_d.lib`: NGX, dynamic C runtime (tests and everyday builds);
- `nvngx_dlss.dll`: the DLSS model (310.x), shipped beside a game pack's DLL.

Their licence is `NVIDIA-DLSS-LICENSE.txt`.
