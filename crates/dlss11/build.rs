//! Links NVIDIA's NGX SDK library (`ngx/`): its static-CRT build when the C runtime is linked in
//! (`+crt-static`, the release builds), its dynamic-CRT build otherwise (tests, everyday builds).

fn main() {
    let dir = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo")).join("ngx");
    let static_crt = std::env::var("CARGO_CFG_TARGET_FEATURE").is_ok_and(|f| f.split(',').any(|f| f == "crt-static"));
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static={}", if static_crt { "nvsdk_ngx_s" } else { "nvsdk_ngx_d" });
    // The registry (where the driver keeps NGX) and the window of the log name.
    println!("cargo:rustc-link-lib=advapi32");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rerun-if-changed=build.rs");
}
