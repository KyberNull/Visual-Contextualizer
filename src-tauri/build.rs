use std::{env, fs};
use std::path::{Path, PathBuf};

fn main() {
    // Build the tauri app
    tauri_build::build();

    // Create bindings for llama.cpp

    // Build llama.cpp
    let dst = cmake::Config::new("llama_cpp")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("GGML_NATIVE", "ON")
        .define("LLAMA_BUILD_TOOLS", "ON")
        .profile("Release")
        .build();

    // Link the files
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-lib=mtmd");

    // Create rust bindings
    let bindings = bindgen::Builder::default()
        .header("llama_cpp/include/llama.h")
        .header("llama_cpp/tools/mtmd/mtmd.h")
        .header("llama_cpp/tools/mtmd/mtmd-helper.h")
        .clang_arg(format!("-I{}", dst.join("include").display()))
        .clang_arg(format!("-I{}", dst.join("tools/mtmd").display()))
        .generate()
        .expect("Unable to generate bindings");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("llama_bindings.rs")).unwrap();

    #[cfg(target_os = "windows")]
    {
        let profile = env::var("PROFILE").unwrap(); // debug or release
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

        let target_dir = manifest_dir
            .join("target")
            .join(&profile);

        let lib_dir = dst.join("bin").exists()
            .then(|| dst.join("bin"))
            .unwrap_or_else(|| dst.join("lib"));

        copy_dlls(&lib_dir, &target_dir);
    }

    println!("cargo:rerun-if-changed=llama_cpp/include/llama.h");
    println!("cargo:rerun-if-changed=llama_cpp/tools/mtmd/mtmd.h");
    println!("cargo:rerun-if-changed=llama_cpp/tools/mtmd/mtmd-helper.h");
}

#[cfg(target_os = "windows")]
fn copy_dlls(from: &Path, to: &Path) {
    if let Ok(entries) = fs::read_dir(from) {
        for entry in entries.flatten() {
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if ext == "dll" {
                    let filename = path.file_name().unwrap();
                    let dest = to.join(filename);

                    let _ = fs::copy(&path, &dest);
                    println!("Copied {:?} -> {:?}", path, dest);
                }
            }
        }
    }
}