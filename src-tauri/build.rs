use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    // Build the tauri app
    tauri_build::build();

    // Create bindings for llama.cpp

    // Build llama.cpp
    let dst = cmake::Config::new("llama_cpp")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_AVX2", "ON")
        .define("GGML_FMA", "ON")
        .define("GGML_F16C", "ON")
        .define("GGML_SSE3", "ON")
        .define("GGML_SSSE", "ON")
        .define("LLAMA_BUILD_TOOLS", "ON")
        .define("CMAKE_BUILD_RPATH", "$ORIGIN")
        .define("CMAKE_INSTALL_RPATH", "$ORIGIN")
        .profile("Release")
        .build();

    // Link the files
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib64").display()
    );
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
    bindings
        .write_to_file(out.join("llama_bindings.rs"))
        .unwrap();

    #[cfg(target_os = "windows")]
    {
        let profile = env::var("PROFILE").unwrap(); // debug or release
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

        let target_dir = manifest_dir.join("target").join(&profile);

        let lib_dir = dst
            .join("bin")
            .exists()
            .then(|| dst.join("bin"))
            .unwrap_or_else(|| dst.join("lib"));

        copy_dlls(&lib_dir, &target_dir);
    }

    #[cfg(target_os = "linux")]
    {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let stage_dir = manifest_dir.join("target/llama-libs");
        let lib_dir = dst.join("lib");
        let lib64_dir = dst.join("lib64");
        let bin_dir = dst.join("bin");

        fs::create_dir_all(&stage_dir).expect("Failed to create staged llama lib directory");
        for (prefix, soname) in [
            ("libllama.so", "libllama.so.0"),
            ("libmtmd.so", "libmtmd.so.0"),
            ("libggml.so", "libggml.so.0"),
            ("libggml-base.so", "libggml-base.so.0"),
            ("libggml-cpu.so", "libggml-cpu.so.0"),
        ] {
            stage_runtime_lib(
                &lib_dir,
                &lib64_dir,
                &bin_dir,
                prefix,
                &stage_dir.join(soname),
            );
        }

        // Make the packaged binary search for private libs under /usr/lib/<appname>.
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/visual-contextualizer");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib64/visual-contextualizer");
    }

    println!("cargo:rerun-if-changed=llama_cpp/include/llama.h");
    println!("cargo:rerun-if-changed=llama_cpp/tools/mtmd/mtmd.h");
    println!("cargo:rerun-if-changed=llama_cpp/tools/mtmd/mtmd-helper.h");
}

#[cfg(target_os = "linux")]
fn stage_runtime_lib(
    primary_dir: &Path,
    secondary_dir: &Path,
    fallback_dir: &Path,
    prefix: &str,
    dest: &Path,
) {
    let source = find_lib_with_prefix(primary_dir, prefix)
        .or_else(|| find_lib_with_prefix(secondary_dir, prefix))
        .or_else(|| find_lib_with_prefix(fallback_dir, prefix))
        .unwrap_or_else(|| panic!("Failed to locate runtime library with prefix {prefix}"));

    let _ = fs::remove_file(dest);
    fs::copy(&source, dest).unwrap_or_else(|e| {
        panic!(
            "Failed to stage {} from {:?} to {:?}: {e}",
            prefix, source, dest
        )
    });
}

#[cfg(target_os = "linux")]
fn find_lib_with_prefix(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| name.starts_with(prefix))
                    .unwrap_or(false)
        })
        .collect();

    candidates.sort();
    candidates.pop()
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
