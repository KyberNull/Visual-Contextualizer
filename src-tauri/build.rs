use std::env;
use std::path::PathBuf;

fn main() {
    tauri_build::build();

    let dst = cmake::Config::new("llama_cpp")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=llama");

    let bindings = bindgen::Builder::default()
        .header("llama_cpp/include/llama.h")
        .clang_arg(format!("-I{}", dst.join("include").display()))
        .generate()
        .expect("Unable to generate bindings");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("bindings.rs")).unwrap();
}
