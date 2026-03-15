# Visual Contextualizer

Visual Contextualizer is a desktop-first accessibility app that turns visual content into clear, naturally spoken explanations. It is designed to help users quickly understand charts, interfaces, and other visual artifacts through concise, context-aware narration.

## Why This Project Stands Out

- Fast, native-grade inference workflow powered by a Rust backend. (We painstakingly wrote a custom wrapper of llama.cpp for this performance on such a heavy model.)
- Cross-platform desktop delivery with Tauri. (Can be extended in the future to mobile and other OSes)
- Lightweight, responsive interface built with vanilla JavaScript.
- Smooth dark/light mode experience for comfortable use in any environment. (Omkar added it, don't ask me why)

## Core Stack

- Backend: Rust
- Desktop runtime: Tauri
- Frontend: HTML, CSS, vanilla JavaScript
- llama.cpp as the engine for VLM, piper.rs for TTS

## Architecture Snapshot

The app uses a Rust Foreign function Interface (FFI) with llama.cpp for model orchestration, token generation. Tauri bridges that backend with a clean web UI, where vanilla JavaScript handles interactions and real-time updates via event-driven streaming. The result is a minimal, efficient desktop app with modern usability.

## User Experience Highlights

- Trigers 
- Real-time streaming output for immediate feedback and TTS
- Clean interface optimized for readability and narration-first flow.
- Works with both upload button and Global Hotkey.
- Focus on practical clarity over visual noise.
- Model understands upto 201 languages.

## Summary

Visual Contextualizer combines Rust performance, Tauri portability, and vanilla JavaScript simplicity into a focused accessibility tool. It is built to be fast, understandable, and delightful to use.

# Installation Guide

1. Go to the releases of this repository and download the latest version for your operating system. (Currently it is released only for Linux, we can compile it for Windows if you want)
2. Extract the downloaded archive and install the .`deb`/`.rpm` (using the `dpkg` / `rpm` command) file inside.
3. Then do `chmod +x` on `run.sh`.
4. Now run `./run.sh` in the unzipped folder.
5. This should start the app, now `Ctrl + Shift + N` can be used to take screenshots on your desktop.

## If there's audio issues

Run the following

### Debian based distros

```sh
sudo apt update
sudo apt install -y libasound2-plugins pulseaudio-utils
```

### Fedora based distros

```sh
sudo dnf install -y alsa-plugins-pulseaudio pulseaudio-utils alsa-utils
```

### Common for both

```sh
cat <<'EOF' > ~/.asoundrc
pcm.!default {
  type pulse
}
ctl.!default {
  type pulse
}
EOF
```

Restart after done.

# TODOs

- [ ] Add support for linking against BLAS libraries for faster matrix operations in llama.cpp.
- [ ] Make a cleaner interface for the Rust wrapper around llama.cpp, with better error handling and more ergonomic APIs.
- [ ] Add support for custom sampling parameters and system prompts in the frontend.
