#!/usr/bin/env bash
set -euo pipefail

APP_NAME="visual-contextualizer"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist/${APP_NAME}-portable"
BIN_SRC="$ROOT_DIR/src-tauri/target/release/$APP_NAME"
MODEL_DIR_SRC="$ROOT_DIR/src-tauri/Qwen3.5-0.8B-GGUF"
PIPER_MODEL_SRC="$ROOT_DIR/src-tauri/en_US-libritts_r-medium.onnx"
PIPER_CONFIG_SRC="$ROOT_DIR/src-tauri/en_US-libritts_r-medium.onnx.json"
ESPEAK_DATA_SRC="$ROOT_DIR/src-tauri/target/espeak-ng-data"

echo "Building release binary (no native bundle)..."
cd "$ROOT_DIR"
if [[ "${SKIP_TAURI_BUILD:-0}" != "1" ]]; then
  bun tauri build --no-bundle
else
  echo "Skipping tauri build (SKIP_TAURI_BUILD=1)"
fi

if [[ ! -f "$BIN_SRC" ]]; then
  echo "ERROR: Binary not found at $BIN_SRC"
  exit 1
fi

if [[ ! -d "$MODEL_DIR_SRC" ]]; then
  echo "ERROR: Model directory not found at $MODEL_DIR_SRC"
  exit 1
fi

if [[ ! -f "$PIPER_MODEL_SRC" || ! -f "$PIPER_CONFIG_SRC" ]]; then
  echo "ERROR: Piper model/config files not found in src-tauri/"
  exit 1
fi

if [[ ! -d "$ESPEAK_DATA_SRC" ]]; then
  echo "ERROR: espeak-ng-data directory not found at $ESPEAK_DATA_SRC, make sure to build first"
  exit 1
fi

echo "Creating portable folder at $DIST_DIR"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/bin" "$DIST_DIR/lib"

cp "$BIN_SRC" "$DIST_DIR/bin/"
cp -a "$MODEL_DIR_SRC" "$DIST_DIR/"
cp "$PIPER_MODEL_SRC" "$DIST_DIR/"
cp "$PIPER_CONFIG_SRC" "$DIST_DIR/"
cp -R "$ESPEAK_DATA_SRC" "$DIST_DIR/"

cat > "$DIST_DIR/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$APP_DIR/lib:${LD_LIBRARY_PATH:-}"
cd "$APP_DIR"
exec "$APP_DIR/bin/visual-contextualizer" "$@"
EOF

chmod +x "$DIST_DIR/run.sh" "$DIST_DIR/bin/$APP_NAME"

echo "Portable build ready: $DIST_DIR"
echo "Run with: $DIST_DIR/run.sh"