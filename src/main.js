const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { register } = window.__TAURI__.globalShortcut;

// Configuration Constants
const CONFIG = {
  MODEL_SIZE: 448,
  PROMPT: "Describe the image.",
  HIGHLIGHT_COLOR: "#6393ce"
};

//UI Controller to cache DOM elements and manage simple state
const UI = {
  get textContainer() { return document.getElementById("model_output_text"); },
  get spinner() { return document.getElementById("status_spinner"); },
  get themeBtn() { return document.querySelector(".dark_mode_image"); },
  
  setLoading(isLoading) {
    if (this.spinner) this.spinner.style.display = isLoading ? "block" : "none";
  },
  
  clearText() {
    if (this.textContainer) this.textContainer.innerHTML = '';
  }
};


//Image Processing Utilities
const ImageProcessor = {
  async toModelBytes(source) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = CONFIG.MODEL_SIZE;
    const ctx = canvas.getContext('2d');

    // Handle both Image objects and File/Blobs
    if (source instanceof HTMLImageElement || source instanceof HTMLCanvasElement) {
      ctx.drawImage(source, 0, 0, CONFIG.MODEL_SIZE, CONFIG.MODEL_SIZE);
    } else {
      const img = await this.loadImage(source);
      ctx.drawImage(img, 0, 0, CONFIG.MODEL_SIZE, CONFIG.MODEL_SIZE);
      URL.revokeObjectURL(img.src);
    }

    const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
    return Array.from(new Uint8Array(await blob.arrayBuffer()));
  },

  loadImage(file) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = URL.createObjectURL(file);
    });
  }
};


//Inference & UI Feedback Logic
async function processInference(imageSource) {
  UI.clearText();
  ensureHighlightSpan(UI.textContainer);
  UI.setLoading(true);

  try {
    const bytes = await ImageProcessor.toModelBytes(imageSource);
    await invoke("generate_text", { prompt: CONFIG.PROMPT, imageBytes: bytes });
    finalizeUI();
  } catch (err) {
    console.error("Inference failed:", err);
  } finally {
    UI.setLoading(false);
  }
}

//FInal UI displayed when a word arrives from event from inference.rs
function finalizeUI() {
  UI.setLoading(false);
  const highlighted = UI.textContainer?.querySelector(".highlighted_word");
  if (highlighted) {
    highlighted.style.backgroundColor = "transparent";
    highlighted.classList.remove("highlighted_word");
  }
  UI.textContainer.scrollTop = UI.textContainer.scrollHeight;
}

function ensureHighlightSpan(container) {
  if (!container) return null;
  let span = container.querySelector(".highlighted_word");
  if (!span) {
    span = document.createElement("span");
    span.className = "highlighted_word";
    span.style.cssText = "padding: 0 2px; border-radius: 2px; display: inline;";
    container.appendChild(span);
  }
  return span;
}

//Snipping tool
async function startNativeSnip() {
  const overlay = document.getElementById('snip-overlay');
  const bgCanvas = document.getElementById('bg-canvas');
  const drawCanvas = document.getElementById('draw-canvas');
  
  if (!overlay || !bgCanvas || !drawCanvas) return;

  const bctx = bgCanvas.getContext('2d');
  const dctx = drawCanvas.getContext('2d', { alpha: false });

  // UI Reset
  const restoreUI = async () => {
    try { await invoke("reset_window_to_initial_size"); } catch {}
    overlay.style.display = "none";
    document.querySelector(".main_ui").style.visibility = "visible";
    document.querySelector(".topbar").style.visibility = "visible";
  };

  try {
    document.querySelector(".main_ui").style.visibility = "hidden";
    document.querySelector(".topbar").style.visibility = "hidden";

    const bytes = await invoke("capture_hidden_window_screenshot");
    const img = await ImageProcessor.loadImage(new Blob([new Uint8Array(bytes)], { type: 'image/png' }));

    bgCanvas.width = drawCanvas.width = window.innerWidth;
    bgCanvas.height = drawCanvas.height = window.innerHeight;
    bctx.drawImage(img, 0, 0, bgCanvas.width, bgCanvas.height);
    
    overlay.style.display = 'block';
    dctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    dctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

    let startX, startY, isDragging = false;

    drawCanvas.onmousedown = (e) => { isDragging = true; startX = e.clientX; startY = e.clientY; };
    
    drawCanvas.onmousemove = (e) => {
      if (!isDragging) return;
      dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      dctx.fillStyle = "rgba(0, 0, 0, 0.5)";
      dctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
      dctx.clearRect(startX, startY, e.clientX - startX, e.clientY - startY);
      dctx.strokeStyle = CONFIG.HIGHLIGHT_COLOR;
      dctx.strokeRect(startX, startY, e.clientX - startX, e.clientY - startY);
    };

    drawCanvas.onmouseup = async (e) => {
      if (!isDragging) return;
      isDragging = false;
      
      const rect = {
        x: Math.min(startX, e.clientX),
        y: Math.min(startY, e.clientY),
        w: Math.abs(e.clientX - startX),
        h: Math.abs(e.clientY - startY)
      };

      await restoreUI();
      if (rect.w < 10 || rect.h < 10) return;

      const snipCanvas = document.createElement('canvas');
      snipCanvas.width = snipCanvas.height = CONFIG.MODEL_SIZE;
      snipCanvas.getContext('2d').drawImage(bgCanvas, rect.x, rect.y, rect.w, rect.h, 0, 0, CONFIG.MODEL_SIZE, CONFIG.MODEL_SIZE);
      
      processInference(snipCanvas);
    };
  } catch (err) { 
    await restoreUI(); 
  }
}

//Event Initializers
window.addEventListener("DOMContentLoaded", async () => {
  // Theme Setup
  document.body.classList.add("dark-theme");
  
  UI.themeBtn?.addEventListener("click", () => {
    const isDark = document.body.classList.toggle("dark-theme");
    UI.themeBtn.textContent = isDark ? "Light" : "Dark";
  });

  // Word Stream Listener
  await listen("got_a_word", (event) => {
    UI.setLoading(false);
    const word = event.payload ?? "";
    const span = ensureHighlightSpan(UI.textContainer);
    
    if (span && UI.textContainer) {
      if (span.textContent !== "") {
        UI.textContainer.insertBefore(document.createTextNode(span.textContent), span);
      }      
      span.textContent = word;
      span.style.backgroundColor = CONFIG.HIGHLIGHT_COLOR;
      span.style.color = "white";
      UI.textContainer.scrollTop = UI.textContainer.scrollHeight;
    }
  });

  // Global Shortcut
  await register('CommandOrControl+Shift+N', (e) => {
    if (e.state === "Pressed") void startNativeSnip();
  });

  // Upload Logic
  const uploadBtn = document.querySelector(".image_button");
  const fileInput = document.getElementById("file_input");
  
  uploadBtn?.addEventListener("click", () => fileInput?.click());
  fileInput?.addEventListener("change", (e) => processInference(e.target.files[0]));
});