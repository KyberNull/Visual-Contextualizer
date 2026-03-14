const { invoke } = window.__TAURI__.core;
const { register } = window.__TAURI__.globalShortcut;
const { listen } = window.__TAURI__.event;

const MODEL_IMAGE_SIZE = 448; 
const SNIP_PROMPT = "Describe the image.";


async function handleImageUploadToRust(file) {
  const textContainer = document.getElementById("model_output_text");
  const spinner = document.getElementById("status_spinner");

  if (!file) return;
  try {
    const resizedImageBytes = await resizeFileToModelBytes(file);
    if (textContainer) {
      textContainer.innerHTML = ''; 
      ensureHighlightSpan(textContainer);
    }
    if (spinner) spinner.style.display = "block";
    await invoke("generate_text", {
      prompt: typeof SNIP_PROMPT !== 'undefined' ? SNIP_PROMPT : "Describe the image.",
      imageBytes: resizedImageBytes
    });
    finalizeUI(textContainer, spinner);
  } catch (err) {
    console.error("Upload process failed:", err);
    if (spinner) spinner.style.display = "none";
  }
}


function finalizeUI(textContainer, spinner) {
  if (spinner) spinner.style.display = "none";
  if (!textContainer) return;
  const currentHighlight = textContainer.querySelector(".highlighted_word");
  if (currentHighlight) {
    currentHighlight.style.backgroundColor = "transparent";
    currentHighlight.style.color = "inherit";
    currentHighlight.classList.remove("highlighted_word");
  }
  textContainer.scrollTop = textContainer.scrollHeight;
}


async function resizeFileToModelBytes(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = async () => {
      const canvas = document.createElement('canvas');
      canvas.width = MODEL_IMAGE_SIZE;
      canvas.height = MODEL_IMAGE_SIZE;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE);
      
      canvas.toBlob(async (blob) => {
        const arrayBuffer = await blob.arrayBuffer();
        resolve(Array.from(new Uint8Array(arrayBuffer)));
      }, 'image/png');
      URL.revokeObjectURL(img.src);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function ensureHighlightSpan(textContainer) {
  if (!textContainer) return null;
  let highlighted = textContainer.querySelector(".highlighted_word");
  if (!highlighted) {
    const span = document.createElement("span");
    span.className = "highlighted_word";
    span.style.padding = "0 2px";
    span.style.borderRadius = "2px";
    textContainer.appendChild(span);
    highlighted = span;
  }
  return highlighted;
}

async function startNativeSnip() {
  const overlay = document.getElementById('snip-overlay');
  const bgCanvas = document.getElementById('bg-canvas');
  const drawCanvas = document.getElementById('draw-canvas');
  const mainUi = document.querySelector(".main_ui");
  const topbar = document.querySelector(".topbar");
  const spinner = document.getElementById("status_spinner");
  const textContainer = document.getElementById("model_output_text");

  if (!overlay || !bgCanvas || !drawCanvas) return;

  const bctx = bgCanvas.getContext('2d');
  const dctx = drawCanvas.getContext('2d', { alpha: false });

  const restoreNormalUi = async () => {
    try { await invoke("reset_window_to_initial_size"); } catch (err) {}
    overlay.style.display = "none";
    if (mainUi) mainUi.style.visibility = "visible";
    if (topbar) topbar.style.visibility = "visible";
  };

  try {
    if (mainUi) mainUi.style.visibility = "hidden";
    if (topbar) topbar.style.visibility = "hidden";

    const bytes = await invoke("capture_hidden_window_screenshot");
    await new Promise((resolve) => setTimeout(resolve, 180));

    const blob = new Blob([new Uint8Array(bytes)], { type: 'image/png' });
    const img = await new Promise((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.src = URL.createObjectURL(blob);
    });

    bgCanvas.width = drawCanvas.width = window.innerWidth;
    bgCanvas.height = drawCanvas.height = window.innerHeight;
    bctx.drawImage(img, 0, 0, bgCanvas.width, bgCanvas.height);
    URL.revokeObjectURL(img.src);
    
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
      dctx.strokeStyle = "#6393ce";
      dctx.lineWidth = 2;
      dctx.strokeRect(startX, startY, e.clientX - startX, e.clientY - startY);
    };

    drawCanvas.onmouseup = async (e) => {
      if (!isDragging) return;
      isDragging = false;
      const rectX = Math.min(startX, e.clientX);
      const rectY = Math.min(startY, e.clientY);
      const rectW = Math.abs(e.clientX - startX);
      const rectH = Math.abs(e.clientY - startY);
      await restoreNormalUi();
      if (rectW < 10 || rectH < 10) return;

      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = finalCanvas.height = MODEL_IMAGE_SIZE;
      finalCanvas.getContext('2d').drawImage(bgCanvas, rectX, rectY, rectW, rectH, 0, 0, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE);
      
      const resizedBlob = await new Promise(res => finalCanvas.toBlob(res, 'image/png'));
      const arrayBuffer = await resizedBlob.arrayBuffer();
      const resizedImageBytes = Array.from(new Uint8Array(arrayBuffer));

      if (textContainer) { textContainer.innerHTML = ''; ensureHighlightSpan(textContainer); }
      if (spinner) spinner.style.display = "block";

      try {
        await invoke("generate_text", { prompt: SNIP_PROMPT, imageBytes: resizedImageBytes });
        finalizeUI(textContainer, spinner);
      } catch (err) { console.error(err); }
    };
  } catch (err) { await restoreNormalUi(); }
}

window.addEventListener("DOMContentLoaded", async() => {  
  document.body.classList.toggle("dark-theme");
  const textContainer = document.getElementById("model_output_text");
  const spinner = document.getElementById("status_spinner");
  await listen("got_a_word", (event) => {
    if (spinner) spinner.style.display = "none";
    const word = event.payload ?? "";
    const highlightedWord = ensureHighlightSpan(textContainer);
    
    if (highlightedWord && textContainer) {
      if (highlightedWord.textContent !== "") {
        textContainer.insertBefore(document.createTextNode(highlightedWord.textContent), highlightedWord);
      }      
      highlightedWord.textContent = word;
      highlightedWord.style.backgroundColor = "#6393ce";
      highlightedWord.style.color = "white";
      highlightedWord.style.display = "inline";
      textContainer.scrollTop = textContainer.scrollHeight;
    }
  });

  await register('CommandOrControl+Shift+N', (event) => {
    if(event.state == "Pressed") void startNativeSnip();
  });

  const uploadBtn = document.querySelector(".image_button");
  const fileInput = document.querySelector("#file_input");
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async() => {
      const file = fileInput.files[0];
      await handleImageUploadToRust(file);
    });
  }
});



const darkThemeButton = document.querySelector("dark_mode_image");
window.addEventListener("click", () => {
    document.body.classList.toggle("dark-theme");
})