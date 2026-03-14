const { invoke } = window.__TAURI__.core;
const { register } = window.__TAURI__.globalShortcut;
const {listen} = window.__TAURI__.event;

function ensureHighlightSpan(textContainer) {
  if (!textContainer) return null;
  let highlighted = textContainer.querySelector(".highlighted_word");
  if (!highlighted) {
    textContainer.innerHTML = '<span class="highlighted_word"></span>';
    highlighted = textContainer.querySelector(".highlighted_word");
  }
  return highlighted;
}


async function handleImageUploadToRust(file)
{
  if(!file)
  {
    alert("Please select a file");
    return;
  }

  const spinner = document.getElementById("status_spinner");
  const textContainer = document.getElementById("model_output_text");


  try{
    const arrayBuffer = await file.arrayBuffer();
    const uintArray  = new Uint8Array(arrayBuffer);
    const imageBytes = Array.from(uintArray);


    if (textContainer) {
      textContainer.innerHTML = '<span class="highlighted_word"></span>';
    }
    spinner.style.display = "block";

    const result = await invoke("generate_text", {prompt : "Describe the image", imageBytes});
    console.log("Generation complete:", result);

    // Fallback: if streaming events are delayed/missed, still show final output.
    const hasRenderedText = !!(textContainer && textContainer.textContent && textContainer.textContent.trim());
    if (!hasRenderedText && typeof result === "string" && result.trim() !== "") {
      textContainer.textContent = result;
    }
  }
  catch (error)
  {
    console.error("Error durin image processing: ", error);
    alert("Check console for error");
  }
  finally{
    spinner.style.display = "none";
  }
}




window.addEventListener("DOMContentLoaded", async() => {  
  //Implementing default dark theme
  const isDark = document.body.classList.toggle("dark-theme")

  const textContainer = document.getElementById("model_output_text");
  const spinner = document.getElementById("status_spinner");

  try {
    await listen("tts_word", (event) => {
      spinner.style.display = "none";
      const word = event.payload?.word ?? "";
      const highlightedWord = ensureHighlightSpan(textContainer);
      if (highlightedWord && textContainer) {
        if (highlightedWord.textContent !== "") {
          const oldNode = document.createTextNode(highlightedWord.textContent + " ");
          textContainer.insertBefore(oldNode, highlightedWord);
        }
        highlightedWord.textContent = word;
        highlightedWord.style.backgroundColor = "#6393ce";
        highlightedWord.style.display = "inline";
      }
    });
  } catch (error) {
    console.log("Failed to attach tts_word listener:", error);
  }

  try {
    await register('CommandOrControl+Shift+N', (event) => {
      if(event.state == "Pressed"){
        console.log('Shortcut triggered');
      }
    });
  } catch (error) {
    console.log("Shortcut registration failed:", error);
  }

  const darkThemeButton = document.querySelector(".dark_mode_container");
  const darkThemeImage = document.querySelector(".dark_mode_image");

  if (darkThemeButton && darkThemeImage) {
    darkThemeButton.addEventListener("click", () => {
      const isDark = document.body.classList.toggle("dark-theme");
      
      darkThemeImage.textContent = isDark ? "Light" : "Dark";
    });
  }
  const uploadBtn = document.querySelector(".image_button");
  const fileInput = document.querySelector("#file_input");
  
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async()=>{
          const file = fileInput.files[0];
          await handleImageUploadToRust(file);
    })
  }
});
