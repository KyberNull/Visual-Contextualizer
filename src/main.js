const { register } = require("@tauri-apps/plugin-global-shortcut");
const { invoke } = window.__TAURI__.core;

await register('CommandOrControl+Shift+C', ()=> {
  alert("Shortcut detected !!");
});


async function handleImageUploadToRust(file)
{
  if(!file)
  {
    alert("Please select a file");
    return;
  }

  try{
    const arrayBuffer = await file.arrayBuffer();
    const uintArray  = new Uint8Array(arrayBuffer);
    const json_data = Array.from(uintArray);

    const response = await invoke("get_img", {data : json_data});
    console.log("Rust response: ", response);
    alert("Image send successfully !!");
  }

  catch (error)
  {
    console.error(error);
    alert("Check console for error");
  }
}



window.addEventListener("DOMContentLoaded", async() => {
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
