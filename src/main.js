const { invoke } = window.__TAURI__.core;

let greetInputEl;
let greetMsgEl;

window.addEventListener("DOMContentLoaded", () => {
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
  }
});