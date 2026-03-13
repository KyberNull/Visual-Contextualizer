const { invoke } = window.__TAURI__.core;
const { register } = window.__TAURI__.globalShortcut;
const {listen} = window.__TAURI__.event;

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

  await register('CommandOrControl+Shift+N', (event) => {
    if(event.state == "Pressed"){
      console.log('Shortcut triggered');
    }
  });

  try{
    const textContainer = document.getElementById("model_output_text");
    const spinner = document.getElementById("status_spinner");

    spinner.style.display = "block";
    const unlisten = await listen("got_a_word", (event) => {
      spinner.style.display = "none";
      const word = event.payload;
      textContainer.textContent += word;
    });



    const result = await invoke("generate_text", {prompt : "I am testing u, in my app. speak for a lot of time approx 300 words"});
    unlisten();
    spinner.style.display = "none";
  }
  catch(error)
  {
    console.log(error);
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
