use crate::llama::bindings::*;
use std::ffi::CString;

pub struct LlamaModel {
    pub ptr: *mut llama_model,
    pub vocab: *const llama_vocab,
}

impl LlamaModel {
    pub fn load(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|_| "Model path contains NUL byte".to_string())?;
        let params = unsafe { llama_model_default_params() };
        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), params) };
        if model.is_null() {
            return Err(format!("Failed to load model from path: {path}"));
        }
        let vocab = unsafe { llama_model_get_vocab(model) };
        if vocab.is_null() {
            unsafe { llama_model_free(model) };
            return Err("Model loaded but vocab pointer is null".to_string());
        }
        Ok(Self { ptr: model, vocab })
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<llama_token>, String> {
        // Create a vector of tokens with extra capacity to handle tokenization overflow
        let mut tokens = vec![0 as llama_token; text.len() + 8];

        // n is the number of tokens written, or a negative value indicating overflow or error
        let mut n = unsafe {
            llama_tokenize(
                self.vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                false,
            )
        };

        // Handle tokenization overflow and errors based on llama.cpp specification
        if n == i32::MIN {
            return Err("Tokenization overflow".to_string());
        }
        if n < 0 {
            let needed = (-n) as usize;
            tokens.resize(needed, 0);
            n = unsafe {
                llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const i8,
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    false,
                )
            };
        }

        // If failed after resizing, return an error
        if n < 0 {
            return Err("Tokenization failed".to_string());
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    pub fn eos_token(&self) -> llama_token {
        unsafe { llama_vocab_eos(self.vocab) }
    }

    // Detokenise a token to its string representation, handling buffer resizing as needed
    pub fn token_to_piece(&self, token: llama_token) -> Result<String, String> {
        let mut buf = vec![0_i8; 32];
        loop {
            let n = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n >= 0 {
                let bytes: Vec<u8> = buf[..n as usize].iter().map(|b| *b as u8).collect();
                return Ok(String::from_utf8_lossy(&bytes).into_owned());
            }
            let needed = (-n) as usize;
            if needed == 0 {
                return Ok(String::new());
            }
            buf.resize(needed, 0);
        }
    }


    pub fn apply_chat_template(&self, prompt: &str) -> Result<String, String>{
        let role_user = CString::new("user").unwrap();
        let cstr_prompt = CString::new(prompt).unwrap();

        let message = [
            llama_chat_message{
                role: role_user.as_ptr(),
                content: cstr_prompt.as_ptr(),
            }
        ];


        let mut buf = vec![0u8; 1024];
        let n = unsafe {
            llama_chat_apply_template(
                std::ptr::null(),
                message.as_ptr(),
                message.len(),
                true,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32
            )

        };

        if n < 0 {
        return Err("Failed to apply chat template".to_string());
        }
        
        let mut n = n as usize;
        if n > buf.len() {
            buf.resize(n, 0);
            let second_attempt = unsafe {
            llama_chat_apply_template(
                std::ptr::null(),
                message.as_ptr(),
                message.len(),
                true,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32
                )
            };

            n = second_attempt as usize;

        }

        let result = String::from_utf8_lossy(&buf[..n]).into_owned();
        Ok(result)
        }

    }





// Implement Drop trait for memory management
impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_model_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
