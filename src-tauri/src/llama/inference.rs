use tauri::AppHandle;
use tauri:: Emitter;

use crate::llama::template;
use crate::llama::bindings::*;
use crate::llama::model::LlamaModel;
use std::env;
use std::ffi::CString;
use num_cpus;

// Structs are made to mirror the C structs in llama.h, but with more Rusty ergonomics where possible. 
// They have default values but let us overwrite them when needed.
// They also handle all the unsafe calls to the backend and memory management, so the rest of the codebase can be safe and ergonomic.

const SYSTEM_PROMPT: &str = "You are a visual accessibility assistant.

Your job is to explain what is happening in the given image to a visually impaired user.

Focus on the visual meaning, not pixel details.

Process the image in three steps:
1. Identify the type of visual content (chart, UI panel, table, text block, diagram, etc).
2. Identify the most important visual elements.
3. Describe the key insight or pattern.

If the image contains a graph or chart:
- identify the chart type
- describe the main trend or comparison
- ignore decorative elements

Write in spoken language suitable for text-to-speech narration.
Avoid symbols, bullet points, and abbreviations.

Focus on the trend, comparison, or message of the visual.
Ignore colors, styling, or decorative UI elements unless they matter.

Do not output bullet points, headings, or numbered lists.
Write 3-4 sentences in normal conversational language.
Be descriptive but concise.
Do not guess details that are not visible.";

pub struct ContextConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            n_ctx: 1024,
            n_batch: 512, // TODO: make it choose the batch size based on the cpu
            n_threads: num_cpus::get() as i32,
        }
    }
}

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
    pub seed: u32,
    pub presence_penalty: f32,
    pub repitition_penalty: f32,
}

// For Qwen 3.5 0.8B VLM acc. to their hugging face
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            temperature: 0.6,
            seed: 6,
            presence_penalty: 0.0,
            repitition_penalty: 1.0,
        }
    }
}

// Doesn’t do anything actively after initialization, but must live as long as models/pipelines use it.
pub struct LlamaRuntime;

impl LlamaRuntime {
    pub fn init() -> Self {
        unsafe { llama_backend_init() }; // Maybe Swap with ggml_backend_load_all()
        Self
    }
}

impl Drop for LlamaRuntime {
    fn drop(&mut self) {
        unsafe { llama_backend_free() };
    }
}

// Actual generation engine
pub struct LlamaPipeline {
    model: LlamaModel,
    ctx: *mut llama_context,
    sampler: *mut llama_sampler,
    mtmd_ctx: *mut mtmd_context,
}

impl LlamaPipeline {
    pub fn from_model(model: LlamaModel, cfg: &ContextConfig) -> Result<Self, String> {
        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = cfg.n_ctx;
        cparams.n_batch = cfg.n_batch;
        cparams.n_ubatch = cfg.n_batch;
        cparams.n_threads = cfg.n_threads;
        cparams.n_threads_batch = (cfg.n_threads / 2).max(4);;
        cparams.kv_unified = true;
        cparams.n_seq_max = 4;

        let ctx = unsafe { llama_init_from_model(model.ptr, cparams) };
        if ctx.is_null() {
            return Err("Failed to create llama context".to_string());
        }

        let path = env::current_dir().unwrap().join("Qwen3.5-0.8B-GGUF").join("mmproj-BF16.gguf");
        let path = path.to_str().ok_or("Error")?;
        let path = CString::new(path).map_err(|_| "Model path contains NUL byte".to_string())?;

        let mtmd_params = unsafe { mtmd_context_params_default() };
        let mtmd_ctx = unsafe { mtmd_init_from_file(path.as_ptr(), model.ptr, mtmd_params) };
        if mtmd_ctx.is_null() {
            unsafe { llama_free(ctx) };
            return Err("Failed to create mtmd context".to_string());
        }

        let sampler = unsafe { llama_sampler_chain_init(llama_sampler_chain_default_params()) };
        if sampler.is_null() {
            unsafe { llama_free(ctx) };
            return Err("Failed to create sampler chain".to_string());
        }

        Ok(Self {
            model,
            ctx,
            sampler,
            mtmd_ctx,
        })
    }
    pub fn generate(&mut self, prompt: &str, image_data: Option<Vec<u8>>,cfg: &GenerationConfig, app: AppHandle) -> Result<String, String> {

        // Clear old context
        unsafe { llama_memory_clear(llama_get_memory(self.ctx), true) };
        self.reset_sampler(cfg);


        let bmp: *mut mtmd_bitmap = match image_data.as_ref() {
            Some(bytes) => {
                let ptr = unsafe {
                    mtmd_helper_bitmap_init_from_buf(
                        self.mtmd_ctx,
                        bytes.as_ptr(),
                        bytes.len(),
                    )
                };

                if ptr.is_null() {
                    return Err("Failed to convert image to bitmap: buffer was invalid or corrupted".into());
                }
                ptr // Return the pointer to be assigned to bmp
            }
            None => std::ptr::null_mut(), // If no image, initialize as null
        };

        let system_prompt = template::Message {
            role: template::Role::System,
            content: vec![template::Content::Text(SYSTEM_PROMPT.to_string())],
        };


        let user_content = if !bmp.is_null(){
            vec![ 
                template::Content::Image, 
                template::Content::Text(format!("\n{}", prompt)),

                ]
        }else {
            vec![template::Content::Text(prompt.to_string())]
        };


        let user_prompt = template::Message {
            role: template::Role::User,
            content: user_content,
        };

        let prompt = [system_prompt, user_prompt];
        let formatted_prompt = template::render(&prompt);

        let c_prompt = match CString::new(formatted_prompt) {
            Ok(v) => v,
            Err(_) => {
                if !bmp.is_null() {
                    unsafe { mtmd_bitmap_free(bmp) };
                }
                return Err("Nul byte found in string".to_string());
            }
        };


        let mut bitmap_ptrs: Vec<*const mtmd_bitmap> = Vec::new();
        if !bmp.is_null() {
            bitmap_ptrs.push(bmp as *const mtmd_bitmap);
}

        let chunks: *mut mtmd_input_chunks = unsafe { mtmd_input_chunks_init() };
        if chunks.is_null() {
            if !bmp.is_null() {
                unsafe { mtmd_bitmap_free(bmp) };
            }
            return Err("Failed to initialize mtmd chunks".to_string());
        }

        let input_text = mtmd_input_text {
            text: c_prompt.as_ptr(),
            add_special: false,
            parse_special: true,

        };

        let rc = unsafe {
            mtmd_tokenize(
                self.mtmd_ctx,
                chunks,
                &input_text,
                if bitmap_ptrs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    bitmap_ptrs.as_mut_ptr()
                },
                bitmap_ptrs.len(),
            )
        };

        if rc != 0 {
            unsafe { mtmd_input_chunks_free(chunks) };
            if !bmp.is_null() {
                unsafe { mtmd_bitmap_free(bmp) };
            }
            return Err(format!("mtmd_tokenization failed : {}", rc));
        }

        let mut new_n_past: llama_pos = 0;
        unsafe {
            let success = mtmd_helper_eval_chunks(self.mtmd_ctx, self.ctx, chunks, 0, 0, 512, true, &mut new_n_past);
            if success != 0 {
                mtmd_input_chunks_free(chunks);
                if !bmp.is_null() {
                    mtmd_bitmap_free(bmp);
                }
                return Err("MTMD Eval Failed: Projector or MTMD Error".into());
            }

            mtmd_input_chunks_free(chunks);
            if !bmp.is_null() {
                mtmd_bitmap_free(bmp);
            }
            

        }

        let stop_seq = "<|endoftext|><|im_start|>";
        let eos = self.model.eos_token();

        // ---- generation loop ----
        let mut out = String::new();
        let mut current_pos = new_n_past;
        // After prompt/chunk eval, logits_last=true gives one logits row to sample from.
        // Sample from logits index 0, not absolute token position.
        let mut token = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1)};
        let mut word_buffer = String::new();

        for _ in 0..cfg.max_tokens {
            // sample next token from logits of last position
            if token == eos {
                break;
            }



            
            let piece = &self.model.token_to_piece(token)?;
            out.push_str(piece);

            word_buffer.push_str(&piece);

            // Check if stop sequence appeared
            if let Some(idx) = out.find(stop_seq) {
                out.truncate(idx); // remove the stop sequence and anything after
                
                // Calculate how many characters from the end of word_buffer to remove
                if let Some(wb_idx) = word_buffer.rfind(stop_seq) {
                    word_buffer.truncate(wb_idx);
                }
                break;
            }

            if let Some((_before, _after)) = word_buffer.split_once(' ') {

                while let Some((before, after)) = word_buffer.split_once(' ') {
                    let completed_word = format!("{} ", before);
                    word_buffer = after.to_string();
                    let _ = app.emit("got_a_word", completed_word);
                }
            
            }


            unsafe { llama_sampler_accept(self.sampler, token) };

            // ---- decode generated token ----
            let mut batch = unsafe { llama_batch_init(1, 0, 1) };
            batch.n_tokens = 1; // <-- Tell the batch it contains 1 token

            unsafe {
                *batch.token.add(0) = token;
                *batch.pos.add(0) = current_pos;
                *batch.n_seq_id.add(0) = 1;

                let seq_ptr = *batch.seq_id.add(0);
                *seq_ptr.add(0) = 0;

                *batch.logits.add(0) = 1;
            }

            let rc = unsafe { llama_decode(self.ctx, batch) };
            unsafe { llama_batch_free(batch) };

            if rc != 0 {
                return Err(format!("Decode failed while generating with code {rc}"));
            }

            current_pos += 1;
            token = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1)};

        }

        if !word_buffer.is_empty()
        {
            let _ = app.emit("got_a_word", word_buffer);
        }
        Ok(out)
    }

// TODO: Improve sampling strategy
    fn reset_sampler(&mut self, cfg: &GenerationConfig) {
        if !self.sampler.is_null() {
            unsafe { llama_sampler_free(self.sampler) };
            self.sampler = std::ptr::null_mut();
        }

        let mut sampler_config = unsafe { llama_sampler_chain_default_params() };
        sampler_config.no_perf = true; // Disable perf logging for cleaner output, and because we don't need it for generation.
        
        self.sampler = unsafe { llama_sampler_chain_init(sampler_config) };
        if self.sampler.is_null() {
            return;
        }

        if cfg.temperature <= 0.0 {
            unsafe { llama_sampler_chain_add(self.sampler, llama_sampler_init_greedy()) };
            return;
        }

        unsafe {
            llama_sampler_chain_add(self.sampler, llama_sampler_init_penalties(64, cfg.repitition_penalty, 0.0, cfg.presence_penalty) );
            llama_sampler_chain_add(self.sampler, llama_sampler_init_temp(cfg.temperature));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_top_k(cfg.top_k));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_top_p(cfg.top_p, 1));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_min_p(cfg.min_p, 0));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_dist(cfg.seed));
        }
    }
}

impl Drop for LlamaPipeline {
    fn drop(&mut self) {
        if !self.sampler.is_null() {
            unsafe { llama_sampler_free(self.sampler) };
            self.sampler = std::ptr::null_mut();
        }
        if !self.mtmd_ctx.is_null() {
            unsafe { mtmd_free(self.mtmd_ctx) };
            self.mtmd_ctx = std::ptr::null_mut();
        }
        if !self.ctx.is_null() {
            unsafe { llama_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for LlamaPipeline {}
unsafe impl Sync for LlamaPipeline {}
unsafe impl Send for LlamaRuntime {}
unsafe impl Sync for LlamaRuntime {}
