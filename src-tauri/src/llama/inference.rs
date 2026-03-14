use crate::llama::bindings::*;
use crate::llama::model::LlamaModel;
use crate::llama::template;
use std::env;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};

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
            n_ctx: 1024,  // Max context length
            n_batch: 512, // TODO: make it choose the batch size based on the cpu
            n_threads: 6,
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
            max_tokens: 256,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            temperature: 0.6,
            seed: 6, // TODO: make it random
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

struct MtmdBitmap {
    ptr: *mut mtmd_bitmap,
}

impl MtmdBitmap {
    fn from_bytes(mtmd_ctx: *mut mtmd_context, image_data: Option<&[u8]>) -> Result<Self, String> {
        let Some(bytes) = image_data else {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
            });
        };

        let ptr =
            unsafe { mtmd_helper_bitmap_init_from_buf(mtmd_ctx, bytes.as_ptr(), bytes.len()) };
        if ptr.is_null() {
            return Err(
                "Failed to convert image to bitmap: buffer was invalid or corrupted".into(),
            );
        }

        Ok(Self { ptr })
    }

    fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    fn as_const_ptr(&self) -> *const mtmd_bitmap {
        self.ptr as *const mtmd_bitmap
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { mtmd_bitmap_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

struct MtmdInputChunks {
    ptr: *mut mtmd_input_chunks,
}

impl MtmdInputChunks {
    fn new() -> Result<Self, String> {
        let ptr = unsafe { mtmd_input_chunks_init() };
        if ptr.is_null() {
            return Err("Failed to initialize mtmd chunks".to_string());
        }
        Ok(Self { ptr })
    }

    fn as_mut_ptr(&self) -> *mut mtmd_input_chunks {
        self.ptr
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { mtmd_input_chunks_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

struct LlamaBatch {
    raw: llama_batch,
}

impl LlamaBatch {
    fn single_token(token: llama_token, pos: llama_pos) -> Self {
        let mut raw = unsafe { llama_batch_init(1, 0, 1) };
        raw.n_tokens = 1;

        unsafe {
            *raw.token.add(0) = token;
            *raw.pos.add(0) = pos;
            *raw.n_seq_id.add(0) = 1;

            let seq_ptr = *raw.seq_id.add(0);
            *seq_ptr.add(0) = 0;

            *raw.logits.add(0) = 1;
        }

        Self { raw }
    }

    fn as_raw(&self) -> llama_batch {
        self.raw
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.raw) };
    }
}

struct LlamaContextHandle {
    ptr: *mut llama_context,
}

impl LlamaContextHandle {
    fn from_model(model: *mut llama_model, params: llama_context_params) -> Result<Self, String> {
        let ptr = unsafe { llama_init_from_model(model, params) };
        if ptr.is_null() {
            return Err("Failed to create llama context".to_string());
        }
        Ok(Self { ptr })
    }

    fn into_raw(mut self) -> *mut llama_context {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        ptr
    }
}

impl Drop for LlamaContextHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

struct MtmdContextHandle {
    ptr: *mut mtmd_context,
}

impl MtmdContextHandle {
    fn from_file(path: *const i8, model: *mut llama_model) -> Result<Self, String> {
        let mtmd_params = unsafe { mtmd_context_params_default() };
        let ptr = unsafe { mtmd_init_from_file(path, model, mtmd_params) };
        if ptr.is_null() {
            return Err("Failed to create mtmd context".to_string());
        }
        Ok(Self { ptr })
    }

    fn into_raw(mut self) -> *mut mtmd_context {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        ptr
    }
}

impl Drop for MtmdContextHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { mtmd_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

struct LlamaSamplerHandle {
    ptr: *mut llama_sampler,
}

impl LlamaSamplerHandle {
    fn new(no_perf: bool) -> Result<Self, String> {
        let mut params = unsafe { llama_sampler_chain_default_params() };
        params.no_perf = no_perf;
        let ptr = unsafe { llama_sampler_chain_init(params) };
        if ptr.is_null() {
            return Err("Failed to create sampler chain".to_string());
        }
        Ok(Self { ptr })
    }

    fn as_ptr(&self) -> *mut llama_sampler {
        self.ptr
    }
}

impl Drop for LlamaSamplerHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_sampler_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

pub fn resolve_dependency_path(relative: &Path) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join(relative));
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join(relative));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join(relative));
            }
        }
    }

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(format!("Could not find dependency: {}", relative.display()))
}

// Actual generation engine
pub struct LlamaPipeline {
    model: LlamaModel,
    ctx: *mut llama_context,
    sampler: Option<LlamaSamplerHandle>,
    mtmd_ctx: *mut mtmd_context,
}

impl LlamaPipeline {
    pub fn from_model(model: LlamaModel, cfg: &ContextConfig) -> Result<Self, String> {
        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = cfg.n_ctx;
        cparams.n_batch = cfg.n_batch;
        cparams.n_ubatch = cfg.n_batch;
        cparams.n_threads = cfg.n_threads;
        cparams.n_threads_batch = (cfg.n_threads / 2).max(4);
        cparams.kv_unified = true;
        cparams.n_seq_max = 4;

        let ctx = LlamaContextHandle::from_model(model.ptr, cparams)?;

        let path = resolve_dependency_path(Path::new("Qwen3.5-0.8B-GGUF/mmproj-BF16.gguf"))?;
        let path = path.to_str().ok_or("Model path contains non-UTF-8 bytes")?;
        let path = CString::new(path).map_err(|_| "Model path contains NUL byte".to_string())?;

        let mtmd_ctx = MtmdContextHandle::from_file(path.as_ptr(), model.ptr)?;
        let sampler = LlamaSamplerHandle::new(true)?;

        Ok(Self {
            model,
            ctx: ctx.into_raw(),
            sampler: Some(sampler),
            mtmd_ctx: mtmd_ctx.into_raw(),
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        image_data: Option<Vec<u8>>,
        cfg: &GenerationConfig,
        app: AppHandle,
    ) -> Result<String, String> {
        // Clear old context
        unsafe { llama_memory_clear(llama_get_memory(self.ctx), true) };
        self.reset_sampler(cfg)?;
        let sampler_ptr = self
            .sampler
            .as_ref()
            .map(LlamaSamplerHandle::as_ptr)
            .ok_or("Sampler is not initialized")?;

        let bmp = MtmdBitmap::from_bytes(self.mtmd_ctx, image_data.as_deref())?;

        let system_prompt = template::Message {
            role: template::Role::System,
            content: vec![template::Content::Text(SYSTEM_PROMPT.to_string())],
        };

        let user_content = if !bmp.is_null() {
            vec![
                template::Content::Image,
                template::Content::Text(format!("\n{}", prompt)),
            ]
        } else {
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
            Err(_) => return Err("Nul byte found in string".to_string()),
        };

        let mut bitmap_ptrs: Vec<*const mtmd_bitmap> = Vec::new();
        if !bmp.is_null() {
            bitmap_ptrs.push(bmp.as_const_ptr());
        }

        let chunks = MtmdInputChunks::new()?;

        let input_text = mtmd_input_text {
            text: c_prompt.as_ptr(),
            add_special: false,
            parse_special: true,
        };

        let rc = unsafe {
            mtmd_tokenize(
                self.mtmd_ctx,
                chunks.as_mut_ptr(),
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
            return Err(format!("mtmd_tokenization failed : {}", rc));
        }

        let mut new_n_past: llama_pos = 0;
        unsafe {
            let success = mtmd_helper_eval_chunks(
                self.mtmd_ctx,
                self.ctx,
                chunks.as_mut_ptr(),
                0,
                0,
                512,
                true,
                &mut new_n_past,
            );
            if success != 0 {
                return Err("MTMD Eval Failed: Projector or MTMD Error".into());
            }
        }

        let stop_seq = "<|endoftext|><|im_start|>";
        let eos = self.model.eos_token();

        // ---- generation loop ----
        let mut out = String::new();
        let mut current_pos = new_n_past;
        // After prompt/chunk eval, logits_last=true gives one logits row to sample from.
        // Sample from logits index 0, not absolute token position.
        let mut token = unsafe { llama_sampler_sample(sampler_ptr, self.ctx, -1) };
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

            unsafe { llama_sampler_accept(sampler_ptr, token) };

            // ---- decode generated token ----
            let batch = LlamaBatch::single_token(token, current_pos);
            let rc = unsafe { llama_decode(self.ctx, batch.as_raw()) };

            if rc != 0 {
                return Err(format!("Decode failed while generating with code {rc}"));
            }

            current_pos += 1;
            token = unsafe { llama_sampler_sample(sampler_ptr, self.ctx, -1) };
        }

        if !word_buffer.is_empty() {
            let _ = app.emit("got_a_word", word_buffer);
        }
        Ok(out)
    }

    // TODO: Improve sampling strategy
    fn reset_sampler(&mut self, cfg: &GenerationConfig) -> Result<(), String> {
        let sampler = LlamaSamplerHandle::new(true)?;
        let sampler_ptr = sampler.as_ptr();

        if cfg.temperature <= 0.0 {
            unsafe { llama_sampler_chain_add(sampler_ptr, llama_sampler_init_greedy()) };
            self.sampler = Some(sampler);
            return Ok(());
        }

        unsafe {
            llama_sampler_chain_add(
                sampler_ptr,
                llama_sampler_init_penalties(64, cfg.repitition_penalty, 0.0, cfg.presence_penalty),
            );
            llama_sampler_chain_add(sampler_ptr, llama_sampler_init_temp(cfg.temperature));
            llama_sampler_chain_add(sampler_ptr, llama_sampler_init_top_k(cfg.top_k));
            llama_sampler_chain_add(sampler_ptr, llama_sampler_init_top_p(cfg.top_p, 1));
            llama_sampler_chain_add(sampler_ptr, llama_sampler_init_min_p(cfg.min_p, 0));
            llama_sampler_chain_add(sampler_ptr, llama_sampler_init_dist(cfg.seed));
        }

        self.sampler = Some(sampler);
        Ok(())
    }
}

impl Drop for LlamaPipeline {
    fn drop(&mut self) {
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
