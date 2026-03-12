use crate::llama::bindings::*;
use crate::llama::model::LlamaModel;

pub struct ContextConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_batch: 512,
            n_threads: 4, //TODO: make it choose the number of threads based on the system
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
            max_tokens: 4096,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            temperature: 0.7,
            seed: 0,
            presence_penalty: 1.5,
            repitition_penalty: 1.0,

        }
    }
}

pub struct LlamaRuntime;

impl LlamaRuntime {
    pub fn init() -> Self {
        unsafe { llama_backend_init() };
        Self
    }
}

impl Drop for LlamaRuntime {
    fn drop(&mut self) {
        unsafe { llama_backend_free() };
    }
}

pub struct LlamaPipeline {
    model: LlamaModel,
    ctx: *mut llama_context,
    sampler: *mut llama_sampler,
}

impl LlamaPipeline {
    pub fn from_model(model: LlamaModel, cfg: &ContextConfig) -> Result<Self, String> {
        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = cfg.n_ctx;
        cparams.n_batch = cfg.n_batch;
        cparams.n_ubatch = cfg.n_batch;
        cparams.n_threads = cfg.n_threads;
        cparams.n_threads_batch = cfg.n_threads;

        let ctx = unsafe { llama_init_from_model(model.ptr, cparams) };
        if ctx.is_null() {
            return Err("Failed to create llama context".to_string());
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
        })
    }

    pub fn generate(&mut self, prompt: &str, cfg: &GenerationConfig) -> Result<String, String> {
        unsafe { llama_memory_clear(llama_get_memory(self.ctx), true) };
        self.reset_sampler(cfg);

        let mut prompt_tokens = self.model.tokenize(prompt, true)?;
        if prompt_tokens.is_empty() {
            return Err("Prompt tokenization produced no tokens".to_string());
        }

        let batch =
            unsafe { llama_batch_get_one(prompt_tokens.as_mut_ptr(), prompt_tokens.len() as i32) };
        let code = unsafe { llama_decode(self.ctx, batch) };
        if code != 0 {
            return Err(format!("Prompt decode failed with code {code}"));
        }

        let eos = self.model.eos_token();
        let mut out = String::new();
        for _ in 0..cfg.max_tokens {
            let token = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1) };
            if token == eos {
                break;
            }

            out.push_str(&self.model.token_to_piece(token)?);
            unsafe { llama_sampler_accept(self.sampler, token) };

            let mut one = [token];
            let next = unsafe { llama_batch_get_one(one.as_mut_ptr(), 1) };
            let step = unsafe { llama_decode(self.ctx, next) };
            if step != 0 {
                return Err(format!("Decode failed while generating with code {step}"));
            }
        }

        Ok(out)
    }

    fn reset_sampler(&mut self, cfg: &GenerationConfig) {
        if !self.sampler.is_null() {
            unsafe { llama_sampler_free(self.sampler) };
            self.sampler = std::ptr::null_mut();
        }

        self.sampler = unsafe { llama_sampler_chain_init(llama_sampler_chain_default_params()) };
        if self.sampler.is_null() {
            return;
        }

        if cfg.temperature <= 0.0 {
            unsafe { llama_sampler_chain_add(self.sampler, llama_sampler_init_greedy()) };
            return;
        }

        unsafe {
            llama_sampler_chain_add(self.sampler, llama_sampler_init_top_k(cfg.top_k));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_top_p(cfg.top_p, 1));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_temp(cfg.temperature));
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
