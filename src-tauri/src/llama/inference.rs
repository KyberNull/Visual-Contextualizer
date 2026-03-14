use crate::llama::template;
use crate::llama::bindings::*;
use crate::llama::model::LlamaModel;

// Structs are made to mirror the C structs in llama.h, but with more Rusty ergonomics where possible. 
// They have default values but let us overwrite them when needed.
// They also handle all the unsafe calls to the backend and memory management, so the rest of the codebase can be safe and ergonomic.

// TODO: make mlock() true based on available ram

const SYSTEM_PROMPT: &str = ""; /*"You are a visual accessibility assistant.

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
Do not guess details that are not visible."; */

pub struct ContextConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512, // TODO: make it choose the batch size based on the cpu
            n_threads: 6, //TODO: make it choose the number of threads based on the system
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
}

impl LlamaPipeline {
    pub fn from_model(model: LlamaModel, cfg: &ContextConfig) -> Result<Self, String> {
        let mut cparams = unsafe { llama_context_default_params() };
        cparams.n_ctx = cfg.n_ctx;
        cparams.n_batch = cfg.n_batch;
        cparams.n_ubatch = cfg.n_batch/16;
        cparams.n_threads = cfg.n_threads;
        cparams.n_threads_batch = cfg.n_threads;
        cparams.kv_unified = true;
        cparams.n_seq_max = 4;

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

        // Clear old context
        unsafe { llama_memory_clear(llama_get_memory(self.ctx), true) };
        self.reset_sampler(cfg);
        
        let system_prompt = template::Message {
            role: template::Role::System,
            content: vec![template::Content::Text(SYSTEM_PROMPT.to_string())],
        };

        let user_prompt = template::Message {
            role: template::Role::User,
            content: vec![template::Content::Text(prompt.to_string())],
        };

        let prompt = [system_prompt, user_prompt];
        let formatted_prompt = template::render(&prompt);
        let prompt_tokens = self.model.tokenize(&formatted_prompt, true)?;
        if prompt_tokens.is_empty() {
            return Err("Prompt tokenization produced no tokens".to_string());
        }

        let n_prompt = prompt_tokens.len();
        let eos = self.model.eos_token();


        // ---- build batch for prompt ----

        //llama_batch_init allocates the memory based on the nos of tokens passed in it.
        let mut batch = unsafe { llama_batch_init(n_prompt as i32, 0, 1) };
        //MANUALLY setting the nos of tokens to be processed.
        batch.n_tokens = n_prompt as i32;

        unsafe {
            for i in 0..n_prompt {
                *batch.token.add(i) = prompt_tokens[i];
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;

                let seq_ptr = *batch.seq_id.add(i);
                *seq_ptr.add(0) = 0;

                *batch.logits.add(i) = if i == n_prompt - 1 { 1 } else { 0 };
            }
        }

        // ---- decode prompt ----
        let rc = unsafe { llama_decode(self.ctx, batch) };
        if rc != 0 {
            unsafe { llama_batch_free(batch) };
            return Err(format!("Prompt decode failed with code {rc}"));
        }

        unsafe { llama_batch_free(batch) };

        // ---- generation loop ----
        let mut out = String::new();
        let mut last_pos = (n_prompt - 1) as i32;
        let mut token = unsafe { llama_sampler_sample(self.sampler, self.ctx, last_pos )};
        for _ in 0..cfg.max_tokens {

            // sample next token from logits of last position

            if token == eos {
                break;
            }

            out.push_str(&self.model.token_to_piece(token)?);

            unsafe { llama_sampler_accept(self.sampler, token) };

            // ---- decode generated token ----
            let mut batch = unsafe { llama_batch_init(1, 0, 1) };
            batch.n_tokens = 1; // <-- Tell the batch it contains 1 token

            unsafe {
                *batch.token.add(0) = token;
                *batch.pos.add(0) = last_pos + 1;
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

            last_pos += 1;
            token = unsafe { llama_sampler_sample(self.sampler, self.ctx, 0)};

        }

        Ok(out)
    }

// TODO: Improve sampling strategy
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
            llama_sampler_chain_add(self.sampler, llama_sampler_init_min_p(cfg.min_p, 0));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_temp(cfg.temperature));
            llama_sampler_chain_add(self.sampler, llama_sampler_init_dist(cfg.seed));
            // TODO: PENALTIES
            // llama_sampler_chain_add(
            //     self.sampler,
            //     llama_sampler_init_presence_penalty(cfg.presence_penalty),
            // );
            // llama_sampler_chain_add(
            //     self.sampler,
            //     llama_sampler_init_repetition_penalty(cfg.repitition_penalty),
            // );
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
