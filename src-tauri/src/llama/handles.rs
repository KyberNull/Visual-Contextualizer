use crate::llama::bindings::*;

pub(super) struct MtmdBitmap {
    ptr: *mut mtmd_bitmap,
}

impl MtmdBitmap {
    pub(super) fn from_bytes(
        mtmd_ctx: *mut mtmd_context,
        image_data: Option<&[u8]>,
    ) -> Result<Self, String> {
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

    pub(super) fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    pub(super) fn as_const_ptr(&self) -> *const mtmd_bitmap {
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

pub(super) struct MtmdInputChunks {
    ptr: *mut mtmd_input_chunks,
}

impl MtmdInputChunks {
    pub(super) fn new() -> Result<Self, String> {
        let ptr = unsafe { mtmd_input_chunks_init() };
        if ptr.is_null() {
            return Err("Failed to initialize mtmd chunks".to_string());
        }
        Ok(Self { ptr })
    }

    pub(super) fn as_mut_ptr(&self) -> *mut mtmd_input_chunks {
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

pub(super) struct LlamaBatch {
    raw: llama_batch,
}

impl LlamaBatch {
    pub(super) fn single_token(token: llama_token, pos: llama_pos) -> Self {
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

    pub(super) fn as_raw(&self) -> llama_batch {
        self.raw
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.raw) };
    }
}

pub(super) struct LlamaContextHandle {
    ptr: *mut llama_context,
}

impl LlamaContextHandle {
    pub(super) fn from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> Result<Self, String> {
        let ptr = unsafe { llama_init_from_model(model, params) };
        if ptr.is_null() {
            return Err("Failed to create llama context".to_string());
        }
        Ok(Self { ptr })
    }

    pub(super) fn as_ptr(&self) -> *mut llama_context {
        self.ptr
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

pub(super) struct MtmdContextHandle {
    ptr: *mut mtmd_context,
}

impl MtmdContextHandle {
    pub(super) fn from_file(path: *const i8, model: *mut llama_model) -> Result<Self, String> {
        let mtmd_params = unsafe { mtmd_context_params_default() };
        let ptr = unsafe { mtmd_init_from_file(path, model, mtmd_params) };
        if ptr.is_null() {
            return Err("Failed to create mtmd context".to_string());
        }
        Ok(Self { ptr })
    }

    pub(super) fn as_ptr(&self) -> *mut mtmd_context {
        self.ptr
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

pub(super) struct LlamaSamplerHandle {
    ptr: *mut llama_sampler,
}

impl LlamaSamplerHandle {
    pub(super) fn new(no_perf: bool) -> Result<Self, String> {
        let mut params = unsafe { llama_sampler_chain_default_params() };
        params.no_perf = no_perf;
        let ptr = unsafe { llama_sampler_chain_init(params) };
        if ptr.is_null() {
            return Err("Failed to create sampler chain".to_string());
        }
        Ok(Self { ptr })
    }

    pub(super) fn as_ptr(&self) -> *mut llama_sampler {
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
