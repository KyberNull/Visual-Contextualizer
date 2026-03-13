pub enum Role {
    System,
    User,
}

pub enum Content {
    Text(String),
    Image,
}

pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

// Renderer for single-turn Qwen-style ChatML prompts with thinking disabled
pub fn render(messages: &[Message]) -> String {
    let mut out = String::with_capacity(512);

    for msg in messages {
        match msg.role {
            Role::System => out.push_str("<|im_start|>system\n"),
            Role::User => out.push_str("<|im_start|>user\n"),
        }

        for item in &msg.content {
            match item {
                Content::Text(text) => out.push_str(text),
                Content::Image => {
                    out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                }
            }
        }

        out.push_str("\n<|im_end|>\n");
    }

    // Assistant generation prompt with thinking forcibly closed
    out.push_str("<|im_start|>assistant\n");
    out.push_str("<think>\n\n</think>\n\n");

    out
}