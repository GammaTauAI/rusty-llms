# rusty-llms: A Rust library for interfacing with HuggingFace and vLLM APIs

Load your favorite language model from HuggingFace and use it to generate text with vLLM;
all within Rust!

## Installation

This library is just a pyo3 wrapper around vLLM and HuggingFace's transformers library.
To use it, you will need to install the Python dependencies for both of those libraries:

```bash
pip install -r requirements.txt
```

## Usage

Example usage is shown below:

```rust
use rusty_llms::huggingface::HuggingFace;

fn main() {
    let llm =
        HuggingFaceLLM::from_pretrained("Writer/palmyra-small", device).unwrap();
    println!("llm: {:?}", llm);
    let gen = llm
        .generate(
            vec![
                "One of the laziest".to_string(),
                "I once was".to_string(),
                "For the first time".to_string(),
            ],
            planner::llm::GenerationOptions {
                max_new_tokens: Some(10),
                temperature: Some(0.7),
                ..Default::default()
            },
        )
        .unwrap();
    for gen in gen {
        println!("gen: {}", gen);
    }
}
```

You can also use `VLLM` instead of `HuggingFaceLLM`; it will provide the same functionality
(implements same trait), but it will be faster.
