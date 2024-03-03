use std::vec;

use tokio;

// pub fn main() {
//     let ds = rusty_llms::HuggingFaceDataset::load_dataset("nuprl/CanItEdit", "test").unwrap();
//     for ex in ds.into_iter() {
//         let before: String = ex.get("before").unwrap();
//         println!("{}", before);
//     }
// }

#[tokio::main]
pub async fn main() {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    let llm_options = rusty_llms::LLMOptionsBuilder::new()
        .api_key(api_key)
        .build();
    let model = rusty_llms::llm_factory("gpt-3.5-turbo-0125", llm_options).unwrap();
    let inputs = vec!["The quick brown fox jumps over the lazy dog.".to_string()];
    let output = model
        .generate(
            inputs,
            rusty_llms::GenerationOptions {
                max_new_tokens: Some(10),
                temperature: Some(0.0),
                top_p: Some(0.95),
                remove_prompt: true,
            },
        )
        .await;
    if let Ok(output) = output {
        println!("{:?}", output);
    } else {
        println!("{:?}", output);
    }
}
