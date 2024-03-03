use async_trait::async_trait;
use lazy_static::lazy_static;
use pyo3::{prelude::*, types::IntoPyDict};
use regex::Regex;
use reqwest::{self};
use serde::{Deserialize, Serialize};
use std::fmt;

/// The options for generation, including the maximum number of tokens to generate,
/// the temperature, and the top p sampling.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate
    pub max_new_tokens: Option<usize>,
    /// Temperature of the generation
    pub temperature: Option<f64>,
    /// Top p sampling
    pub top_p: Option<f64>,
    /// Trims the generated text to remove the prompt from it
    pub remove_prompt: bool,
}

fn remove_prompt(prompts: &[String], gens: &mut [String]) {
    for (prompt, gen) in prompts.iter().zip(gens.iter_mut()) {
        if gen.starts_with(prompt) {
            *gen = gen[prompt.len()..].to_string();
        }
    }
}

/// Returns true if CUDA is available according to PyTorch
pub fn is_cuda_available() -> bool {
    // torch.cuda.is_available()
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        let cuda = torch.getattr("cuda")?.getattr("is_available")?.call0()?;
        cuda.extract::<bool>()
    })
    .unwrap_or(false)
}

/// The options for building a language model
#[derive(Debug, Clone, Default)]
pub struct LLMOptions {
    /// Whether or not to use the vllm version (HuggingFace Hub models only)
    pub use_vllm: Option<bool>,
    /// The device to use for the model (HuggingFace Hub models only)
    pub device: Option<String>,
    /// The OpenAI API key (OpenAI models only)
    pub api_key: Option<String>,
}

/// A builder for LLMOptions
pub struct LLMOptionsBuilder {
    options: LLMOptions,
}

impl LLMOptionsBuilder {
    pub fn new() -> Self {
        Self {
            options: LLMOptions::default(),
        }
    }

    pub fn use_vllm(mut self, use_vllm: bool) -> Self {
        self.options.use_vllm = Some(use_vllm);
        self
    }

    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.options.device = Some(device.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.options.api_key = Some(api_key.into());
        self
    }

    pub fn build(self) -> LLMOptions {
        self.options
    }
}

/// Builds a new language model from a model name and a set of options
pub fn llm_factory(
    name: impl Into<String>,
    options: LLMOptions,
) -> Result<Box<dyn LanguageModel>, Box<dyn std::error::Error>> {
    let name = name.into();
    match get_model_source_type(name.clone()) {
        LanguageModelSourceType::OpenAI => {
            if let Some(api_key) = options.api_key {
                Ok(Box::new(OpenAILLM::new(name, api_key)))
            } else {
                Err(Box::new(ChatError {
                    message: "OpenAI API key is required".to_string(),
                }))
            }
        }
        LanguageModelSourceType::HuggingFace => {
            let device = options.device.unwrap_or_else(|| {
                if is_cuda_available() {
                    "cuda:0".to_string()
                } else {
                    "cpu".to_string()
                }
            });
            if options.use_vllm.unwrap_or(false) {
                if let Ok(vllm) = VLLM::from_pretrained(name.clone(), device) {
                    Ok(Box::new(vllm))
                } else {
                    Err(Box::new(ChatError {
                        message: format!("Failed to load vLLM model: {}", name),
                    }))
                }
            } else {
                if let Ok(hf_llm) = HuggingFaceLLM::from_pretrained(name.clone(), device) {
                    Ok(Box::new(hf_llm))
                } else {
                    Err(Box::new(ChatError {
                        message: format!("Failed to load HuggingFace model: {}", name),
                    }))
                }
            }
        }
    }
}

enum LanguageModelSourceType {
    HuggingFace,
    OpenAI,
}

fn get_model_source_type(name: impl Into<String>) -> LanguageModelSourceType {
    if is_openai_model(&name.into()) {
        LanguageModelSourceType::OpenAI
    } else {
        LanguageModelSourceType::HuggingFace
    }
}

lazy_static! {
    static ref REGISTERED_OPENAI_MODEL_PATTERNS: [Regex; 2] = [
        Regex::new(r"^gpt-4-.*$").unwrap(),
        Regex::new(r"^gpt-3\.5-turbo-.*$").unwrap(),
    ];
}

fn is_openai_model(name: &str) -> bool {
    REGISTERED_OPENAI_MODEL_PATTERNS
        .iter()
        .any(|pattern| pattern.is_match(name))
}

/// Defines the interface for a language model
#[async_trait]
pub trait LanguageModel {
    /// Generates text from a list of prompts; returns a list of generated text.
    /// Takes in generation options.
    async fn generate(
        &self,
        prompts: Vec<String>,
        options: GenerationOptions,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>>;
}

/// Defines the interface for a language model loader that is modeled after HuggingFace's ModelHubMixin
pub trait FromPretrained {
    /// Creates a new language model from a pretrained model name
    /// and device name.
    fn from_pretrained(
        name: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
}

/// Simple HuggingFace LLM wrapper
#[derive(Debug, Clone)]
pub struct HuggingFaceLLM {
    /// Path to the LLM file or HuggingFace repo
    pub name: String,
    /// Model weights (transformers AutoModel)
    pub model: pyo3::PyObject,
    /// Tokenizer (transformers AutoTokenizer)
    pub tokenizer: pyo3::PyObject,
    /// device being used. "cuda", "cpu", "cuda:0", etc.
    pub device: String,
}

impl FromPretrained for HuggingFaceLLM {
    fn from_pretrained(
        name: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let name = name.into();
            let device = device.into();
            let transformers = PyModule::import(py, "transformers")?;
            // AutoModelForCausalLM.from_pretrained("Writer/palmyra-small")
            let model: pyo3::PyObject = transformers
                .getattr("AutoModelForCausalLM")?
                .getattr("from_pretrained")?
                .call1((name.clone(),))?
                .extract()?;
            let model = model.call_method1(py, "to", (device.clone(),))?;
            // AutoTokenizer.from_pretrained("Writer/palmyra-small")
            let tokenizer: pyo3::PyObject = transformers
                .getattr("AutoTokenizer")?
                .getattr("from_pretrained")?
                .call1((name.clone(),))?
                .extract()?;
            // tokenizer.padding_side = "left"
            // tokenizer.pad_token = tokenizer.eos_token
            // tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.setattr(py, "padding_side", "left")?;
            tokenizer.setattr(py, "pad_token", tokenizer.getattr(py, "eos_token")?)?;
            tokenizer.setattr(py, "pad_token_id", tokenizer.getattr(py, "eos_token_id")?)?;
            Ok(Self {
                name,
                model,
                tokenizer,
                device,
            })
        })
    }
}

#[async_trait]
impl LanguageModel for HuggingFaceLLM {
    async fn generate(
        &self,
        prompts: Vec<String>,
        options: GenerationOptions,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let max_new_tokens = options.max_new_tokens.unwrap_or(50);
            let temperature = options.temperature.unwrap_or(0.0);
            let top_p = options.top_p.unwrap_or(1.0);
            // tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt")
            let kwargs = [("return_tensors", "pt")].into_py_dict(py);
            kwargs.getattr("__setitem__")?.call1(("padding", true))?;
            let tokens: pyo3::PyObject = self.tokenizer.getattr(py, "batch_encode_plus")?.call(
                py,
                (prompts.clone(),),
                Some(kwargs),
            )?;
            let encoding: pyo3::PyObject = tokens
                .getattr(py, "to")?
                .call1(py, (self.device.clone(),))?;
            // model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p, temperature=temperature)
            let kwargs = vec![("max_new_tokens", max_new_tokens)].into_py_dict(py);
            if temperature > 0.0 {
                // not greedy!
                kwargs.getattr("__setitem__")?.call1(("do_sample", true))?;
                kwargs
                    .getattr("__setitem__")?
                    .call1(("temperature", temperature))?;
                kwargs.getattr("__setitem__")?.call1(("top_p", top_p))?;
            }
            kwargs
                .getattr("__setitem__")?
                .call1(("input_ids", encoding.getattr(py, "input_ids")?))?;
            kwargs
                .getattr("__setitem__")?
                .call1(("attention_mask", encoding.getattr(py, "attention_mask")?))?;
            let result: pyo3::PyObject =
                self.model
                    .getattr(py, "generate")?
                    .call(py, (), Some(kwargs))?;
            // tokenizer.batch_decode(result, skip_special_tokens=True)
            let kwargs = [("skip_special_tokens", true)].into_py_dict(py);
            let mut gens: Vec<String> = self
                .tokenizer
                .getattr(py, "batch_decode")?
                .call(py, (result,), Some(kwargs))?
                .extract(py)?;
            if options.remove_prompt {
                remove_prompt(&prompts, &mut gens);
            }
            Ok(gens)
        })
    }
}

/// Simple vllm LLM wrapper, if HuggingFace is too slow for you.
/// vLLM does not care about the given device, it will always need a GPU.
#[derive(Debug, Clone)]
pub struct VLLM {
    /// Path to the LLM file or HuggingFace repo
    pub name: String,
    /// Model weights (vllm LLM)
    pub model: pyo3::PyObject,
}

/// def autodetect_dtype() -> str:
///     if torch.cuda.is_bf16_supported():
///         return "bfloat16"
///     else:
///         return "auto"
pub fn autodetect_dtype() -> String {
    Python::with_gil(|py| {
        let torch = PyModule::import(py, "torch")?;
        if torch
            .getattr("cuda")?
            .getattr("is_bf16_supported")?
            .call0()?
            .extract()?
        {
            Ok::<_, pyo3::PyErr>("bfloat16".to_string())
        } else {
            Ok::<_, pyo3::PyErr>("auto".to_string())
        }
    })
    .unwrap_or("auto".to_string())
}

impl FromPretrained for VLLM {
    fn from_pretrained(
        name: impl Into<String>,
        _device: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // self.model = LLM(
        // model_name,
        // dtype=autodetect_dtype(),
        // )
        Python::with_gil(|py| {
            let name = name.into();
            let vllm = PyModule::import(py, "vllm")?;
            let kwargs = vec![("dtype", autodetect_dtype())].into_py_dict(py);
            let model: pyo3::PyObject = vllm
                .getattr("LLM")?
                .call((name.clone(),), Some(kwargs))?
                .extract()?;
            Ok(Self { name, model })
        })
    }
}

#[async_trait]
impl LanguageModel for VLLM {
    async fn generate(
        &self,
        prompts: Vec<String>,
        options: GenerationOptions,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let max_new_tokens = options.max_new_tokens.unwrap_or(50);
            // model.generate(prompts=prompts, sampling_params=vllm.SamplingParams(max_tokens=max_new_tokens))
            let kwargs = vec![("max_tokens", max_new_tokens)].into_py_dict(py);
            let sampling_params: pyo3::PyObject = PyModule::import(py, "vllm")?
                .getattr("SamplingParams")?
                .call((), Some(kwargs))?
                .extract()?;
            let kwargs = vec![("prompts", prompts.clone())].into_py_dict(py);
            kwargs
                .getattr("__setitem__")?
                .call1(("sampling_params", sampling_params))?;
            let result: pyo3::PyObject =
                self.model
                    .getattr(py, "generate")?
                    .call(py, (), Some(kwargs))?;
            let mut gens = vec![];
            for (i, p) in prompts.iter().enumerate() {
                let req: pyo3::PyObject = result.getattr(py, "__getitem__")?.call1(py, (i,))?;
                // req.outputs[0].text
                let text: String = req
                    .getattr(py, "outputs")?
                    .getattr(py, "__getitem__")?
                    .call1(py, (0,))?
                    .getattr(py, "text")?
                    .extract(py)?;
                gens.push(format!("{}{}", p, text));
            }
            if options.remove_prompt {
                remove_prompt(&prompts, &mut gens);
            }
            Ok(gens)
        })
    }
}

/// An OpenAI language model
#[derive(Debug, Clone)]
pub struct OpenAILLM {
    pub name: String,
    pub api_key: String,
    pub client: reqwest::Client,
}

impl OpenAILLM {
    pub fn new(name: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }
}

const OPENAI_API_HOST: &str = "api.openai.com";
const OPENAI_API_CHAT_ENDPOINT: &str = "/v1/chat/completions";

#[async_trait]
impl LanguageModel for OpenAILLM {
    async fn generate(
        &self,
        prompts: Vec<String>,
        options: GenerationOptions,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let url = format!("https://{}{}", OPENAI_API_HOST, OPENAI_API_CHAT_ENDPOINT);
        let body = ChatRequestBody {
            model: self.name.clone(),
            messages: prompts
                .iter()
                .map(|p| Message {
                    role: Role::User,
                    content: p.clone(),
                })
                .collect(),
            temperature: options.temperature.unwrap_or(0.0),
            max_tokens: options.max_new_tokens,
            top_p: options.top_p,
        };
        let response = self
            .client
            .post(&url)
            .bearer_auth(self.api_key.clone())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await;
        if let Ok(response) = response {
            let response: ChatResponse = response.json().await?;
            let mut completions = Vec::with_capacity(response.choices.len());
            for (i, c) in response.choices.iter().enumerate() {
                let content = c.message.content.clone().unwrap_or_default();
                if options.remove_prompt {
                    completions.push(content);
                } else {
                    completions.push(format!("{}{}", prompts[i], content));
                }
            }
            Ok(completions)
        } else {
            Err(Box::new(ChatError {
                message: format!("Failed to generate: {:?}", response),
            }))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequestBody {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: Role,
    pub content: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

impl Serialize for Role {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        match self {
            Role::System => serializer.serialize_str("system"),
            Role::User => serializer.serialize_str("user"),
            Role::Assistant => serializer.serialize_str("assistant"),
        }
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D>(deserializer: D) -> Result<Role, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            _ => Err(serde::de::Error::custom(format!("unknown role: {}", s))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug)]
pub struct ChatError {
    pub message: String,
}

impl std::error::Error for ChatError {
    fn description(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChatError: {}", self.message)
    }
}

impl From<serde_json::Error> for ChatError {
    fn from(error: serde_json::Error) -> Self {
        ChatError {
            message: format!("{}", error),
        }
    }
}

/// A wrapper around a dataset from the HuggingFace Datasets library.
#[derive(Debug, Clone)]
pub struct HuggingFaceDataset {
    /// Path to the dataset or HuggingFace repo
    pub name: String,
    /// Split of the dataset to use
    pub split: String,
    /// The dataset (datasets.Dataset)
    pub dataset: pyo3::PyObject,
}

/// A single example from a dataset.
#[derive(Debug, Clone)]
pub struct DatasetExample {
    /// The underlying Python object
    pub example: pyo3::PyObject,
}

impl HuggingFaceDataset {
    /// Loads a new dataset from HuggingFace datasets.
    pub fn load_dataset(
        name: impl Into<String>,
        split: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let name = name.into();
            let split = split.into();
            let datasets = PyModule::import(py, "datasets")?;
            let kwargs = vec![("split", split.clone())].into_py_dict(py);
            let dataset: pyo3::PyObject = datasets
                .getattr("load_dataset")?
                .call((name.clone(),), Some(kwargs))?
                .extract()?;
            Ok(Self {
                name,
                split,
                dataset,
            })
        })
    }
}

/// Iterator over a HF dataset.
pub struct HFDatasetIterator<'a> {
    /// The dataset
    pub dataset: &'a HuggingFaceDataset,
    /// The current index
    pub index: usize,
}

impl<'a> Iterator for HFDatasetIterator<'a> {
    type Item = DatasetExample;

    fn next(&mut self) -> Option<Self::Item> {
        Python::with_gil(|py| {
            let example: pyo3::PyObject = self
                .dataset
                .dataset
                .getattr(py, "__getitem__")?
                .call1(py, (self.index,))?
                .extract(py)?;
            self.index += 1;
            Ok::<_, pyo3::PyErr>(DatasetExample { example })
        })
        .ok()
    }
}

impl<'a> IntoIterator for &'a HuggingFaceDataset {
    type Item = DatasetExample;
    type IntoIter = HFDatasetIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        HFDatasetIterator {
            dataset: self,
            index: 0,
        }
    }
}

impl DatasetExample {
    /// Gets an element from the example. An example is basically a dictionary.
    pub fn get<D>(&self, key: impl Into<String>) -> Result<D, Box<dyn std::error::Error>>
    where
        D: for<'p> FromPyObject<'p>,
    {
        Python::with_gil(|py| {
            let key = key.into();
            let pyval = self.example.getattr(py, "__getitem__")?.call1(py, (key,))?;
            let value = pyval.extract(py)?;
            Ok(value)
        })
    }
}
