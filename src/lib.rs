use pyo3::{prelude::*, types::IntoPyDict};

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

/// Defines the interface for a language model
pub trait LanguageModel {
    /// Creates a new language model from a pretrained model name
    /// and device name.
    fn from_pretrained(
        name: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    /// Generates text from a list of prompts; returns a list of generated text.
    /// Takes in generation options.
    fn generate(
        &self,
        prompts: Vec<String>,
        options: GenerationOptions,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>>;
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

impl LanguageModel for HuggingFaceLLM {
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

    fn generate(
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

impl LanguageModel for VLLM {
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

    fn generate(
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
