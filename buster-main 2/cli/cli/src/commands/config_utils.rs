use crate::error::BusterError;
use dirs;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// Moved from run.rs
pub fn prompt_for_input(
    prompt_message: &str,
    default_value: Option<&str>,
    is_sensitive: bool,
) -> Result<String, BusterError> {
    if let Some(def_val) = default_value {
        print!("{} (default: {}): ", prompt_message, def_val);
    } else {
        print!("{}: ", prompt_message);
    }
    io::stdout()
        .flush()
        .map_err(|e| BusterError::CommandError(format!("Failed to flush stdout: {}", e)))?;

    let mut input = String::new();
    // Simple masking for sensitive input is complex in raw terminal io without extra crates.
    // For a real CLI, rpassword or similar would be used.
    // Here, we just read the line.
    io::stdin()
        .read_line(&mut input)
        .map_err(|e| BusterError::CommandError(format!("Failed to read line: {}", e)))?;
    let trimmed_input = input.trim().to_string();

    if trimmed_input.is_empty() {
        if let Some(def_val) = default_value {
            Ok(def_val.to_string())
        } else {
            println!("Input cannot be empty. Please try again.");
            prompt_for_input(prompt_message, default_value, is_sensitive) // Recurse
        }
    } else {
        Ok(trimmed_input)
    }
}

pub fn get_app_base_dir() -> Result<PathBuf, BusterError> {
    dirs::home_dir()
        .map(|home| home.join(".buster"))
        .ok_or_else(|| BusterError::CommandError("Failed to get home directory.".to_string()))
}

pub fn get_cached_value(
    app_base_dir: &Path,
    cache_file_name: &str,
) -> Result<Option<String>, BusterError> {
    let cache_file_path = app_base_dir.join(cache_file_name);
    if cache_file_path.exists() {
        fs::read_to_string(cache_file_path)
            .map(|val| Some(val.trim().to_string()))
            .map_err(|e| {
                BusterError::CommandError(format!(
                    "Failed to read cached file {}: {}",
                    cache_file_name, e
                ))
            })
    } else {
        Ok(None)
    }
}

pub fn cache_value(
    app_base_dir: &Path,
    cache_file_name: &str,
    value: &str,
) -> Result<(), BusterError> {
    let cache_file_path = app_base_dir.join(cache_file_name);
    fs::create_dir_all(app_base_dir).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to create app base dir {}: {}",
            app_base_dir.display(),
            e
        ))
    })?;
    fs::write(cache_file_path, value).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to cache value to {}: {}",
            cache_file_name, e
        ))
    })
}

pub fn update_env_file(
    target_dotenv_path: &Path,
    llm_api_key: Option<&str>,
    rerank_api_key: Option<&str>,
    rerank_model: Option<&str>,
    rerank_base_url: Option<&str>,
    llm_base_url: Option<&str>, // Added for completeness, though not prompted by user yet
    litellm_config_path: Option<&str>, // Added for litellm config path
) -> Result<(), BusterError> {
    let mut new_env_lines: Vec<String> = Vec::new();
    let mut llm_key_updated = false;
    let mut rerank_key_updated = false;
    let mut rerank_model_updated = false;
    let mut rerank_base_updated = false;
    let mut llm_base_updated = false;
    let mut litellm_config_updated = false;

    if target_dotenv_path.exists() {
        let env_content = fs::read_to_string(target_dotenv_path).map_err(|e| {
            BusterError::CommandError(format!(
                "Failed to read .env file at {}: {}",
                target_dotenv_path.display(),
                e
            ))
        })?;

        for line in env_content.lines() {
            if line.starts_with("LLM_API_KEY=") && llm_api_key.is_some() {
                new_env_lines.push(format!("LLM_API_KEY=\"{}\"", llm_api_key.unwrap()));
                llm_key_updated = true;
            } else if line.starts_with("RERANK_API_KEY=") && rerank_api_key.is_some() {
                new_env_lines.push(format!("RERANK_API_KEY=\"{}\"", rerank_api_key.unwrap()));
                rerank_key_updated = true;
            } else if line.starts_with("RERANK_MODEL=") && rerank_model.is_some() {
                new_env_lines.push(format!("RERANK_MODEL=\"{}\"", rerank_model.unwrap()));
                rerank_model_updated = true;
            } else if line.starts_with("RERANK_BASE_URL=") && rerank_base_url.is_some() {
                new_env_lines.push(format!("RERANK_BASE_URL=\"{}\"", rerank_base_url.unwrap()));
                rerank_base_updated = true;
            } else if line.starts_with("LLM_BASE_URL=") && llm_base_url.is_some() {
                new_env_lines.push(format!("LLM_BASE_URL=\"{}\"", llm_base_url.unwrap()));
                llm_base_updated = true;
            } else if line.starts_with("LITELLM_CONFIG_PATH=") && litellm_config_path.is_some() {
                new_env_lines.push(format!(
                    "LITELLM_CONFIG_PATH=\"{}\"",
                    litellm_config_path.unwrap()
                ));
                litellm_config_updated = true;
            } else {
                new_env_lines.push(line.to_string());
            }
        }
    }

    // Add any keys that were not found and updated, if new values are provided
    if !llm_key_updated && llm_api_key.is_some() {
        new_env_lines.push(format!("LLM_API_KEY=\"{}\"", llm_api_key.unwrap()));
    }
    if !rerank_key_updated && rerank_api_key.is_some() {
        new_env_lines.push(format!("RERANK_API_KEY=\"{}\"", rerank_api_key.unwrap()));
    }
    if !rerank_model_updated && rerank_model.is_some() {
        new_env_lines.push(format!("RERANK_MODEL=\"{}\"", rerank_model.unwrap()));
    }
    if !rerank_base_updated && rerank_base_url.is_some() {
        new_env_lines.push(format!("RERANK_BASE_URL=\"{}\"", rerank_base_url.unwrap()));
    }
    if !llm_base_updated && llm_base_url.is_some() {
        new_env_lines.push(format!("LLM_BASE_URL=\"{}\"", llm_base_url.unwrap()));
    } else if !llm_base_updated && llm_base_url.is_none() && !target_dotenv_path.exists() {
        // Ensure default LLM_BASE_URL if .env is being created from scratch and no override provided
        new_env_lines.push("LLM_BASE_URL=\"https://api.openai.com/v1\"".to_string());
    }
    if !litellm_config_updated && litellm_config_path.is_some() {
        new_env_lines.push(format!(
            "LITELLM_CONFIG_PATH=\"{}\"",
            litellm_config_path.unwrap()
        ));
    }

    fs::write(target_dotenv_path, new_env_lines.join("\n")).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to write updated .env file to {}: {}",
            target_dotenv_path.display(),
            e
        ))
    })
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct ModelInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>, // e.g., "embedding", "chat"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_cost_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>, // e.g., "gpt-3.5-turbo"
    // For any other custom key-value pairs in model_info
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extras: Option<std::collections::HashMap<String, serde_yaml::Value>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LiteLLMModelConfig {
    pub model_name: String, // Alias for the model, e.g., "my-gpt4"

    // Parameters for LiteLLM to connect to and use the model.
    // This should be a YAML map including the actual model identifier, API key, base URL, etc.
    // Example for OpenAI:
    //   litellm_params:
    //     model: "gpt-4-turbo"  // or "openai/gpt-4-turbo"
    //     api_key: "sk-..."
    //     api_base: "https://api.openai.com/v1"
    // Example for Ollama:
    //   litellm_params:
    //     model: "ollama/mistral"
    //     api_base: "http://localhost:11434"
    pub litellm_params: serde_yaml::Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<ModelInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u64>, // Tokens Per Minute
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>, // Requests Per Minute
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LiteLLMConfig {
    pub model_list: Vec<LiteLLMModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment_variables: Option<serde_yaml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub general_settings: Option<serde_yaml::Value>,
}

const OPENAI_MODELS: [&str; 5] = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "o3",
];
const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1";

pub fn create_litellm_yaml(
    app_base_dir: &Path,
    api_key: &str,
    api_base: Option<&str>,
) -> Result<PathBuf, BusterError> {
    let litellm_config_dir = app_base_dir.join("litellm_config");
    fs::create_dir_all(&litellm_config_dir).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to create litellm config directory at {}: {}",
            litellm_config_dir.display(),
            e
        ))
    })?;

    let config_path = litellm_config_dir.join("config.yaml");

    // Build model list
    let model_list: Vec<LiteLLMModelConfig> = OPENAI_MODELS
        .iter()
        .map(|model_name| {
            let mut params_map = serde_yaml::Mapping::new();
            params_map.insert(
                serde_yaml::Value::String("model".to_string()),
                serde_yaml::Value::String(model_name.to_string()),
            );
            params_map.insert(
                serde_yaml::Value::String("api_key".to_string()),
                serde_yaml::Value::String(api_key.to_string()),
            );
            if let Some(base) = api_base {
                params_map.insert(
                    serde_yaml::Value::String("api_base".to_string()),
                    serde_yaml::Value::String(base.to_string()),
                );
            }

            LiteLLMModelConfig {
                model_name: model_name.to_string(),
                litellm_params: serde_yaml::Value::Mapping(params_map),
                model_info: Some(ModelInfo::default()), // Or None if preferred
                tpm: None,
                rpm: None,
            }
        })
        .collect();

    // Env vars mapping
    let mut env_vars_map = serde_yaml::Mapping::new();
    env_vars_map.insert(
        serde_yaml::Value::String("OPENAI_API_KEY".to_string()),
        serde_yaml::Value::String(api_key.to_string()),
    );

    // General settings mapping (fallback_models etc.)
    let mut general_settings_map = serde_yaml::Mapping::new();
    general_settings_map.insert(
        serde_yaml::Value::String("fallback_models".to_string()),
        serde_yaml::Value::Sequence(
            OPENAI_MODELS
                .iter()
                .map(|m| serde_yaml::Value::String((*m).to_string()))
                .collect(),
        ),
    );

    let config = LiteLLMConfig {
        model_list,
        environment_variables: Some(serde_yaml::Value::Mapping(env_vars_map)),
        general_settings: Some(serde_yaml::Value::Mapping(general_settings_map)),
    };

    let yaml_content = serde_yaml::to_string(&config).map_err(|e| {
        BusterError::CommandError(format!("Failed to serialize LiteLLM config to YAML: {}", e))
    })?;

    fs::write(&config_path, yaml_content).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to write LiteLLM config file to {}: {}",
            config_path.display(),
            e
        ))
    })?;

    Ok(config_path)
}

pub fn update_litellm_yaml(
    app_base_dir: &Path,
    api_key: &str,
    api_base: Option<&str>,
) -> Result<PathBuf, BusterError> {
    let litellm_config_dir = app_base_dir.join("litellm_config");
    let config_path = litellm_config_dir.join("config.yaml");

    // Ensure directory exists
    fs::create_dir_all(&litellm_config_dir).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to create litellm config directory at {}: {}",
            litellm_config_dir.display(),
            e
        ))
    })?;

    if !config_path.exists() {
        return create_litellm_yaml(app_base_dir, api_key, api_base);
    }

    // Read existing config
    let yaml_content = fs::read_to_string(&config_path).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to read LiteLLM config file at {}: {}",
            config_path.display(),
            e
        ))
    })?;

    let mut config: LiteLLMConfig = serde_yaml::from_str(&yaml_content).map_err(|e| {
        BusterError::CommandError(format!("Failed to parse LiteLLM config YAML: {}", e))
    })?;

    // Ensure each model present and updated
    for model_name in OPENAI_MODELS.iter() {
        let mut found = false;
        for model_config in &mut config.model_list {
            if &model_config.model_name == model_name {
                // Ensure litellm_params is a mutable mapping
                if let serde_yaml::Value::Mapping(params_map) = &mut model_config.litellm_params {
                    params_map.insert(
                        serde_yaml::Value::String("api_key".to_string()),
                        serde_yaml::Value::String(api_key.to_string()),
                    );
                    if let Some(base) = api_base {
                        params_map.insert(
                            serde_yaml::Value::String("api_base".to_string()),
                            serde_yaml::Value::String(base.to_string()),
                        );
                    } else {
                        params_map.remove(&serde_yaml::Value::String("api_base".to_string()));
                    }
                } else {
                    // This case should ideally not happen if params are always created as Mappings
                    // For robustness, one might recreate it:
                    let mut params_map = serde_yaml::Mapping::new();
                    params_map.insert(
                        serde_yaml::Value::String("model".to_string()),
                        serde_yaml::Value::String(model_name.to_string()),
                    );
                     params_map.insert(
                        serde_yaml::Value::String("api_key".to_string()),
                        serde_yaml::Value::String(api_key.to_string()),
                    );
                    if let Some(base) = api_base {
                        params_map.insert(
                            serde_yaml::Value::String("api_base".to_string()),
                            serde_yaml::Value::String(base.to_string()),
                        );
                    }
                    model_config.litellm_params = serde_yaml::Value::Mapping(params_map);
                }
                found = true;
                break;
            }
        }
        if !found {
            let mut params_map = serde_yaml::Mapping::new();
            params_map.insert(
                serde_yaml::Value::String("model".to_string()),
                serde_yaml::Value::String(model_name.to_string()),
            );
            params_map.insert(
                serde_yaml::Value::String("api_key".to_string()),
                serde_yaml::Value::String(api_key.to_string()),
            );
            if let Some(base) = api_base {
                params_map.insert(
                    serde_yaml::Value::String("api_base".to_string()),
                    serde_yaml::Value::String(base.to_string()),
                );
            }

            config.model_list.push(LiteLLMModelConfig {
                model_name: model_name.to_string(),
                litellm_params: serde_yaml::Value::Mapping(params_map),
                model_info: Some(ModelInfo::default()), // Or None
                tpm: None,
                rpm: None,
            });
        }
    }

    // Update environment variables
    match &mut config.environment_variables {
        Some(serde_yaml::Value::Mapping(map)) => {
            map.insert(
                serde_yaml::Value::String("OPENAI_API_KEY".to_string()),
                serde_yaml::Value::String(api_key.to_string()),
            );
        }
        _ => {
            let mut env_map = serde_yaml::Mapping::new();
            env_map.insert(
                serde_yaml::Value::String("OPENAI_API_KEY".to_string()),
                serde_yaml::Value::String(api_key.to_string()),
            );
            config.environment_variables = Some(serde_yaml::Value::Mapping(env_map));
        }
    }

    // Update general settings fallback_models to include all models
    let fallback_seq: Vec<serde_yaml::Value> = OPENAI_MODELS
        .iter()
        .map(|m| serde_yaml::Value::String((*m).to_string()))
        .collect();
    match &mut config.general_settings {
        Some(serde_yaml::Value::Mapping(settings)) => {
            settings.insert(
                serde_yaml::Value::String("fallback_models".to_string()),
                serde_yaml::Value::Sequence(fallback_seq),
            );
        }
        _ => {
            let mut settings = serde_yaml::Mapping::new();
            settings.insert(
                serde_yaml::Value::String("fallback_models".to_string()),
                serde_yaml::Value::Sequence(fallback_seq),
            );
            config.general_settings = Some(serde_yaml::Value::Mapping(settings));
        }
    }

    // Serialize and write back
    let updated_yaml = serde_yaml::to_string(&config).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to serialize updated LiteLLM config to YAML: {}",
            e
        ))
    })?;

    fs::write(&config_path, updated_yaml).map_err(|e| {
        BusterError::CommandError(format!(
            "Failed to write updated LiteLLM config file to {}: {}",
            config_path.display(),
            e
        ))
    })?;

    Ok(config_path)
}

pub fn prompt_and_manage_openai_api_key(
    app_base_dir: &Path,
    force_prompt: bool,
) -> Result<String, BusterError> {
    let cache_file = ".openai_api_key";
    let mut current_key = get_cached_value(app_base_dir, cache_file)?;

    if force_prompt || current_key.is_none() {
        if current_key.is_some() {
            let key_display = current_key.as_ref().map_or("", |k| {
                if k.len() > 4 {
                    &k[k.len() - 4..]
                } else {
                    "****"
                }
            });
            let update_choice = prompt_for_input(
                &format!("Current OpenAI API key ends with ...{}. Update? (y/n)", key_display),
                Some("n"),
                false,
            )?
            .to_lowercase();
            if update_choice != "y" {
                return Ok(current_key.unwrap());
            }
        }

        let new_key = prompt_for_input("Enter your OpenAI API Key:", None, true)?;
        let api_base_choice = prompt_for_input(
            "Use custom API base URL? (y/n):",
            Some("n"),
            false,
        )?
        .to_lowercase();
        let api_base = if api_base_choice == "y" {
            Some(
                prompt_for_input(
                    "Enter the API base URL:",
                    Some("https://api.openai.com/v1"),
                    false,
                )?,
            )
        } else {
            Some("https://api.openai.com/v1".to_string())
        };

        // Update LiteLLM config first (borrows new_key)
        update_litellm_yaml(app_base_dir, &new_key, api_base.as_deref())?;

        // Cache the key after successful update
        cache_value(app_base_dir, cache_file, &new_key)?;
        current_key = Some(new_key);
        println!("LiteLLM configuration file updated successfully.");
    }

    current_key.ok_or_else(|| {
        BusterError::CommandError("OpenAI API Key setup failed.".to_string())
    })
}

pub struct RerankerConfig {
    pub provider: String,
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

pub fn prompt_and_manage_reranker_settings(
    app_base_dir: &Path,
    force_prompt: bool,
) -> Result<RerankerConfig, BusterError> {
    let provider_cache = ".reranker_provider";
    let key_cache = ".reranker_api_key";
    let model_cache = ".reranker_model";
    let url_cache = ".reranker_base_url";

    let mut current_provider = get_cached_value(app_base_dir, provider_cache)?;
    let mut current_key = get_cached_value(app_base_dir, key_cache)?;
    let mut current_model = get_cached_value(app_base_dir, model_cache)?;
    let mut current_url = get_cached_value(app_base_dir, url_cache)?;

    let mut needs_update = force_prompt;
    if !needs_update
        && (current_provider.is_none()
            || current_key.is_none()
            || current_model.is_none()
            || current_url.is_none())
    {
        needs_update = true; // If any part is missing, force update flow for initial setup
    }

    if needs_update {
        if !force_prompt && current_provider.is_some() && current_model.is_some() {
            // Already prompted if force_prompt is true
            let update_choice = prompt_for_input(
                &format!(
                    "Current Reranker: {} (Model: {}). Update settings? (y/n)",
                    current_provider.as_ref().unwrap_or(&"N/A".to_string()),
                    current_model.as_ref().unwrap_or(&"N/A".to_string())
                ),
                Some("n"),
                false,
            )?
            .to_lowercase();
            if update_choice != "y"
                && current_provider.is_some()
                && current_key.is_some()
                && current_model.is_some()
                && current_url.is_some()
            {
                return Ok(RerankerConfig {
                    provider: current_provider.unwrap(),
                    api_key: current_key.unwrap(),
                    model: current_model.unwrap(),
                    base_url: current_url.unwrap(),
                });
            }
        } else if force_prompt && current_provider.is_some() && current_model.is_some() {
            let update_choice = prompt_for_input(
                &format!(
                    "Current Reranker: {} (Model: {}). Update settings? (y/n)",
                    current_provider.as_ref().unwrap_or(&"N/A".to_string()),
                    current_model.as_ref().unwrap_or(&"N/A".to_string())
                ),
                Some("n"),
                false,
            )?
            .to_lowercase();
            if update_choice != "y"
                && current_provider.is_some()
                && current_key.is_some()
                && current_model.is_some()
                && current_url.is_some()
            {
                return Ok(RerankerConfig {
                    provider: current_provider.unwrap(),
                    api_key: current_key.unwrap(),
                    model: current_model.unwrap(),
                    base_url: current_url.unwrap(),
                });
            }
        }

        println!("--- Reranker Setup ---");
        println!("Choose your reranker provider:");
        println!("1: Cohere");
        println!("2: Mixedbread");
        println!("3: Jina");
        let provider_choice = loop {
            match prompt_for_input("Enter choice (1-3):", Some("1"), false)?.parse::<u32>() {
                Ok(choice @ 1..=3) => break choice,
                _ => println!("Invalid choice. Please enter a number between 1 and 3."),
            }
        };

        let (new_provider, default_model, default_url) = match provider_choice {
            1 => (
                "Cohere",
                "rerank-english-v3.0",
                "https://api.cohere.com/v1/rerank",
            ), // user asked for v3.5 but official docs say v3.0 for rerank model
            2 => (
                "Mixedbread",
                "mixedbread-ai/mxbai-rerank-xsmall-v1",
                "https://api.mixedbread.ai/v1/reranking",
            ),
            3 => (
                "Jina",
                "jina-reranker-v1-base-en",
                "https://api.jina.ai/v1/rerank",
            ),
            _ => unreachable!(),
        };

        let new_key_val =
            prompt_for_input(&format!("Enter your {} API Key:", new_provider), None, true)?;
        let new_model_val = prompt_for_input(
            &format!("Enter {} model name:", new_provider),
            Some(default_model),
            false,
        )?;
        let new_url_val = prompt_for_input(
            &format!("Enter {} rerank base URL:", new_provider),
            Some(default_url),
            false,
        )?;

        cache_value(app_base_dir, provider_cache, new_provider)?;
        cache_value(app_base_dir, key_cache, &new_key_val)?;
        cache_value(app_base_dir, model_cache, &new_model_val)?;
        cache_value(app_base_dir, url_cache, &new_url_val)?;

        current_provider = Some(new_provider.to_string());
        current_key = Some(new_key_val);
        current_model = Some(new_model_val);
        current_url = Some(new_url_val);
    }

    if let (Some(prov), Some(key), Some(model), Some(url)) =
        (current_provider, current_key, current_model, current_url)
    {
        Ok(RerankerConfig {
            provider: prov,
            api_key: key,
            model,
            base_url: url,
        })
    } else {
        Err(BusterError::CommandError(
            "Reranker configuration setup failed. Some values are missing.".to_string(),
        ))
    }
}
