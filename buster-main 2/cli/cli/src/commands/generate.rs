use anyhow::{anyhow, Context, Result};
use colored::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
// use std::time::Duration; // Duration seems unused here now

use crate::utils::config::BusterConfig;
use crate::commands::init::{YamlModel, YamlDimension, YamlMeasure, is_measure_type};

// Use new struct names from dbt_utils
use dbt_utils::models::{DbtCatalog, CatalogNode, ColumnMetadata, TableMetadata}; // CatalogMetadata might not be directly used here
use dbt_utils::{run_dbt_docs_generate, load_and_parse_catalog};

use indicatif::{ProgressBar, ProgressStyle}; // Keep for progress spinners if any remain or are added
use inquire::Confirm;
use glob::{glob, Pattern};


pub async fn generate_semantic_models_command(
    path_arg: Option<String>,
    target_output_dir_arg: Option<String>, // This argument determines the *base* output dir if specified
) -> Result<()> {
    println!("{}", "🚀 Starting semantic model generation/update...".bold().blue());

    let buster_config_dir = std::env::current_dir().context("Failed to get current directory")?;
    let buster_config = BusterConfig::load_from_dir(&buster_config_dir)?.ok_or_else(|| {
        anyhow!("❌ buster.yml not found in {}. Please run 'buster init' first.", buster_config_dir.display())
    })?;

    // --- 1. Load Catalog & Build Lookup Map ---
    println!("\n{}", "🔍 Loading dbt catalog...".dimmed());
    let catalog_json_path = buster_config_dir.join("target").join("catalog.json");
    if Confirm::new("Run 'dbt docs generate' to refresh dbt catalog (catalog.json)?")
        .with_default(true)
        .prompt()?
    {
        match run_dbt_docs_generate(&buster_config_dir).await {
            Ok(_) => {},
            Err(e) => eprintln!("{}", format!("⚠️ 'dbt docs generate' error. Proceeding with existing catalog. Error: {}", e).yellow()),
        }
    } else {
        println!("{}", "ℹ️ Skipping 'dbt docs generate'. Using existing catalog.json.".dimmed());
    }

    if !catalog_json_path.exists() {
        eprintln!("{}", format!("❌ catalog.json not found at {}. Cannot generate/update models.", catalog_json_path.display()).red());
        return Ok(());
    }
    let dbt_catalog = match load_and_parse_catalog(&catalog_json_path) {
        Ok(catalog) => {
            println!("{}", "✅ Successfully parsed catalog.json.".green());
            catalog
        }
        Err(e) => {
            eprintln!("{}", format!("❌ Error loading/parsing catalog.json: {}. Cannot generate/update.", e).red());
            return Ok(());
        }
    };

    let catalog_nodes_by_name: HashMap<String, &CatalogNode> = dbt_catalog.nodes.values()
        .filter_map(|node| {
            if node.derived_resource_type.as_deref() == Some("model") {
                node.derived_model_name_from_file.as_ref().map(|name| (name.clone(), node))
            } else { None }
        })
        .collect();

    if catalog_nodes_by_name.is_empty() {
        println!("{}", "ℹ️ No models found in dbt catalog. Nothing to generate/update.".yellow());
        return Ok(());
    }

    // --- 2. Determine SQL Files to Process (based on path_arg or buster.yml model_paths) ---
    println!("\n{}", "⚙️ Determining SQL files to process...".dimmed());
    let mut sql_files_to_process: HashSet<PathBuf> = HashSet::new();
    let dbt_project_model_roots_for_stripping = crate::commands::init::parse_dbt_project_file_content(&buster_config_dir)?.as_ref()
        .map(|c| c.model_paths.iter().map(PathBuf::from).collect::<Vec<PathBuf>>())
        .unwrap_or_else(|| vec![PathBuf::from("models")]);

    if let Some(pa_str) = &path_arg {
        let target_path = buster_config_dir.join(pa_str);
        if target_path.is_file() && target_path.extension().map_or(false, |ext| ext == "sql") {
            sql_files_to_process.insert(target_path);
        } else if target_path.is_dir() {
            let glob_pattern = target_path.join("**/*.sql");
            match glob(&glob_pattern.to_string_lossy()) {
                Ok(paths) => paths.for_each(|entry| if let Ok(path) = entry { if path.is_file() { sql_files_to_process.insert(path); } }),
                Err(e) => eprintln!("{}", format!("Error globbing '{}': {}", glob_pattern.display(), e).yellow()),
            }
        } else {
            eprintln!("{}", format!("⚠️ Warning: path_arg '{}' is not a valid SQL file or directory. Processing all configured models.", pa_str).yellow());
            // Fall through to buster.yml model_paths if path_arg is invalid
        }
    }
    
    if sql_files_to_process.is_empty() { // If path_arg didn't yield files, or wasn't provided, use buster.yml config
        let mut processed_via_buster_yml_paths = false;
        if let Some(projects) = &buster_config.projects {
            if let Some(first_project) = projects.first() {
                if let Some(config_model_paths) = &first_project.model_paths { // Vec<String>
                    if !config_model_paths.is_empty() { // Check if there are paths to process
                        println!("{}", format!("ℹ️ No SQL files from path_arg. Scanning based on buster.yml model_paths: {:?}", config_model_paths).dimmed());
                        for path_entry_from_config in config_model_paths {
                            if path_entry_from_config.trim().is_empty() {
                                continue; // Skip empty path strings
                            }
                            let final_glob_pattern_str: String;
                            let path_is_absolute = Path::new(path_entry_from_config).is_absolute();

                            let base_path_for_glob = if path_is_absolute {
                                PathBuf::from(path_entry_from_config)
                            } else {
                                buster_config_dir.join(path_entry_from_config)
                            };

                            if path_entry_from_config.contains('*') || path_entry_from_config.contains('?') || path_entry_from_config.contains('[') {
                                final_glob_pattern_str = base_path_for_glob.to_string_lossy().into_owned();
                            } else {
                                final_glob_pattern_str = base_path_for_glob.join("**/*.sql").to_string_lossy().into_owned();
                            }

                            match glob(&final_glob_pattern_str) {
                                Ok(paths) => paths.for_each(|entry| {
                                    if let Ok(p) = entry {
                                        if p.is_file() && p.extension().map_or(false, |e| e == "sql") {
                                            sql_files_to_process.insert(p);
                                        }
                                    }
                                }),
                                Err(e) => eprintln!("{}", format!("Glob pattern error for buster.yml path '{}': {}", final_glob_pattern_str, e).yellow()),
                            }
                        }
                        // If config_model_paths had at least one non-empty string, consider this path taken
                        if config_model_paths.iter().any(|s| !s.trim().is_empty()) {
                             processed_via_buster_yml_paths = true;
                        }
                    }
                }
            }
        }

        if !processed_via_buster_yml_paths {
            // Fallback to dbt_project.yml defaults
            println!("{}", "ℹ️ No SQL files from path_arg, and no model_paths in buster.yml (or they were empty). Using dbt_project.yml model-paths as fallback.".dimmed());
            for dbt_root_rel in &dbt_project_model_roots_for_stripping {
                let glob_pattern = buster_config_dir.join(dbt_root_rel).join("**/*.sql");
                match glob(&glob_pattern.to_string_lossy()) {
                    Ok(paths) => paths.for_each(|entry| if let Ok(p) = entry { if p.is_file() {sql_files_to_process.insert(p);}}),
                    Err(e) => eprintln!("{}", format!("Error globbing default dbt path '{}': {}",glob_pattern.display(),e).yellow()),
                }
            }
        }
    }

    if sql_files_to_process.is_empty() {
        println!("{}", "ℹ️ No SQL model files found to process/update based on configuration.".yellow());
        return Ok(());
    }
    println!("{}", format!("✅ Found {} SQL model file(s) for potential generation/update.", sql_files_to_process.len()).dimmed());

    // --- 3. Determine Output Base Directory for Semantic Models ---
    println!("\n{}", "⚙️ Determining output directory for semantic models...".dimmed());
    let semantic_models_base_dir_path_str = target_output_dir_arg.or_else(|| 
        buster_config.projects.as_ref()
            .and_then(|p| p.first())
            .and_then(|proj| proj.semantic_model_paths.as_ref())
            .and_then(|paths| paths.first().cloned())
    ).unwrap_or_else(|| 
        // Default to side-by-side (empty string means relative to SQL file later)
        String::new()
    );

    let (is_side_by_side_generation, semantic_output_base_abs_dir) = if semantic_models_base_dir_path_str.is_empty() {
        println!("{}", "ℹ️ Semantic models output set to side-by-side with SQL files.".dimmed());
        (true, buster_config_dir.clone()) // Base for side-by-side is project root
    } else {
        let abs_path = if Path::new(&semantic_models_base_dir_path_str).is_absolute() {
            PathBuf::from(&semantic_models_base_dir_path_str)
        } else {
            buster_config_dir.join(&semantic_models_base_dir_path_str)
        };
        println!("{}", format!("➡️ Semantic models output base directory: {}", abs_path.display()).cyan());
        fs::create_dir_all(&abs_path).context(format!("Failed to create semantic models output dir: {}", abs_path.display()))?;
        (false, abs_path)
    };
    
    // --- 4. Iterate SQL Files, Match to Catalog, Generate/Update YamlModels ---
    println!("\n{}", "✨ Processing SQL files and generating/updating YAML models...".dimmed());
    let mut models_generated_count = 0;
    let mut models_updated_count = 0;
    let mut columns_added_count = 0;
    let mut columns_updated_count = 0;
    let mut columns_removed_count = 0;
    let mut sql_models_successfully_processed_from_catalog_count = 0; // New counter

    // Get project defaults for comparison
    let proj_default_ds_name = buster_config.projects.as_ref()
        .and_then(|p| p.first()).and_then(|pc| pc.data_source_name.as_deref());
    let proj_default_database = buster_config.projects.as_ref()
        .and_then(|p| p.first()).and_then(|pc| pc.database.as_deref());
    let proj_default_schema = buster_config.projects.as_ref()
        .and_then(|p| p.first()).and_then(|pc| pc.schema.as_deref());

    for sql_file_abs_path in sql_files_to_process {
        let model_name_from_filename = sql_file_abs_path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        if model_name_from_filename.is_empty() {
            eprintln!("{}", format!("⚠️ Warning: Could not get model name from file {}. Skipping.", sql_file_abs_path.display()).yellow());
            continue;
        }

        let Some(catalog_node) = catalog_nodes_by_name.get(&model_name_from_filename) else {
            eprintln!("{}", format!("ℹ️ Info: SQL model file '{}' found, but no corresponding entry ('{}') in dbt catalog. Skipping.", sql_file_abs_path.display(), model_name_from_filename).dimmed());
            continue;
        };

        let Some(ref table_meta) = catalog_node.metadata else {
            eprintln!("{}", format!("⚠️ Warning: Catalog entry for '{}' (file: {}) is missing metadata. Skipping.", model_name_from_filename, sql_file_abs_path.display()).yellow());
            continue;
        };
        // actual_model_name_in_yaml is from catalog metadata.name
        let actual_model_name_in_yaml = table_meta.name.clone(); 

        println!("➡️ Processing: SQL '{}' -> Catalog Model '{}' (UniqueID: {})", 
            sql_file_abs_path.display().to_string().cyan(), 
            actual_model_name_in_yaml.purple(),
            catalog_node.unique_id.as_deref().unwrap_or("N/A").dimmed()
        );
        sql_models_successfully_processed_from_catalog_count += 1; // Increment here
        
        let relative_sql_path_str = pathdiff::diff_paths(&sql_file_abs_path, &buster_config_dir)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| sql_file_abs_path.to_string_lossy().into_owned());

        let individual_semantic_yaml_path: PathBuf = if is_side_by_side_generation {
            sql_file_abs_path.with_extension("yml")
        } else {
            let mut stripped_suffix_for_yaml: Option<PathBuf> = None;
            for dbt_root in &dbt_project_model_roots_for_stripping {
                let abs_dbt_root = buster_config_dir.join(dbt_root);
                if let Ok(stripped) = sql_file_abs_path.strip_prefix(&abs_dbt_root) {
                    stripped_suffix_for_yaml = Some(stripped.with_extension("yml"));
                    break;
                }
            }
            let final_suffix = stripped_suffix_for_yaml.unwrap_or_else(|| PathBuf::from(&model_name_from_filename).with_extension("yml"));
            semantic_output_base_abs_dir.join(final_suffix)
        };
        if let Some(p) = individual_semantic_yaml_path.parent() { fs::create_dir_all(p)?; }

        // --- Reconciliation Logic (Create or Update) ---
        let existing_yaml_model_opt: Option<YamlModel> = if individual_semantic_yaml_path.exists() {
            fs::read_to_string(&individual_semantic_yaml_path)
                .ok()
                .and_then(|content| serde_yaml::from_str::<YamlModel>(&content).ok())
        } else { None };

        match existing_yaml_model_opt {
            Some(mut existing_model) => {
                let mut model_was_updated = false;
                // Update name if it differs (dbt catalog is source of truth for relation name)
                if existing_model.name != actual_model_name_in_yaml {
                    existing_model.name = actual_model_name_in_yaml.clone(); model_was_updated = true;
                }

                // Preserve manual description, otherwise update from catalog if catalog has one.
                let placeholder_desc = "Description missing - please update.".to_string();
                match &existing_model.description {
                    Some(existing_desc) if existing_desc != &placeholder_desc => {
                        // Manual description exists and is not the placeholder, do nothing to preserve it.
                    }
                    _ => { // Existing is None or is the placeholder
                        if table_meta.comment.is_some() && existing_model.description != table_meta.comment {
                            existing_model.description = table_meta.comment.clone();
                            model_was_updated = true;
                        }
                    }
                }

                // Preserve manual database override, otherwise update from catalog.
                if existing_model.database.is_none() {
                    let cat_db_from_meta = &table_meta.database; // Option<String>
                    let new_yaml_db = cat_db_from_meta.as_ref()
                        .filter(|cat_db_val_str_ref| proj_default_database != Some(cat_db_val_str_ref.as_str()))
                        .cloned();
                    if existing_model.database != new_yaml_db { // Check if it actually changes
                        existing_model.database = new_yaml_db;
                        model_was_updated = true;
                    }
                } // If Some, it's preserved.

                // Preserve manual schema override, otherwise update from catalog.
                if existing_model.schema.is_none() {
                    let cat_schema_from_meta = &table_meta.schema; // String
                    let new_yaml_schema = if proj_default_schema.as_deref() == Some(cat_schema_from_meta.as_str()) {
                        None
                    } else {
                        Some(cat_schema_from_meta.clone())
                    };
                    if existing_model.schema != new_yaml_schema { // Check if it actually changes
                        existing_model.schema = new_yaml_schema;
                        model_was_updated = true;
                    }
                } // If Some, it's preserved.

                // Reconcile columns
                let mut current_dims: Vec<YamlDimension> = Vec::new();
                let mut current_measures: Vec<YamlMeasure> = Vec::new();
                let mut dbt_columns_map: HashMap<String, &ColumnMetadata> = catalog_node.columns.values().map(|c| (c.name.clone(), c)).collect();

                for existing_dim in std::mem::take(&mut existing_model.dimensions) {
                    if let Some(dbt_col) = dbt_columns_map.get(&existing_dim.name) { // Use .get() to keep it in map for measure pass
                        let mut updated_dim = existing_dim.clone();
                        let mut dim_col_updated = false;

                        if !crate::commands::init::is_measure_type(&dbt_col.type_) { // Still a dimension
                            // Preserve manual type if Some, otherwise update from catalog.
                            if updated_dim.type_.is_none() {
                                if updated_dim.type_.as_deref() != Some(&dbt_col.type_) { // Check if it actually changes
                                    updated_dim.type_ = Some(dbt_col.type_.clone());
                                    dim_col_updated = true;
                                }
                            }

                            // Preserve manual description if Some and not placeholder, otherwise update from catalog.
                            let placeholder_col_desc = "Description missing - please update.".to_string();
                            match &updated_dim.description {
                                Some(existing_col_desc) if existing_col_desc != &placeholder_col_desc => {
                                    // Manual description exists and is not placeholder, do nothing.
                                }
                                _ => { // Existing is None or is placeholder
                                    let new_description_from_catalog = dbt_col.comment.as_ref().filter(|s| !s.is_empty()).cloned();
                                    if updated_dim.description != new_description_from_catalog {
                                        updated_dim.description = new_description_from_catalog.or_else(|| Some(placeholder_col_desc));
                                        dim_col_updated = true;
                                    }
                                }
                            }

                            // Preserve existing_dim.searchable and existing_dim.options, so no changes needed here for them.
                            // If updated_dim.searchable was true, it remains true.
                            // If updated_dim.options was Some, it remains Some.

                            if dim_col_updated { columns_updated_count +=1; model_was_updated = true; }
                            current_dims.push(updated_dim);
                            dbt_columns_map.remove(&existing_dim.name); // Consume it now that it's processed as a dim
                        } else { // Was a dimension, but is now a measure according to dbt_col type
                            println!("{}", format!("   ✏️ Column '{}' changed from Dimension to Measure. It will be re-added as Measure.", existing_dim.name).yellow());
                            columns_removed_count += 1; // Count as removed dimension
                            model_was_updated = true;
                            // Do not remove from dbt_columns_map yet, it will be picked up as a new measure.
                        }
                    } else { // Dimension no longer in catalog
                        println!("{}", format!("   ➖ Dimension '{}' removed (not in catalog).", existing_dim.name).yellow());
                        columns_removed_count += 1; model_was_updated = true;
                        // dbt_columns_map.remove(&existing_dim.name); // Not needed, it's not in the map
                    }
                }

                for existing_measure in std::mem::take(&mut existing_model.measures) {
                    if let Some(dbt_col) = dbt_columns_map.get(&existing_measure.name) { // Use .get() initially
                        let mut updated_measure = existing_measure.clone();
                        let mut measure_col_updated = false;

                        if crate::commands::init::is_measure_type(&dbt_col.type_) { // Still a measure
                            // Preserve manual type if Some, otherwise update from catalog.
                            if updated_measure.type_.is_none() {
                                if updated_measure.type_.as_deref() != Some(&dbt_col.type_) { // Check if it actually changes
                                    updated_measure.type_ = Some(dbt_col.type_.clone());
                                    measure_col_updated = true;
                                }
                            }

                            // Preserve manual description if Some and not placeholder, otherwise update from catalog.
                            let placeholder_col_desc = "Description missing - please update.".to_string();
                            match &updated_measure.description {
                                Some(existing_col_desc) if existing_col_desc != &placeholder_col_desc => {
                                    // Manual description exists and is not placeholder, do nothing.
                                }
                                _ => { // Existing is None or is placeholder
                                    let new_description_from_catalog = dbt_col.comment.as_ref().filter(|s| !s.is_empty()).cloned();
                                    if updated_measure.description != new_description_from_catalog {
                                        updated_measure.description = new_description_from_catalog.or_else(|| Some(placeholder_col_desc));
                                        measure_col_updated = true;
                                    }
                                }
                            }

                            if measure_col_updated { columns_updated_count +=1; model_was_updated = true; }
                            current_measures.push(updated_measure);
                            dbt_columns_map.remove(&existing_measure.name); // Consume it
                        } else { // Was a measure, but is now a dimension
                            println!("{}", format!("   ✏️ Column '{}' changed from Measure to Dimension. It will be re-added as Dimension.", existing_measure.name).cyan());
                            columns_removed_count += 1; // Count as removed measure
                            model_was_updated = true;
                            // Do not remove from dbt_columns_map yet, it will be picked up as a new dimension.
                        }
                    } else { // Measure no longer in catalog
                        println!("{}", format!("   ➖ Measure '{}' removed (not in catalog).", existing_measure.name).yellow());
                        columns_removed_count += 1; model_was_updated = true;
                        // dbt_columns_map.remove(&existing_measure.name); // Not needed
                    }
                }
                for (_col_name, dbt_col) in dbt_columns_map { // Remaining are new columns
                    if crate::commands::init::is_measure_type(&dbt_col.type_) { 
                        current_measures.push(YamlMeasure { 
                            name: dbt_col.name.clone(), 
                            description: dbt_col.comment.as_ref().filter(|s| !s.is_empty()).cloned().or_else(|| Some("Description missing - please update.".to_string())), 
                            type_: Some(dbt_col.type_.clone()) 
                        });
                    } else {
                        current_dims.push(YamlDimension { 
                            name: dbt_col.name.clone(), 
                            description: dbt_col.comment.as_ref().filter(|s| !s.is_empty()).cloned().or_else(|| Some("Description missing - please update.".to_string())), 
                            type_: Some(dbt_col.type_.clone()), 
                            searchable: false, // Ensure searchable is false 
                            options: None 
                        });
                    }
                    columns_added_count += 1; model_was_updated = true;
                }
                existing_model.dimensions = current_dims;
                existing_model.measures = current_measures;
                
                if model_was_updated {
                    models_updated_count += 1;
                    let yaml_string = serde_yaml::to_string(&existing_model)?;
                    fs::write(&individual_semantic_yaml_path, yaml_string)?;
                    println!("   {} Updated semantic model: {}", "✅".cyan(), individual_semantic_yaml_path.display().to_string().cyan());
                } else {
                    println!("   {} No changes needed for existing model: {}", "➖".dimmed(), individual_semantic_yaml_path.display().to_string().dimmed());
                }
            }
            None => { // New semantic model
                let mut dimensions = Vec::new();
                let mut measures = Vec::new();
                for (_col_name, col_meta) in &catalog_node.columns {
                    if crate::commands::init::is_measure_type(&col_meta.type_) { // type_ is String
                        measures.push(YamlMeasure { 
                            name: col_meta.name.clone(), 
                            description: col_meta.comment.as_ref().filter(|s| !s.is_empty()).cloned().or_else(|| Some("Description missing - please update.".to_string())), 
                            type_: Some(col_meta.type_.clone()) 
                        });
                    } else {
                        dimensions.push(YamlDimension { 
                            name: col_meta.name.clone(), 
                            description: col_meta.comment.as_ref().filter(|s| !s.is_empty()).cloned().or_else(|| Some("Description missing - please update.".to_string())), 
                            type_: Some(col_meta.type_.clone()), 
                            searchable: false, // Ensure searchable is false
                            options: None 
                        });
                    }
                }
                let new_model = YamlModel {
                    name: actual_model_name_in_yaml,
                    description: table_meta.comment.clone(),
                    data_source_name: None, // Per user request, dbt catalog doesn't provide this, so imply project default for new models
                    database: {
                        let model_db_from_catalog = &table_meta.database; // Option<String>
                        model_db_from_catalog.as_ref()
                            .filter(|catalog_db_str_ref| proj_default_database != Some(catalog_db_str_ref.as_str()))
                            .cloned()
                    },
                    schema: {
                        let model_schema_from_catalog = &table_meta.schema; // String
                        if proj_default_schema.as_deref() == Some(model_schema_from_catalog.as_str()) {
                            None
                        } else {
                            Some(model_schema_from_catalog.clone())
                        }
                    },
                    dimensions,
                    measures,
                };
                let yaml_string = serde_yaml::to_string(&new_model)?;
                fs::write(&individual_semantic_yaml_path, yaml_string)?;
                models_generated_count += 1;
                println!("   {} Generated new semantic model: {}", "✨".green(), individual_semantic_yaml_path.display().to_string().green());
            }
        }
    }

    println!("\n{}", "📊 Semantic Model Generation/Update Summary:".bold().green());
    println!("  --------------------------------------------------");
    println!("  SQL models processed with catalog entry: {}", sql_models_successfully_processed_from_catalog_count.to_string().cyan()); 
    println!("  New semantic models generated        : {}", models_generated_count.to_string().green());
    println!("  Existing semantic models updated     : {}", models_updated_count.to_string().cyan());
    println!("  Columns added to existing models     : {}", columns_added_count.to_string().green());
    println!("  Columns updated in existing models   : {}", columns_updated_count.to_string().cyan());
    println!("  Columns removed from existing models : {}", columns_removed_count.to_string().red());
    println!("  --------------------------------------------------");
    if sql_models_successfully_processed_from_catalog_count == 0 && models_generated_count == 0 && models_updated_count == 0 {
        println!("{}", "ℹ️ No models were generated or updated.".yellow());
    } else {
        println!("🎉 Semantic model generation/update complete.");
    }

    Ok(())
} 