-- ============================================================
-- AiFineTune Platform - Initial Database Schema
-- Created: 2025-12-31
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. USERS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS finetune_users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,

    -- HuggingFace integration
    hf_token_encrypted TEXT,

    -- User preferences
    preferences JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_finetune_users_email ON finetune_users(email);
CREATE INDEX idx_finetune_users_username ON finetune_users(username);
CREATE INDEX idx_finetune_users_active ON finetune_users(is_active) WHERE is_active = TRUE;

-- ============================================================
-- 2. DATASETS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS finetune_datasets (
    dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES finetune_users(user_id) ON DELETE CASCADE,

    -- Dataset identification
    name VARCHAR(200) NOT NULL,
    description TEXT,
    format VARCHAR(20) NOT NULL CHECK (format IN ('jsonl', 'csv', 'parquet', 'json')),
    dataset_type VARCHAR(20) NOT NULL CHECK (dataset_type IN ('sft', 'dpo', 'orpo', 'chat')),

    -- File information
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64),  -- SHA-256

    -- Dataset statistics
    total_examples INTEGER,
    train_examples INTEGER,
    validation_examples INTEGER,
    avg_input_length INTEGER,
    avg_output_length INTEGER,

    -- Validation
    is_validated BOOLEAN DEFAULT FALSE,
    validation_errors JSONB,
    validation_warnings JSONB,

    -- Source information
    source_type VARCHAR(50) CHECK (source_type IN ('upload', 'database_export', 'huggingface')),
    source_reference TEXT,  -- HF dataset ID or table name

    -- Column mapping for custom formats
    column_mapping JSONB,

    -- Metadata
    tags TEXT[],
    metadata JSONB DEFAULT '{}',

    -- Status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'validating', 'ready', 'error', 'archived')),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(user_id, name)
);

CREATE INDEX idx_finetune_datasets_user ON finetune_datasets(user_id);
CREATE INDEX idx_finetune_datasets_status ON finetune_datasets(status);
CREATE INDEX idx_finetune_datasets_type ON finetune_datasets(dataset_type);
CREATE INDEX idx_finetune_datasets_created ON finetune_datasets(created_at DESC);

-- ============================================================
-- 3. MODELS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS finetune_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES finetune_users(user_id) ON DELETE CASCADE,

    -- Model identification
    name VARCHAR(200) NOT NULL,
    display_name VARCHAR(300),
    description TEXT,

    -- Model details
    model_type VARCHAR(20) NOT NULL CHECK (model_type IN ('base', 'lora', 'merged', 'gguf')),
    base_model_id VARCHAR(200) NOT NULL,  -- HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
    base_model_size VARCHAR(20),  -- '1.5B', '3B', '7B', '13B', '70B'

    -- For LoRA/trained models
    parent_model_id UUID REFERENCES finetune_models(model_id),
    training_job_id UUID,  -- Will be updated with FK after training_jobs table

    -- File information
    file_path TEXT,
    file_size_mb DOUBLE PRECISION,
    file_hash VARCHAR(64),

    -- HuggingFace integration
    hf_repo_id VARCHAR(200),
    is_pushed_to_hf BOOLEAN DEFAULT FALSE,
    hf_pushed_at TIMESTAMP WITH TIME ZONE,

    -- Ollama integration
    ollama_model_name VARCHAR(100),
    is_imported_to_ollama BOOLEAN DEFAULT FALSE,
    ollama_import_date TIMESTAMP WITH TIME ZONE,

    -- Performance metrics
    avg_inference_time_ms DOUBLE PRECISION,
    quality_score DOUBLE PRECISION,
    total_inferences INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'available' CHECK (status IN ('available', 'downloading', 'ready', 'error', 'archived')),
    is_favorite BOOLEAN DEFAULT FALSE,

    -- Metadata
    tags TEXT[],
    config JSONB,  -- Model config (quantization, architecture, etc.)
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(user_id, name)
);

CREATE INDEX idx_finetune_models_user ON finetune_models(user_id);
CREATE INDEX idx_finetune_models_type ON finetune_models(model_type);
CREATE INDEX idx_finetune_models_base ON finetune_models(base_model_id);
CREATE INDEX idx_finetune_models_status ON finetune_models(status);
CREATE INDEX idx_finetune_models_hf ON finetune_models(hf_repo_id) WHERE hf_repo_id IS NOT NULL;

-- ============================================================
-- 4. TRAINING JOBS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS finetune_training_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES finetune_users(user_id) ON DELETE CASCADE,

    -- Job identification
    name VARCHAR(200) NOT NULL,
    description TEXT,

    -- References
    dataset_id UUID REFERENCES finetune_datasets(dataset_id) NOT NULL,
    base_model_id UUID REFERENCES finetune_models(model_id) NOT NULL,
    output_model_id UUID REFERENCES finetune_models(model_id),

    -- Training method
    training_method VARCHAR(20) NOT NULL CHECK (training_method IN ('sft', 'lora', 'qlora', 'dpo', 'orpo', 'full')),

    -- Execution environment
    execution_env VARCHAR(20) NOT NULL CHECK (execution_env IN ('local', 'modal', 'hf_spaces', 'runpod')),
    device_type VARCHAR(20),  -- 'mps', 'cuda', 'cpu'

    -- Training configuration (full config as JSONB)
    config JSONB NOT NULL,
    /*
    Example config structure:
    {
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none"
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "optimizer": "adamw_8bit"
        },
        "quantization": {
            "load_in_4bit": true,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": true
        }
    }
    */

    -- Status tracking
    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN (
        'queued', 'preparing', 'training', 'evaluating', 'saving', 'uploading',
        'completed', 'failed', 'cancelled', 'paused'
    )),

    -- Progress tracking
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER,
    progress_percentage DOUBLE PRECISION DEFAULT 0.0,

    -- Metrics (updated during training)
    current_loss DOUBLE PRECISION,
    best_loss DOUBLE PRECISION,
    training_metrics JSONB DEFAULT '[]',  -- Array of {step, loss, lr, grad_norm, etc.}
    evaluation_metrics JSONB,  -- Final eval metrics {eval_loss, perplexity, etc.}

    -- Timing
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,

    -- Resource usage
    gpu_memory_used_mb INTEGER,
    peak_gpu_memory_mb INTEGER,
    total_training_time_seconds INTEGER,

    -- Error handling
    error_message TEXT,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- External references (for cloud training)
    external_job_id VARCHAR(200),  -- Modal/HF Spaces job ID
    external_logs_url TEXT,

    -- Output
    output_path TEXT,
    checkpoint_paths JSONB,  -- Array of checkpoint paths

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_jobs_user ON finetune_training_jobs(user_id);
CREATE INDEX idx_training_jobs_status ON finetune_training_jobs(status);
CREATE INDEX idx_training_jobs_dataset ON finetune_training_jobs(dataset_id);
CREATE INDEX idx_training_jobs_base_model ON finetune_training_jobs(base_model_id);
CREATE INDEX idx_training_jobs_created ON finetune_training_jobs(created_at DESC);
CREATE INDEX idx_training_jobs_active ON finetune_training_jobs(status)
    WHERE status IN ('queued', 'preparing', 'training', 'evaluating', 'saving');

-- Add FK from models to training_jobs
ALTER TABLE finetune_models
    ADD CONSTRAINT fk_models_training_job
    FOREIGN KEY (training_job_id)
    REFERENCES finetune_training_jobs(job_id);

-- ============================================================
-- 5. TRAINING TEMPLATES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS finetune_training_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES finetune_users(user_id) ON DELETE CASCADE,  -- NULL for system templates

    -- Template identification
    name VARCHAR(200) NOT NULL,
    description TEXT,
    training_method VARCHAR(20) NOT NULL CHECK (training_method IN ('sft', 'lora', 'qlora', 'dpo', 'orpo', 'full')),

    -- Default configuration
    config JSONB NOT NULL,

    -- Recommendations
    recommended_for TEXT[],  -- ['conversational', 'code', 'instruction', 'creative']
    base_model_recommendations TEXT[],  -- ['Qwen2.5', 'Llama3', 'Mistral']

    -- Visibility
    is_system BOOLEAN DEFAULT FALSE,  -- System-provided templates
    is_public BOOLEAN DEFAULT FALSE,  -- Visible to all users

    -- Usage tracking
    usage_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_templates_user ON finetune_training_templates(user_id);
CREATE INDEX idx_training_templates_method ON finetune_training_templates(training_method);
CREATE INDEX idx_training_templates_system ON finetune_training_templates(is_system) WHERE is_system = TRUE;

-- ============================================================
-- 6. TRIGGERS FOR updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_finetune_users_updated_at
    BEFORE UPDATE ON finetune_users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_finetune_datasets_updated_at
    BEFORE UPDATE ON finetune_datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_finetune_models_updated_at
    BEFORE UPDATE ON finetune_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_finetune_training_jobs_updated_at
    BEFORE UPDATE ON finetune_training_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_finetune_training_templates_updated_at
    BEFORE UPDATE ON finetune_training_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- 7. SEED SYSTEM TEMPLATES
-- ============================================================
INSERT INTO finetune_training_templates (name, description, training_method, config, recommended_for, base_model_recommendations, is_system, is_public) VALUES

-- QLoRA Template (Most Common)
('QLoRA - Fast & Efficient',
 '4-bit quantized LoRA training. Best for consumer GPUs with limited VRAM.',
 'qlora',
 '{
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none"
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "optimizer": "adamw_8bit",
        "lr_scheduler_type": "cosine"
    },
    "quantization": {
        "load_in_4bit": true,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": true
    }
 }',
 ARRAY['conversational', 'instruction', 'general'],
 ARRAY['Qwen2.5', 'Llama3.2', 'Mistral'],
 TRUE, TRUE),

-- LoRA Template (Better Quality)
('LoRA - Balanced',
 'Standard LoRA training. Better quality than QLoRA but requires more VRAM.',
 'lora',
 '{
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none"
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 2048,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine"
    }
 }',
 ARRAY['conversational', 'instruction', 'code'],
 ARRAY['Qwen2.5', 'Llama3.2', 'CodeLlama'],
 TRUE, TRUE),

-- DPO Template
('DPO - Preference Alignment',
 'Direct Preference Optimization for aligning model with human preferences.',
 'dpo',
 '{
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none"
    },
    "training": {
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "optimizer": "adamw_8bit"
    },
    "dpo": {
        "beta": 0.1,
        "loss_type": "sigmoid"
    },
    "quantization": {
        "load_in_4bit": true,
        "bnb_4bit_compute_dtype": "float16"
    }
 }',
 ARRAY['alignment', 'safety', 'preference'],
 ARRAY['Qwen2.5', 'Llama3.2'],
 TRUE, TRUE),

-- SFT Template (Simple)
('SFT - Supervised Fine-Tuning',
 'Basic supervised fine-tuning. Simple and effective for most use cases.',
 'sft',
 '{
    "lora": {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none"
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "optimizer": "adamw_8bit"
    },
    "quantization": {
        "load_in_4bit": true,
        "bnb_4bit_compute_dtype": "float16"
    }
 }',
 ARRAY['conversational', 'instruction', 'general'],
 ARRAY['Qwen2.5', 'Llama3.2', 'Mistral', 'Phi'],
 TRUE, TRUE);

-- ============================================================
-- DONE
-- ============================================================
