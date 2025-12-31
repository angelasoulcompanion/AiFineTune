"""
Training Job Pydantic models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TrainingMethod(str, Enum):
    SFT = "sft"
    LORA = "lora"
    QLORA = "qlora"
    DPO = "dpo"
    ORPO = "orpo"
    FULL = "full"


class ExecutionEnv(str, Enum):
    LOCAL = "local"
    MODAL = "modal"
    HF_SPACES = "hf_spaces"
    RUNPOD = "runpod"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING = "saving"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class LoRAConfig(BaseModel):
    """LoRA configuration"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    bias: str = "none"


class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    logging_steps: int = 10


class QuantizationConfig(BaseModel):
    """Quantization configuration for QLoRA"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class DPOConfig(BaseModel):
    """DPO-specific configuration"""
    beta: float = 0.1
    loss_type: str = "sigmoid"


class FullTrainingConfig(BaseModel):
    """Complete training configuration"""
    lora: Optional[LoRAConfig] = None
    training: TrainingConfig = TrainingConfig()
    quantization: Optional[QuantizationConfig] = None
    dpo: Optional[DPOConfig] = None


class TrainingJobCreate(BaseModel):
    """Model for creating a training job"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None

    # References
    dataset_id: UUID
    base_model_id: UUID

    # Training settings
    training_method: TrainingMethod
    execution_env: ExecutionEnv

    # Configuration
    config: FullTrainingConfig

    # Optional: use template
    template_id: Optional[UUID] = None


class TrainingJobUpdate(BaseModel):
    """Model for updating job (limited fields)"""
    name: Optional[str] = None
    description: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Training metrics at a point in time"""
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    timestamp: datetime


class EvaluationMetrics(BaseModel):
    """Evaluation metrics"""
    eval_loss: float
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    custom_metrics: dict[str, float] = {}


class TrainingJobResponse(BaseModel):
    """Training job response"""
    job_id: UUID
    user_id: UUID
    name: str
    description: Optional[str] = None

    # References
    dataset_id: UUID
    base_model_id: UUID
    output_model_id: Optional[UUID] = None

    # Training settings
    training_method: TrainingMethod
    execution_env: ExecutionEnv
    device_type: Optional[str] = None
    config: dict[str, Any]

    # Status
    status: JobStatus

    # Progress
    current_epoch: int = 0
    total_epochs: Optional[int] = None
    current_step: int = 0
    total_steps: Optional[int] = None
    progress_percentage: float = 0.0

    # Metrics
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    training_metrics: list[dict[str, Any]] = []
    evaluation_metrics: Optional[dict[str, Any]] = None

    # Timing
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    # Resource usage
    gpu_memory_used_mb: Optional[int] = None
    peak_gpu_memory_mb: Optional[int] = None
    total_training_time_seconds: Optional[int] = None

    # Error handling
    error_message: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None
    retry_count: int = 0

    # External
    external_job_id: Optional[str] = None
    external_logs_url: Optional[str] = None

    # Output
    output_path: Optional[str] = None
    checkpoint_paths: Optional[list[str]] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TrainingJobList(BaseModel):
    """Paginated training job list"""
    jobs: list[TrainingJobResponse]
    total: int
    page: int
    per_page: int


class TrainingJobProgress(BaseModel):
    """Real-time training progress (for WebSocket)"""
    job_id: UUID
    status: JobStatus
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    progress_percentage: float
    current_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    eta_seconds: Optional[int] = None


class TrainingTemplateResponse(BaseModel):
    """Training template response"""
    template_id: UUID
    user_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    training_method: TrainingMethod
    config: dict[str, Any]
    recommended_for: list[str] = []
    base_model_recommendations: list[str] = []
    is_system: bool = False
    is_public: bool = False
    usage_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TrainingTemplateList(BaseModel):
    """Training template list"""
    templates: list[TrainingTemplateResponse]
    total: int
