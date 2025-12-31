"""
Model (LLM) Pydantic models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    BASE = "base"
    LORA = "lora"
    MERGED = "merged"
    GGUF = "gguf"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class ModelBase(BaseModel):
    """Base model"""
    name: str = Field(..., min_length=1, max_length=200)
    display_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None


class ModelCreate(ModelBase):
    """Model for registering a new model"""
    model_type: ModelType
    base_model_id: str = Field(..., min_length=1)  # HuggingFace model ID
    base_model_size: Optional[str] = None
    config: Optional[dict[str, Any]] = None


class ModelUpdate(BaseModel):
    """Model for updating model metadata"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    is_favorite: Optional[bool] = None


class ModelResponse(ModelBase):
    """Model response"""
    model_id: UUID
    user_id: Optional[UUID] = None

    # Model details
    model_type: ModelType
    base_model_id: str
    base_model_size: Optional[str] = None

    # Relationships
    parent_model_id: Optional[UUID] = None
    training_job_id: Optional[UUID] = None

    # File info
    file_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    file_hash: Optional[str] = None

    # HuggingFace integration
    hf_repo_id: Optional[str] = None
    is_pushed_to_hf: bool = False
    hf_pushed_at: Optional[datetime] = None

    # Ollama integration
    ollama_model_name: Optional[str] = None
    is_imported_to_ollama: bool = False

    # Metrics
    avg_inference_time_ms: Optional[float] = None
    quality_score: Optional[float] = None
    total_inferences: int = 0

    # Status
    status: ModelStatus
    is_favorite: bool = False

    # Config & Metadata
    config: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ModelList(BaseModel):
    """Paginated model list"""
    models: list[ModelResponse]
    total: int
    page: int
    per_page: int


class HFModelSearchResult(BaseModel):
    """HuggingFace model search result"""
    model_id: str
    author: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = None
    tags: list[str] = []
    last_modified: Optional[str] = None


class HFModelSearchResponse(BaseModel):
    """HuggingFace search response"""
    results: list[HFModelSearchResult]
    total: int
    query: str


class ModelDownloadRequest(BaseModel):
    """Request to download model from HuggingFace"""
    hf_model_id: str
    name: str
    quantization: Optional[str] = None  # 'int4', 'int8', 'fp16'


class ModelDownloadProgress(BaseModel):
    """Model download progress"""
    model_id: UUID
    hf_model_id: str
    status: str  # 'downloading', 'extracting', 'ready', 'error'
    progress: float  # 0.0 to 1.0
    downloaded_bytes: int
    total_bytes: int
    error: Optional[str] = None


class ModelTestRequest(BaseModel):
    """Request to test model with prompt"""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    system_prompt: Optional[str] = None


class ModelTestResponse(BaseModel):
    """Model test response"""
    model_id: UUID
    prompt: str
    response: str
    inference_time_ms: float
    tokens_generated: int
