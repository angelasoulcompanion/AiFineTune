"""
Dataset Pydantic models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DatasetFormat(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"


class DatasetType(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    ORPO = "orpo"
    CHAT = "chat"


class DatasetStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class DatasetBase(BaseModel):
    """Base dataset model"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    dataset_type: DatasetType
    tags: Optional[list[str]] = None


class DatasetCreate(DatasetBase):
    """Model for creating dataset (metadata only)"""
    format: DatasetFormat
    column_mapping: Optional[dict[str, str]] = None


class DatasetUpdate(BaseModel):
    """Model for updating dataset metadata"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    column_mapping: Optional[dict[str, str]] = None


class DatasetStatistics(BaseModel):
    """Dataset statistics"""
    total_examples: int
    train_examples: int
    validation_examples: int
    avg_input_length: int
    avg_output_length: int


class DatasetValidation(BaseModel):
    """Dataset validation results"""
    is_valid: bool
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []


class DatasetResponse(DatasetBase):
    """Dataset response model"""
    dataset_id: UUID
    user_id: UUID
    format: DatasetFormat
    file_path: str
    file_size_bytes: Optional[int] = None
    file_hash: Optional[str] = None

    # Statistics
    total_examples: Optional[int] = None
    train_examples: Optional[int] = None
    validation_examples: Optional[int] = None
    avg_input_length: Optional[int] = None
    avg_output_length: Optional[int] = None

    # Validation
    is_validated: bool = False
    validation_errors: Optional[list[dict[str, Any]]] = None
    validation_warnings: Optional[list[dict[str, Any]]] = None

    # Source
    source_type: Optional[str] = None
    source_reference: Optional[str] = None

    # Mapping
    column_mapping: Optional[dict[str, str]] = None

    # Metadata
    metadata: dict[str, Any] = {}
    status: DatasetStatus

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetList(BaseModel):
    """Paginated dataset list"""
    datasets: list[DatasetResponse]
    total: int
    page: int
    per_page: int


class DatasetPreview(BaseModel):
    """Dataset preview response"""
    dataset_id: UUID
    format: DatasetFormat
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    preview_rows: int
