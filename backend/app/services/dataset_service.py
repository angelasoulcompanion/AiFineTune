"""
Dataset Service - Business logic for dataset management
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import aiofiles
import pandas as pd
from fastapi import UploadFile

from ..config import settings
from ..models.dataset import (
    DatasetFormat,
    DatasetList,
    DatasetPreview,
    DatasetResponse,
    DatasetStatus,
    DatasetType,
)
from ..repositories.dataset_repository import dataset_repository
from ..validators.dataset_validator import validate_dataset

logger = logging.getLogger(__name__)


class DatasetService:
    """Service for dataset operations"""

    def __init__(self):
        self.upload_dir = Path(settings.DATASETS_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def upload_dataset(
        self,
        user_id: UUID,
        file: UploadFile,
        name: str,
        description: Optional[str],
        dataset_type: DatasetType,
        column_mapping: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> DatasetResponse:
        """Upload and create a new dataset"""
        # Check if name exists
        if await dataset_repository.check_name_exists(user_id, name):
            raise ValueError(f"Dataset with name '{name}' already exists")

        # Determine format from filename
        suffix = Path(file.filename).suffix.lower()
        format_map = {
            ".jsonl": DatasetFormat.JSONL,
            ".json": DatasetFormat.JSON,
            ".csv": DatasetFormat.CSV,
            ".parquet": DatasetFormat.PARQUET,
        }

        if suffix not in format_map:
            raise ValueError(f"Unsupported file format: {suffix}. Use .jsonl, .json, .csv, or .parquet")

        dataset_format = format_map[suffix]

        # Create user directory
        user_dir = self.upload_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        import uuid as uuid_lib
        file_id = str(uuid_lib.uuid4())[:8]
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)
        file_name = f"{safe_name}_{file_id}{suffix}"
        file_path = user_dir / file_name

        # Save file and calculate hash
        hasher = hashlib.sha256()
        file_size = 0

        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(8192):
                await f.write(chunk)
                hasher.update(chunk)
                file_size += len(chunk)

        file_hash = hasher.hexdigest()

        # Create database record
        try:
            dataset = await dataset_repository.create(
                user_id=user_id,
                name=name,
                description=description,
                format=dataset_format.value,
                dataset_type=dataset_type.value,
                file_path=str(file_path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                source_type="upload",
                column_mapping=column_mapping,
                tags=tags,
            )

            logger.info(f"Dataset uploaded: {dataset.dataset_id} by user {user_id}")
            return dataset

        except Exception as e:
            # Cleanup file on error
            if file_path.exists():
                file_path.unlink()
            raise e

    async def get_dataset(self, dataset_id: UUID, user_id: UUID) -> Optional[DatasetResponse]:
        """Get dataset by ID (with user check)"""
        dataset = await dataset_repository.get_by_id(dataset_id)
        if dataset and dataset.user_id == user_id:
            return dataset
        return None

    async def list_datasets(
        self,
        user_id: UUID,
        page: int = 1,
        per_page: int = 20,
        status: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> DatasetList:
        """List datasets for a user"""
        datasets, total = await dataset_repository.get_by_user(
            user_id=user_id,
            page=page,
            per_page=per_page,
            status=status,
            dataset_type=dataset_type,
        )
        return DatasetList(
            datasets=datasets,
            total=total,
            page=page,
            per_page=per_page,
        )

    async def update_dataset(
        self,
        dataset_id: UUID,
        user_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        column_mapping: Optional[dict] = None,
    ) -> Optional[DatasetResponse]:
        """Update dataset metadata"""
        # Check ownership
        dataset = await self.get_dataset(dataset_id, user_id)
        if not dataset:
            return None

        # Check name uniqueness if changing
        if name and name != dataset.name:
            if await dataset_repository.check_name_exists(user_id, name, exclude_id=dataset_id):
                raise ValueError(f"Dataset with name '{name}' already exists")

        # Update
        return await dataset_repository.update(
            dataset_id,
            name=name,
            description=description,
            tags=tags,
            column_mapping=column_mapping,
        )

    async def delete_dataset(self, dataset_id: UUID, user_id: UUID) -> bool:
        """Delete a dataset"""
        dataset = await self.get_dataset(dataset_id, user_id)
        if not dataset:
            return False

        # Delete file
        file_path = Path(dataset.file_path)
        if file_path.exists():
            file_path.unlink()

        # Delete record
        return await dataset_repository.delete(dataset_id)

    async def validate_dataset(self, dataset_id: UUID, user_id: UUID) -> DatasetResponse:
        """Validate a dataset"""
        dataset = await self.get_dataset(dataset_id, user_id)
        if not dataset:
            raise ValueError("Dataset not found")

        # Update status to validating
        await dataset_repository.update_status(dataset_id, DatasetStatus.VALIDATING)

        try:
            # Run validation
            result = validate_dataset(dataset.file_path, dataset.dataset_type.value)

            # Update with results
            updated = await dataset_repository.update_validation(
                dataset_id=dataset_id,
                is_validated=True,
                validation_errors=[e.to_dict() for e in result.errors] if result.errors else None,
                validation_warnings=[w.to_dict() for w in result.warnings] if result.warnings else None,
                total_examples=result.total_examples,
                train_examples=result.train_examples,
                validation_examples=result.validation_examples,
                avg_input_length=result.avg_input_length,
                avg_output_length=result.avg_output_length,
            )

            logger.info(f"Dataset validated: {dataset_id}, valid={result.is_valid}")
            return updated

        except Exception as e:
            # Update status to error
            await dataset_repository.update_validation(
                dataset_id=dataset_id,
                is_validated=False,
                validation_errors=[{"row": 0, "column": "", "message": str(e), "severity": "error"}],
            )
            raise

    async def preview_dataset(
        self, dataset_id: UUID, user_id: UUID, limit: int = 10
    ) -> DatasetPreview:
        """Preview dataset rows"""
        dataset = await self.get_dataset(dataset_id, user_id)
        if not dataset:
            raise ValueError("Dataset not found")

        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise ValueError("Dataset file not found")

        rows = []
        columns = []
        total_rows = 0

        suffix = file_path.suffix.lower()

        if suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        total_rows += 1
                        if i < limit:
                            row = json.loads(line)
                            rows.append(row)
                            if not columns:
                                columns = list(row.keys())

        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                total_rows = len(data)
                rows = data[:limit]
                if rows:
                    columns = list(rows[0].keys())

        elif suffix == ".csv":
            df = pd.read_csv(file_path)
            total_rows = len(df)
            columns = df.columns.tolist()
            rows = df.head(limit).to_dict("records")

        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
            total_rows = len(df)
            columns = df.columns.tolist()
            rows = df.head(limit).to_dict("records")

        return DatasetPreview(
            dataset_id=dataset_id,
            format=dataset.format,
            columns=columns,
            rows=rows,
            total_rows=total_rows,
            preview_rows=len(rows),
        )

    async def get_dataset_statistics(self, dataset_id: UUID, user_id: UUID) -> dict[str, Any]:
        """Get detailed dataset statistics"""
        dataset = await self.get_dataset(dataset_id, user_id)
        if not dataset:
            raise ValueError("Dataset not found")

        return {
            "dataset_id": str(dataset.dataset_id),
            "name": dataset.name,
            "format": dataset.format.value,
            "dataset_type": dataset.dataset_type.value,
            "file_size_bytes": dataset.file_size_bytes,
            "file_size_mb": round(dataset.file_size_bytes / (1024 * 1024), 2) if dataset.file_size_bytes else 0,
            "total_examples": dataset.total_examples,
            "train_examples": dataset.train_examples,
            "validation_examples": dataset.validation_examples,
            "avg_input_length": dataset.avg_input_length,
            "avg_output_length": dataset.avg_output_length,
            "is_validated": dataset.is_validated,
            "status": dataset.status.value,
            "created_at": dataset.created_at.isoformat(),
        }


# Global instance
dataset_service = DatasetService()
