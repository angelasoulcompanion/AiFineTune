"""
Dataset Repository - Database access for datasets
"""

from typing import Optional
from uuid import UUID

from ..database import db
from ..models.dataset import (
    DatasetCreate,
    DatasetResponse,
    DatasetStatus,
    DatasetUpdate,
)


class DatasetRepository:
    """Repository for dataset database operations"""

    async def create(
        self,
        user_id: UUID,
        name: str,
        description: Optional[str],
        format: str,
        dataset_type: str,
        file_path: str,
        file_size_bytes: Optional[int] = None,
        file_hash: Optional[str] = None,
        source_type: str = "upload",
        source_reference: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> DatasetResponse:
        """Create a new dataset record"""
        row = await db.fetchrow(
            """
            INSERT INTO finetune_datasets (
                user_id, name, description, format, dataset_type,
                file_path, file_size_bytes, file_hash,
                source_type, source_reference, column_mapping, tags, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'pending')
            RETURNING *
            """,
            user_id,
            name,
            description,
            format,
            dataset_type,
            file_path,
            file_size_bytes,
            file_hash,
            source_type,
            source_reference,
            column_mapping,
            tags,
        )
        return self._row_to_response(row)

    async def get_by_id(self, dataset_id: UUID) -> Optional[DatasetResponse]:
        """Get dataset by ID"""
        row = await db.fetchrow(
            "SELECT * FROM finetune_datasets WHERE dataset_id = $1",
            dataset_id,
        )
        return self._row_to_response(row) if row else None

    async def get_by_user(
        self,
        user_id: UUID,
        page: int = 1,
        per_page: int = 20,
        status: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> tuple[list[DatasetResponse], int]:
        """Get datasets for a user with pagination"""
        # Build query
        conditions = ["user_id = $1"]
        params = [user_id]
        param_idx = 2

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if dataset_type:
            conditions.append(f"dataset_type = ${param_idx}")
            params.append(dataset_type)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Get total count
        total = await db.fetchval(
            f"SELECT COUNT(*) FROM finetune_datasets WHERE {where_clause}",
            *params,
        )

        # Get paginated results
        offset = (page - 1) * per_page
        rows = await db.fetch(
            f"""
            SELECT * FROM finetune_datasets
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """,
            *params,
            per_page,
            offset,
        )

        datasets = [self._row_to_response(row) for row in rows]
        return datasets, total

    async def update(
        self, dataset_id: UUID, **kwargs
    ) -> Optional[DatasetResponse]:
        """Update dataset fields"""
        if not kwargs:
            return await self.get_by_id(dataset_id)

        # Build update query
        updates = []
        params = [dataset_id]
        param_idx = 2

        for key, value in kwargs.items():
            if value is not None:
                updates.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not updates:
            return await self.get_by_id(dataset_id)

        query = f"""
            UPDATE finetune_datasets
            SET {', '.join(updates)}
            WHERE dataset_id = $1
            RETURNING *
        """

        row = await db.fetchrow(query, *params)
        return self._row_to_response(row) if row else None

    async def update_status(
        self, dataset_id: UUID, status: DatasetStatus
    ) -> Optional[DatasetResponse]:
        """Update dataset status"""
        return await self.update(dataset_id, status=status.value)

    async def update_validation(
        self,
        dataset_id: UUID,
        is_validated: bool,
        validation_errors: Optional[list] = None,
        validation_warnings: Optional[list] = None,
        total_examples: Optional[int] = None,
        train_examples: Optional[int] = None,
        validation_examples: Optional[int] = None,
        avg_input_length: Optional[int] = None,
        avg_output_length: Optional[int] = None,
    ) -> Optional[DatasetResponse]:
        """Update dataset validation results"""
        row = await db.fetchrow(
            """
            UPDATE finetune_datasets
            SET is_validated = $2,
                validation_errors = $3,
                validation_warnings = $4,
                total_examples = $5,
                train_examples = $6,
                validation_examples = $7,
                avg_input_length = $8,
                avg_output_length = $9,
                status = CASE WHEN $2 AND ($3 IS NULL OR jsonb_array_length($3) = 0) THEN 'ready' ELSE 'error' END
            WHERE dataset_id = $1
            RETURNING *
            """,
            dataset_id,
            is_validated,
            validation_errors,
            validation_warnings,
            total_examples,
            train_examples,
            validation_examples,
            avg_input_length,
            avg_output_length,
        )
        return self._row_to_response(row) if row else None

    async def delete(self, dataset_id: UUID) -> bool:
        """Delete a dataset"""
        result = await db.execute(
            "DELETE FROM finetune_datasets WHERE dataset_id = $1",
            dataset_id,
        )
        return "DELETE 1" in result

    async def check_name_exists(self, user_id: UUID, name: str, exclude_id: Optional[UUID] = None) -> bool:
        """Check if dataset name already exists for user"""
        if exclude_id:
            count = await db.fetchval(
                "SELECT COUNT(*) FROM finetune_datasets WHERE user_id = $1 AND name = $2 AND dataset_id != $3",
                user_id,
                name,
                exclude_id,
            )
        else:
            count = await db.fetchval(
                "SELECT COUNT(*) FROM finetune_datasets WHERE user_id = $1 AND name = $2",
                user_id,
                name,
            )
        return count > 0

    def _row_to_response(self, row) -> DatasetResponse:
        """Convert database row to response model"""
        return DatasetResponse(
            dataset_id=row["dataset_id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            format=row["format"],
            dataset_type=row["dataset_type"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"],
            file_hash=row["file_hash"],
            total_examples=row["total_examples"],
            train_examples=row["train_examples"],
            validation_examples=row["validation_examples"],
            avg_input_length=row["avg_input_length"],
            avg_output_length=row["avg_output_length"],
            is_validated=row["is_validated"],
            validation_errors=row["validation_errors"],
            validation_warnings=row["validation_warnings"],
            source_type=row["source_type"],
            source_reference=row["source_reference"],
            column_mapping=row["column_mapping"],
            tags=row["tags"] or [],
            metadata=row["metadata"] or {},
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Global instance
dataset_repository = DatasetRepository()
