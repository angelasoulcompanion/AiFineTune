"""
Training Job Repository - Database operations for finetune_training_jobs
"""
from typing import Optional
from uuid import UUID
from datetime import datetime
import json

from ..database import get_db_pool


class TrainingJobRepository:
    """Repository for training job database operations"""

    async def create(
        self,
        user_id: UUID,
        dataset_id: UUID,
        base_model_id: UUID,
        job_name: str,
        training_method: str,
        execution_env: str = "local",
        config: Optional[dict] = None,
    ) -> dict:
        """Create a new training job"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO finetune_training_jobs (
                    user_id, dataset_id, base_model_id, name,
                    training_method, execution_env, config, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'queued')
                RETURNING *
                """,
                user_id, dataset_id, base_model_id, job_name,
                training_method, execution_env, json.dumps(config) if config else None
            )
            return self._row_to_dict(row)

    async def get_by_id(self, job_id: UUID) -> Optional[dict]:
        """Get training job by ID"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT tj.*,
                       d.name as dataset_name,
                       d.file_path as dataset_path,
                       d.dataset_type,
                       m.name as model_name,
                       m.file_path as model_path,
                       m.base_model_id as hf_model_id
                FROM finetune_training_jobs tj
                LEFT JOIN finetune_datasets d ON tj.dataset_id = d.dataset_id
                LEFT JOIN finetune_models m ON tj.base_model_id = m.model_id
                WHERE tj.job_id = $1
                """,
                job_id
            )
            return self._row_to_dict(row) if row else None

    async def get_by_user(
        self,
        user_id: UUID,
        page: int = 1,
        per_page: int = 20,
        status: Optional[str] = None,
    ) -> tuple[list[dict], int]:
        """Get training jobs by user with pagination"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Build WHERE clause
            conditions = ["tj.user_id = $1"]
            params = [user_id]
            param_idx = 2

            if status:
                conditions.append(f"tj.status = ${param_idx}")
                params.append(status)
                param_idx += 1

            where_clause = " AND ".join(conditions)

            # Get total count
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM finetune_training_jobs tj WHERE {where_clause}",
                *params
            )

            # Get paginated results
            offset = (page - 1) * per_page
            rows = await conn.fetch(
                f"""
                SELECT tj.*,
                       d.name as dataset_name,
                       m.name as model_name
                FROM finetune_training_jobs tj
                LEFT JOIN finetune_datasets d ON tj.dataset_id = d.dataset_id
                LEFT JOIN finetune_models m ON tj.base_model_id = m.model_id
                WHERE {where_clause}
                ORDER BY tj.created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params, per_page, offset
            )

            return [self._row_to_dict(row) for row in rows], count

    async def update_status(
        self,
        job_id: UUID,
        status: str,
        error_message: Optional[str] = None,
        error_details: Optional[dict] = None,
    ) -> Optional[dict]:
        """Update job status"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Handle started_at and completed_at timestamps
            extra_updates = ""
            if status == "training":
                extra_updates = ", started_at = COALESCE(started_at, NOW())"
            elif status in ("completed", "failed", "cancelled"):
                extra_updates = ", completed_at = NOW()"

            row = await conn.fetchrow(
                f"""
                UPDATE finetune_training_jobs
                SET status = $1,
                    error_message = $2,
                    error_details = $3,
                    updated_at = NOW()
                    {extra_updates}
                WHERE job_id = $4
                RETURNING *
                """,
                status, error_message,
                json.dumps(error_details) if error_details else None,
                job_id
            )
            return self._row_to_dict(row) if row else None

    async def update_progress(
        self,
        job_id: UUID,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        progress_percentage: Optional[float] = None,
        current_loss: Optional[float] = None,
        best_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,  # Stored in training_metrics, not separate column
    ) -> Optional[dict]:
        """Update training progress"""
        updates = []
        params = []
        param_idx = 1

        # Note: learning_rate is stored in training_metrics JSONB, not a separate column
        fields = {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percentage": progress_percentage,
            "current_loss": current_loss,
            "best_loss": best_loss,
        }

        for field, value in fields.items():
            if value is not None:
                updates.append(f"{field} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not updates:
            return await self.get_by_id(job_id)

        params.append(job_id)

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE finetune_training_jobs
                SET {', '.join(updates)}, updated_at = NOW()
                WHERE job_id = ${param_idx}
                RETURNING *
                """,
                *params
            )
            return self._row_to_dict(row) if row else None

    async def update_metrics(
        self,
        job_id: UUID,
        metrics: dict,
    ) -> Optional[dict]:
        """Update training metrics (append to existing)"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Get current metrics
            current = await conn.fetchval(
                "SELECT training_metrics FROM finetune_training_jobs WHERE job_id = $1",
                job_id
            )

            current_metrics = json.loads(current) if current else {"history": []}

            # Append new metrics to history
            if "history" not in current_metrics:
                current_metrics["history"] = []
            current_metrics["history"].append({
                **metrics,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Update latest values
            current_metrics["latest"] = metrics

            row = await conn.fetchrow(
                """
                UPDATE finetune_training_jobs
                SET training_metrics = $1, updated_at = NOW()
                WHERE job_id = $2
                RETURNING *
                """,
                json.dumps(current_metrics), job_id
            )
            return self._row_to_dict(row) if row else None

    async def set_output_model(
        self,
        job_id: UUID,
        output_model_id: UUID,
    ) -> Optional[dict]:
        """Set the output model for completed training"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE finetune_training_jobs
                SET output_model_id = $1, updated_at = NOW()
                WHERE job_id = $2
                RETURNING *
                """,
                output_model_id, job_id
            )
            return self._row_to_dict(row) if row else None

    async def delete(self, job_id: UUID) -> bool:
        """Delete a training job"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM finetune_training_jobs WHERE job_id = $1",
                job_id
            )
            return result == "DELETE 1"

    async def get_queued_jobs(self, limit: int = 10) -> list[dict]:
        """Get queued jobs for processing"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tj.*,
                       d.file_path as dataset_path,
                       m.file_path as model_path,
                       m.base_model_id as hf_model_id
                FROM finetune_training_jobs tj
                JOIN finetune_datasets d ON tj.dataset_id = d.dataset_id
                JOIN finetune_models m ON tj.base_model_id = m.model_id
                WHERE tj.status = 'queued' AND tj.execution_env = 'local'
                ORDER BY tj.created_at ASC
                LIMIT $1
                """,
                limit
            )
            return [self._row_to_dict(row) for row in rows]

    async def get_running_jobs(self) -> list[dict]:
        """Get currently running jobs"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM finetune_training_jobs
                WHERE status IN ('preparing', 'training', 'evaluating', 'saving')
                ORDER BY started_at ASC
                """
            )
            return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row) -> dict:
        """Convert database row to dict with JSON parsing"""
        if not row:
            return None

        result = dict(row)

        # Parse JSON fields
        if result.get("config") and isinstance(result["config"], str):
            result["config"] = json.loads(result["config"])
        if result.get("training_metrics") and isinstance(result["training_metrics"], str):
            result["training_metrics"] = json.loads(result["training_metrics"])
        if result.get("error_details") and isinstance(result["error_details"], str):
            result["error_details"] = json.loads(result["error_details"])

        return result
