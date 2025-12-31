"""
Model Repository - Database operations for finetune_models
"""
from typing import Optional
from uuid import UUID
from datetime import datetime

from ..database import get_db_pool


class ModelRepository:
    """Repository for model database operations"""

    async def create(
        self,
        user_id: UUID,
        name: str,
        model_type: str,
        base_model_id: str,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        file_size_mb: Optional[float] = None,
        hf_repo_id: Optional[str] = None,
    ) -> dict:
        """Create a new model record"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO finetune_models (
                    user_id, name, model_type, base_model_id, description,
                    file_path, file_size_mb, hf_repo_id, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'available')
                RETURNING *
                """,
                user_id, name, model_type, base_model_id, description,
                file_path, file_size_mb, hf_repo_id
            )
            return dict(row)

    async def get_by_id(self, model_id: UUID) -> Optional[dict]:
        """Get model by ID"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM finetune_models WHERE model_id = $1",
                model_id
            )
            return dict(row) if row else None

    async def get_by_user(
        self,
        user_id: UUID,
        page: int = 1,
        per_page: int = 20,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[dict], int]:
        """Get models by user with pagination"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Build WHERE clause
            conditions = ["user_id = $1"]
            params = [user_id]
            param_idx = 2

            if model_type:
                conditions.append(f"model_type = ${param_idx}")
                params.append(model_type)
                param_idx += 1

            if status:
                conditions.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1

            where_clause = " AND ".join(conditions)

            # Get total count
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM finetune_models WHERE {where_clause}",
                *params
            )

            # Get paginated results
            offset = (page - 1) * per_page
            rows = await conn.fetch(
                f"""
                SELECT * FROM finetune_models
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params, per_page, offset
            )

            return [dict(row) for row in rows], count

    async def update(self, model_id: UUID, **kwargs) -> Optional[dict]:
        """Update model fields"""
        if not kwargs:
            return await self.get_by_id(model_id)

        # Build SET clause dynamically
        set_parts = []
        params = []
        for idx, (key, value) in enumerate(kwargs.items(), start=1):
            set_parts.append(f"{key} = ${idx}")
            params.append(value)

        params.append(model_id)
        set_clause = ", ".join(set_parts)

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE finetune_models
                SET {set_clause}, updated_at = NOW()
                WHERE model_id = ${len(params)}
                RETURNING *
                """,
                *params
            )
            return dict(row) if row else None

    async def update_status(
        self,
        model_id: UUID,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[dict]:
        """Update model status"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE finetune_models
                SET status = $1, updated_at = NOW()
                WHERE model_id = $2
                RETURNING *
                """,
                status, model_id
            )
            return dict(row) if row else None

    async def update_hf_push(
        self,
        model_id: UUID,
        hf_repo_id: str,
        is_pushed: bool = True
    ) -> Optional[dict]:
        """Update HuggingFace push status"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE finetune_models
                SET hf_repo_id = $1, is_pushed_to_hf = $2, updated_at = NOW()
                WHERE model_id = $3
                RETURNING *
                """,
                hf_repo_id, is_pushed, model_id
            )
            return dict(row) if row else None

    async def update_ollama(
        self,
        model_id: UUID,
        ollama_name: str,
        is_imported: bool = True
    ) -> Optional[dict]:
        """Update Ollama import status"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE finetune_models
                SET ollama_model_name = $1, is_imported_to_ollama = $2, updated_at = NOW()
                WHERE model_id = $3
                RETURNING *
                """,
                ollama_name, is_imported, model_id
            )
            return dict(row) if row else None

    async def delete(self, model_id: UUID) -> bool:
        """Delete a model"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM finetune_models WHERE model_id = $1",
                model_id
            )
            return result == "DELETE 1"

    async def get_base_models(self, user_id: UUID) -> list[dict]:
        """Get all base models for user (for training)"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model_id, name, base_model_id, model_type, status
                FROM finetune_models
                WHERE user_id = $1 AND model_type = 'base' AND status = 'ready'
                ORDER BY name
                """,
                user_id
            )
            return [dict(row) for row in rows]

    async def get_lora_models(self, user_id: UUID) -> list[dict]:
        """Get all LoRA adapters for user"""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT model_id, name, base_model_id, model_type, file_path, status
                FROM finetune_models
                WHERE user_id = $1 AND model_type = 'lora' AND status = 'ready'
                ORDER BY created_at DESC
                """,
                user_id
            )
            return [dict(row) for row in rows]
