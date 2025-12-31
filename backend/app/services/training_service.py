"""
Training Service - Business logic for training job management
"""
import asyncio
from typing import Optional, Callable
from uuid import UUID
from pathlib import Path

from ..repositories.training_job_repository import TrainingJobRepository
from ..repositories.dataset_repository import DatasetRepository
from ..repositories.model_repository import ModelRepository
from ..config import settings


# Default training configurations
DEFAULT_LORA_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

DEFAULT_TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "fp16": False,  # Will be set based on hardware
    "bf16": False,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
}

DEFAULT_QLORA_CONFIG = {
    **DEFAULT_LORA_CONFIG,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


class TrainingService:
    """Service for training job operations"""

    def __init__(self):
        self.job_repo = TrainingJobRepository()
        self.dataset_repo = DatasetRepository()
        self.model_repo = ModelRepository()
        self.checkpoints_dir = Path(settings.CHECKPOINTS_DIR)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Progress callbacks for WebSocket
        self._progress_callbacks: dict[str, list[Callable]] = {}

    # ==================== Job Creation ====================

    async def create_training_job(
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
        # Validate dataset
        dataset = await self.dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
        if dataset["user_id"] != user_id:
            raise ValueError("Dataset access denied")
        if dataset["status"] != "ready":
            raise ValueError("Dataset is not ready for training")

        # Validate model
        model = await self.model_repo.get_by_id(base_model_id)
        if not model:
            raise ValueError("Model not found")
        if model["user_id"] != user_id:
            raise ValueError("Model access denied")
        if model["status"] != "ready":
            raise ValueError("Model is not ready for training")

        # Validate training method
        valid_methods = ["sft", "lora", "qlora", "dpo", "orpo"]
        if training_method not in valid_methods:
            raise ValueError(f"Invalid training method. Valid: {valid_methods}")

        # Build config
        final_config = self._build_config(training_method, config)

        # Create job
        job = await self.job_repo.create(
            user_id=user_id,
            dataset_id=dataset_id,
            base_model_id=base_model_id,
            job_name=job_name,
            training_method=training_method,
            execution_env=execution_env,
            config=final_config,
        )

        return job

    def _build_config(self, method: str, custom_config: Optional[dict]) -> dict:
        """Build training config with defaults"""
        config = {
            "training": {**DEFAULT_TRAINING_CONFIG},
        }

        # Add method-specific config
        if method in ("lora", "sft"):
            config["lora"] = {**DEFAULT_LORA_CONFIG}
        elif method == "qlora":
            config["lora"] = {**DEFAULT_QLORA_CONFIG}
        elif method == "dpo":
            config["lora"] = {**DEFAULT_LORA_CONFIG}
            config["dpo"] = {
                "beta": 0.1,
                "loss_type": "sigmoid",
            }
        elif method == "orpo":
            config["lora"] = {**DEFAULT_LORA_CONFIG}
            config["orpo"] = {
                "beta": 0.1,
            }

        # Merge custom config
        if custom_config:
            for key, value in custom_config.items():
                if key in config and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value

        return config

    # ==================== Job Management ====================

    async def get_job(self, job_id: UUID, user_id: UUID) -> dict:
        """Get training job by ID"""
        job = await self.job_repo.get_by_id(job_id)
        if not job:
            raise ValueError("Training job not found")
        if job["user_id"] != user_id:
            raise ValueError("Access denied")
        return job

    async def list_jobs(
        self,
        user_id: UUID,
        page: int = 1,
        per_page: int = 20,
        status: Optional[str] = None,
    ) -> dict:
        """List user's training jobs"""
        jobs, total = await self.job_repo.get_by_user(
            user_id, page, per_page, status
        )
        return {
            "jobs": jobs,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
        }

    async def cancel_job(self, job_id: UUID, user_id: UUID) -> dict:
        """Cancel a training job"""
        job = await self.get_job(job_id, user_id)

        if job["status"] in ("completed", "failed", "cancelled"):
            raise ValueError(f"Cannot cancel job in status: {job['status']}")

        # Update status
        job = await self.job_repo.update_status(job_id, "cancelled")

        # Notify via WebSocket
        await self._notify_progress(str(job_id), {
            "type": "cancelled",
            "job_id": str(job_id),
        })

        return job

    async def delete_job(self, job_id: UUID, user_id: UUID) -> bool:
        """Delete a training job"""
        job = await self.get_job(job_id, user_id)

        # Cannot delete running jobs
        if job["status"] in ("preparing", "training", "evaluating", "saving"):
            raise ValueError("Cannot delete a running job. Cancel it first.")

        # Delete checkpoint files
        job_checkpoint_dir = self.checkpoints_dir / str(job_id)
        if job_checkpoint_dir.exists():
            import shutil
            shutil.rmtree(job_checkpoint_dir)

        return await self.job_repo.delete(job_id)

    # ==================== Progress Callbacks ====================

    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for training progress updates"""
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)

    def unregister_progress_callback(self, job_id: str, callback: Callable):
        """Unregister a progress callback"""
        if job_id in self._progress_callbacks:
            try:
                self._progress_callbacks[job_id].remove(callback)
            except ValueError:
                pass

    async def _notify_progress(self, job_id: str, data: dict):
        """Notify all callbacks about progress"""
        if job_id in self._progress_callbacks:
            for callback in self._progress_callbacks[job_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception:
                    pass

    # ==================== Training Execution ====================

    async def start_training(self, job_id: UUID, user_id: UUID) -> dict:
        """Start training for a job"""
        job = await self.get_job(job_id, user_id)

        if job["status"] != "queued":
            raise ValueError(f"Cannot start job in status: {job['status']}")

        # Update to preparing
        job = await self.job_repo.update_status(job_id, "preparing")

        # Import trainer and start in background
        from ..training.local_trainer import LocalTrainer

        trainer = LocalTrainer(
            job_id=job_id,
            progress_callback=lambda data: asyncio.create_task(
                self._handle_training_progress(job_id, data)
            ),
        )

        # Start training in background task
        asyncio.create_task(self._run_training(trainer, job))

        return job

    async def _run_training(self, trainer, job: dict):
        """Run training in background"""
        job_id = job["job_id"]

        try:
            # Run training
            result = await trainer.train(job)

            if result.get("success"):
                # Create output model record
                output_model = await self.model_repo.create(
                    user_id=job["user_id"],
                    name=f"{job['name']} - {job['training_method'].upper()}",
                    model_type="lora" if job["training_method"] in ("lora", "qlora") else "merged",
                    base_model_id=job.get("hf_model_id", ""),
                    description=f"Trained from {job.get('model_name', 'base model')}",
                    file_path=result.get("output_path"),
                    file_size_mb=result.get("output_size_mb"),
                )

                # Update status to ready
                await self.model_repo.update_status(output_model["model_id"], "ready")

                # Link output model to job
                await self.job_repo.set_output_model(job_id, output_model["model_id"])

                # Update job status
                await self.job_repo.update_status(job_id, "completed")

                await self._notify_progress(str(job_id), {
                    "type": "completed",
                    "job_id": str(job_id),
                    "output_model_id": str(output_model["model_id"]),
                })
            else:
                await self.job_repo.update_status(
                    job_id, "failed",
                    error_message=result.get("error", "Training failed"),
                )

                await self._notify_progress(str(job_id), {
                    "type": "failed",
                    "job_id": str(job_id),
                    "error": result.get("error"),
                })

        except Exception as e:
            await self.job_repo.update_status(
                job_id, "failed",
                error_message=str(e),
            )

            await self._notify_progress(str(job_id), {
                "type": "failed",
                "job_id": str(job_id),
                "error": str(e),
            })

    async def _handle_training_progress(self, job_id: UUID, data: dict):
        """Handle progress updates from trainer"""
        # Update database
        if data.get("type") == "progress":
            await self.job_repo.update_progress(
                job_id,
                current_epoch=data.get("epoch"),
                total_epochs=data.get("total_epochs"),
                current_step=data.get("step"),
                total_steps=data.get("total_steps"),
                progress_percentage=data.get("progress"),
                current_loss=data.get("loss"),
                learning_rate=data.get("learning_rate"),
            )

            # Update best loss if improved
            if data.get("loss"):
                job = await self.job_repo.get_by_id(job_id)
                if not job.get("best_loss") or data["loss"] < job["best_loss"]:
                    await self.job_repo.update_progress(job_id, best_loss=data["loss"])

        elif data.get("type") == "metrics":
            await self.job_repo.update_metrics(job_id, data.get("metrics", {}))

        elif data.get("type") == "status":
            await self.job_repo.update_status(job_id, data.get("status"))

        # Notify WebSocket clients
        await self._notify_progress(str(job_id), data)

    # ==================== Templates ====================

    async def get_training_templates(self) -> list[dict]:
        """Get pre-configured training templates"""
        from ..database import get_db_pool

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT template_id, name, description, training_method, config,
                       recommended_for, base_model_recommendations
                FROM finetune_training_templates
                WHERE is_system = TRUE OR is_public = TRUE
                ORDER BY name
                """
            )
            return [dict(row) for row in rows]

    async def get_template(self, template_id: UUID) -> Optional[dict]:
        """Get a specific training template"""
        from ..database import get_db_pool

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM finetune_training_templates WHERE template_id = $1",
                template_id
            )
            return dict(row) if row else None


# Singleton instance
training_service = TrainingService()
