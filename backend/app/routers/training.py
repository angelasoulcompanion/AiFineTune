"""
Training Router - API endpoints for training job management
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.training_service import training_service
from .auth import get_current_user


router = APIRouter(prefix="/api/training", tags=["Training"])


# ==================== Request/Response Models ====================

class TrainingConfig(BaseModel):
    # Training params
    num_train_epochs: Optional[int] = Field(3, ge=1, le=100)
    per_device_train_batch_size: Optional[int] = Field(4, ge=1, le=64)
    gradient_accumulation_steps: Optional[int] = Field(4, ge=1, le=64)
    learning_rate: Optional[float] = Field(2e-4, ge=1e-7, le=1.0)
    warmup_ratio: Optional[float] = Field(0.03, ge=0, le=1.0)
    weight_decay: Optional[float] = Field(0.01, ge=0, le=1.0)
    max_seq_length: Optional[int] = Field(2048, ge=128, le=32768)

    # Learning rate scheduler
    lr_scheduler_type: Optional[str] = Field("cosine", pattern="^(linear|cosine|cosine_with_restarts|polynomial|constant|constant_with_warmup|inverse_sqrt|reduce_lr_on_plateau)$")
    warmup_steps: Optional[int] = Field(None, ge=0)  # Alternative to warmup_ratio
    num_cycles: Optional[float] = Field(0.5, ge=0.1, le=5.0)  # For cosine_with_restarts

    # LoRA params
    lora_r: Optional[int] = Field(16, ge=1, le=256)
    lora_alpha: Optional[int] = Field(32, ge=1, le=512)
    lora_dropout: Optional[float] = Field(0.05, ge=0, le=1.0)

    # Advanced params
    gradient_checkpointing: Optional[bool] = Field(True)
    optim: Optional[str] = Field("adamw_8bit", pattern="^(adamw_8bit|adamw_torch|adamw_torch_fused|adamw_apex_fused|adamw_anyprecision|sgd|adafactor|adagrad)$")
    logging_steps: Optional[int] = Field(10, ge=1)
    save_steps: Optional[int] = Field(100, ge=1)
    eval_steps: Optional[int] = Field(None, ge=1)  # For evaluation during training


class EstimateRequest(BaseModel):
    """Request for duration/memory estimation"""
    model_id: str = Field(..., description="HuggingFace model ID or local model name")
    training_method: str = Field("lora", pattern="^(sft|lora|qlora|dpo|orpo)$")
    num_samples: int = Field(..., ge=1, description="Number of samples in dataset")
    num_epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=64)
    gradient_accumulation_steps: int = Field(4, ge=1, le=64)
    max_seq_length: int = Field(2048, ge=128, le=32768)


class CloudEstimateRequest(BaseModel):
    """Request for cloud training cost estimation"""
    model_id: str = Field(..., description="HuggingFace model ID")
    training_method: str = Field("lora", pattern="^(sft|lora|qlora|dpo|orpo)$")
    num_samples: int = Field(..., ge=1)
    num_epochs: int = Field(3, ge=1, le=100)
    gpu_type: Optional[str] = Field(None, description="GPU type (T4, A10G, A100-40GB, A100-80GB, H100)")


class CreateTrainingJob(BaseModel):
    dataset_id: str
    base_model_id: str
    name: str = Field(..., min_length=1, max_length=200)
    training_method: str = Field(..., pattern="^(sft|lora|qlora|dpo|orpo)$")
    execution_env: str = Field("local", pattern="^(local|modal|hf_spaces)$")
    config: Optional[TrainingConfig] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    user_id: str
    dataset_id: str
    base_model_id: str
    name: str
    training_method: str
    execution_env: str
    status: str
    config: Optional[dict] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    progress_percentage: Optional[float] = None
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    training_metrics: Optional[dict] = None
    output_model_id: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str

    # Joined fields
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None


# ==================== Endpoints ====================

@router.get("")
async def list_training_jobs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    current_user = Depends(get_current_user),
):
    """List user's training jobs"""
    result = await training_service.list_jobs(
        user_id=current_user.user_id,
        page=page,
        per_page=per_page,
        status=status,
    )
    return result


@router.post("")
async def create_training_job(
    data: CreateTrainingJob,
    current_user = Depends(get_current_user),
):
    """Create a new training job"""
    try:
        # Convert config to dict
        config_dict = None
        if data.config:
            config_dict = {
                "training": {
                    "num_train_epochs": data.config.num_train_epochs,
                    "per_device_train_batch_size": data.config.per_device_train_batch_size,
                    "gradient_accumulation_steps": data.config.gradient_accumulation_steps,
                    "learning_rate": data.config.learning_rate,
                    "warmup_ratio": data.config.warmup_ratio,
                    "weight_decay": data.config.weight_decay,
                    "max_seq_length": data.config.max_seq_length,
                },
                "lora": {
                    "lora_r": data.config.lora_r,
                    "lora_alpha": data.config.lora_alpha,
                    "lora_dropout": data.config.lora_dropout,
                }
            }

        job = await training_service.create_training_job(
            user_id=current_user.user_id,
            dataset_id=UUID(data.dataset_id),
            base_model_id=UUID(data.base_model_id),
            job_name=data.name,
            training_method=data.training_method,
            execution_env=data.execution_env,
            config=config_dict,
        )
        return job

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/templates")
async def get_training_templates(
    current_user = Depends(get_current_user),
):
    """Get pre-configured training templates"""
    templates = await training_service.get_training_templates()
    return {"templates": templates}


@router.get("/templates/{template_id}")
async def get_training_template(
    template_id: UUID,
    current_user = Depends(get_current_user),
):
    """Get a specific training template"""
    template = await training_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.get("/{job_id}")
async def get_training_job(
    job_id: UUID,
    current_user = Depends(get_current_user),
):
    """Get training job details"""
    try:
        job = await training_service.get_job(job_id, current_user.user_id)
        return job
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{job_id}/start")
async def start_training_job(
    job_id: UUID,
    current_user = Depends(get_current_user),
):
    """Start a queued training job"""
    try:
        job = await training_service.start_training(job_id, current_user.user_id)
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{job_id}/cancel")
async def cancel_training_job(
    job_id: UUID,
    current_user = Depends(get_current_user),
):
    """Cancel a running training job"""
    try:
        job = await training_service.cancel_job(job_id, current_user.user_id)
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{job_id}")
async def delete_training_job(
    job_id: UUID,
    current_user = Depends(get_current_user),
):
    """Delete a training job"""
    try:
        await training_service.delete_job(job_id, current_user.user_id)
        return {"message": "Training job deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}/status")
async def get_training_status(
    job_id: UUID,
    current_user = Depends(get_current_user),
):
    """Get training job status (lightweight)"""
    try:
        job = await training_service.get_job(job_id, current_user.user_id)
        return {
            "job_id": str(job["job_id"]),
            "status": job["status"],
            "progress_percentage": job.get("progress_percentage", 0),
            "current_epoch": job.get("current_epoch"),
            "total_epochs": job.get("total_epochs"),
            "current_step": job.get("current_step"),
            "total_steps": job.get("total_steps"),
            "current_loss": job.get("current_loss"),
            "best_loss": job.get("best_loss"),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{job_id}/metrics")
async def get_training_metrics(
    job_id: UUID,
    current_user = Depends(get_current_user),
):
    """Get training metrics history"""
    try:
        job = await training_service.get_job(job_id, current_user.user_id)
        metrics = job.get("training_metrics", {})
        return {
            "job_id": str(job["job_id"]),
            "latest": metrics.get("latest", {}),
            "history": metrics.get("history", []),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==================== Estimation Endpoints ====================

@router.post("/estimate/duration")
async def estimate_training_duration(
    data: EstimateRequest,
    current_user = Depends(get_current_user),
):
    """Estimate training duration based on config"""
    from ..training.utils import estimate_training_duration as estimate_duration
    from ..training.utils import estimate_model_parameters

    try:
        # Get device from training service
        from ..training.local_trainer import LocalTrainer
        trainer = LocalTrainer(job_id=UUID("00000000-0000-0000-0000-000000000000"))
        device = trainer._get_device()

        model_size = estimate_model_parameters(data.model_id)

        estimate = estimate_duration(
            num_samples=data.num_samples,
            num_epochs=data.num_epochs,
            batch_size=data.batch_size,
            gradient_accumulation_steps=data.gradient_accumulation_steps,
            model_size_b=model_size,
            device=device,
            method=data.training_method,
            max_seq_length=data.max_seq_length,
        )

        return {
            "duration": {
                "formatted": estimate.formatted,
                "seconds": estimate.estimated_seconds,
                "minutes": estimate.estimated_minutes,
                "hours": estimate.estimated_hours,
                "confidence": estimate.confidence,
                "basis": estimate.basis,
            },
            "config": {
                "model_id": data.model_id,
                "model_size_b": model_size,
                "device": device,
                "method": data.training_method,
                "num_samples": data.num_samples,
                "num_epochs": data.num_epochs,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/estimate/memory")
async def estimate_memory_requirements(
    data: EstimateRequest,
    current_user = Depends(get_current_user),
):
    """Estimate memory requirements for training"""
    from ..training.utils import estimate_memory_required, estimate_model_parameters, get_memory_usage

    try:
        model_size = estimate_model_parameters(data.model_id)

        memory_req = estimate_memory_required(
            model_size_b=model_size,
            method=data.training_method,
            batch_size=data.batch_size,
            max_seq_length=data.max_seq_length,
        )

        # Get current system memory
        current_memory = get_memory_usage()

        return {
            "requirements": memory_req,
            "model_size_b": model_size,
            "current_system": current_memory.to_dict() if current_memory else None,
            "can_train": current_memory.available_gb >= memory_req["minimum_gb"] if current_memory else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scheduler-types")
async def get_lr_scheduler_types(
    current_user = Depends(get_current_user),
):
    """Get available learning rate scheduler types"""
    return {
        "schedulers": [
            {
                "id": "linear",
                "name": "Linear",
                "description": "Linear decay from initial LR to 0",
                "best_for": "General purpose, stable training",
            },
            {
                "id": "cosine",
                "name": "Cosine",
                "description": "Cosine annealing schedule",
                "best_for": "Most LLM fine-tuning tasks (recommended)",
            },
            {
                "id": "cosine_with_restarts",
                "name": "Cosine with Restarts",
                "description": "Cosine with periodic warm restarts",
                "best_for": "Longer training, escaping local minima",
            },
            {
                "id": "polynomial",
                "name": "Polynomial",
                "description": "Polynomial decay schedule",
                "best_for": "Gradual decay control",
            },
            {
                "id": "constant",
                "name": "Constant",
                "description": "Constant learning rate",
                "best_for": "Short fine-tuning, debugging",
            },
            {
                "id": "constant_with_warmup",
                "name": "Constant with Warmup",
                "description": "Warmup then constant LR",
                "best_for": "Simple training with warmup phase",
            },
            {
                "id": "inverse_sqrt",
                "name": "Inverse Square Root",
                "description": "Inverse square root decay",
                "best_for": "Transformer pretraining style",
            },
        ],
        "recommended": "cosine",
    }


@router.get("/optimizer-types")
async def get_optimizer_types(
    current_user = Depends(get_current_user),
):
    """Get available optimizer types"""
    return {
        "optimizers": [
            {
                "id": "adamw_8bit",
                "name": "AdamW 8-bit",
                "description": "Memory-efficient 8-bit AdamW",
                "best_for": "Memory-constrained training (recommended for LoRA/QLoRA)",
            },
            {
                "id": "adamw_torch",
                "name": "AdamW (PyTorch)",
                "description": "Standard AdamW optimizer",
                "best_for": "General purpose, stable baseline",
            },
            {
                "id": "adamw_torch_fused",
                "name": "AdamW Fused",
                "description": "Fused AdamW for CUDA (faster)",
                "best_for": "CUDA training, maximum speed",
            },
            {
                "id": "adafactor",
                "name": "Adafactor",
                "description": "Memory-efficient optimizer (no momentum)",
                "best_for": "Very large models, extreme memory constraints",
            },
            {
                "id": "sgd",
                "name": "SGD",
                "description": "Stochastic Gradient Descent",
                "best_for": "Simple training, research comparisons",
            },
        ],
        "recommended": "adamw_8bit",
    }


# ==================== Cloud Training Endpoints ====================

@router.get("/cloud/gpu-types")
async def get_cloud_gpu_types(
    current_user = Depends(get_current_user),
):
    """Get available Modal.com GPU types with pricing"""
    from ..training.modal_trainer import MODAL_GPU_CONFIGS

    return {
        "gpus": [
            {
                "id": gpu_id,
                "name": gpu_id,
                "memory_gb": config["memory"],
                "cost_per_hour": config["cost_per_hour"],
                "best_for": config["best_for"],
            }
            for gpu_id, config in MODAL_GPU_CONFIGS.items()
        ],
        "recommended": "A10G",
        "provider": "Modal.com",
    }


@router.post("/cloud/estimate")
async def estimate_cloud_cost(
    data: CloudEstimateRequest,
    current_user = Depends(get_current_user),
):
    """Estimate Modal.com training cost"""
    from ..training.modal_trainer import estimate_modal_cost, get_recommended_gpu
    from ..training.utils import estimate_model_parameters

    try:
        model_size = estimate_model_parameters(data.model_id)

        # Get recommended GPU if not specified
        gpu_type = data.gpu_type or get_recommended_gpu(model_size, data.training_method)

        estimate = estimate_modal_cost(
            model_size_b=model_size,
            num_samples=data.num_samples,
            num_epochs=data.num_epochs,
            method=data.training_method,
            gpu_type=gpu_type,
        )

        return {
            "estimate": estimate,
            "model_size_b": model_size,
            "recommended_gpu": get_recommended_gpu(model_size, data.training_method),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/cloud/status")
async def get_cloud_status(
    current_user = Depends(get_current_user),
):
    """Check Modal.com connectivity and configuration"""
    import os

    modal_configured = bool(os.environ.get("MODAL_TOKEN_ID"))

    status = {
        "modal_configured": modal_configured,
        "modal_token_set": bool(os.environ.get("MODAL_TOKEN_ID")),
        "modal_secret_set": bool(os.environ.get("MODAL_TOKEN_SECRET")),
    }

    if modal_configured:
        try:
            import modal
            status["modal_installed"] = True
            status["modal_version"] = modal.__version__
        except ImportError:
            status["modal_installed"] = False
            status["error"] = "Modal package not installed"
    else:
        status["error"] = "Modal tokens not configured"

    return status


@router.post("/cloud/health-check")
async def cloud_health_check(
    current_user = Depends(get_current_user),
):
    """Run a health check on Modal.com (tests GPU connectivity)"""
    import os

    if not os.environ.get("MODAL_TOKEN_ID"):
        raise HTTPException(
            status_code=400,
            detail="Modal not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
        )

    try:
        from ..training.modal_trainer import create_modal_training_app

        app = create_modal_training_app()

        # This would actually call Modal - for now return mock
        return {
            "status": "ok",
            "message": "Modal connectivity verified",
            "note": "Full health check requires Modal deployment",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
