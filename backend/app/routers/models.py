"""
Model Router - API endpoints for model management
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from ..services.model_service import ModelService
from ..services.auth_service import auth_service
from .auth import get_current_user


router = APIRouter(prefix="/api/models", tags=["Models"])
model_service = ModelService()


# ==================== Request/Response Models ====================

class ModelCreate(BaseModel):
    name: str
    model_type: str = "base"  # base, lora, merged
    base_model_id: str
    description: Optional[str] = None


class ModelDownload(BaseModel):
    hf_model_id: str
    name: str
    description: Optional[str] = None


class HFPush(BaseModel):
    repo_name: str
    private: bool = False


class OllamaImport(BaseModel):
    ollama_name: str


class TestPrompt(BaseModel):
    prompt: str
    max_tokens: int = 256


# ==================== Search HuggingFace ====================

@router.get("/huggingface/search")
async def search_huggingface(
    query: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50),
    current_user = Depends(get_current_user),
):
    """Search models on HuggingFace Hub"""
    try:
        results = await model_service.search_huggingface(query, limit)
        return {"models": results, "total": len(results)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/huggingface/info/{model_id:path}")
async def get_hf_model_info(
    model_id: str,
    current_user = Depends(get_current_user),
):
    """Get detailed info about a HuggingFace model"""
    try:
        # Get user's HF token if available
        hf_token = await auth_service.get_hf_token(current_user.user_id)
        info = await model_service.get_model_info(model_id, hf_token)
        return info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/huggingface/popular")
async def get_popular_models(
    current_user = Depends(get_current_user),
):
    """Get list of popular/recommended base models"""
    return {"models": model_service.get_popular_models()}


# ==================== Download ====================

@router.post("/huggingface/download")
async def download_model(
    data: ModelDownload,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
):
    """Download a model from HuggingFace (runs in background)"""
    try:
        # Get user's HF token
        hf_token = await auth_service.get_hf_token(current_user.user_id)

        # Start download (this will update status as it progresses)
        model = await model_service.download_model(
            user_id=current_user.user_id,
            hf_model_id=data.hf_model_id,
            name=data.name,
            hf_token=hf_token,
            description=data.description,
        )

        return model

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== CRUD ====================

@router.get("")
async def list_models(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user = Depends(get_current_user),
):
    """List user's models"""
    result = await model_service.list_models(
        user_id=current_user.user_id,
        page=page,
        per_page=per_page,
        model_type=model_type,
        status=status,
    )
    return result


@router.post("")
async def create_model(
    data: ModelCreate,
    current_user = Depends(get_current_user),
):
    """Create a model record (for local models)"""
    try:
        model = await model_service.create_model(
            user_id=current_user.user_id,
            name=data.name,
            model_type=data.model_type,
            base_model_id=data.base_model_id,
            description=data.description,
        )
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_id}")
async def get_model(
    model_id: UUID,
    current_user = Depends(get_current_user),
):
    """Get model details"""
    try:
        model = await model_service.get_model(model_id, current_user.user_id)
        return model
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(
    model_id: UUID,
    current_user = Depends(get_current_user),
):
    """Delete a model"""
    try:
        await model_service.delete_model(model_id, current_user.user_id)
        return {"message": "Model deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== HuggingFace Push ====================

@router.post("/{model_id}/push-hf")
async def push_to_huggingface(
    model_id: UUID,
    data: HFPush,
    current_user = Depends(get_current_user),
):
    """Push model to HuggingFace Hub"""
    try:
        # Get user's HF token
        hf_token = await auth_service.get_hf_token(current_user.user_id)
        if not hf_token:
            raise HTTPException(
                status_code=400,
                detail="HuggingFace token not set. Go to Settings to add your token."
            )

        model = await model_service.push_to_huggingface(
            model_id=model_id,
            user_id=current_user.user_id,
            repo_name=data.repo_name,
            hf_token=hf_token,
            private=data.private,
        )
        return model

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Ollama ====================

@router.post("/{model_id}/import-ollama")
async def import_to_ollama(
    model_id: UUID,
    data: OllamaImport,
    current_user = Depends(get_current_user),
):
    """Import model to Ollama for local inference"""
    try:
        model = await model_service.import_to_ollama(
            model_id=model_id,
            user_id=current_user.user_id,
            ollama_name=data.ollama_name,
        )
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Test ====================

@router.post("/{model_id}/test")
async def test_model(
    model_id: UUID,
    data: TestPrompt,
    current_user = Depends(get_current_user),
):
    """Test model with a prompt"""
    try:
        result = await model_service.test_model(
            model_id=model_id,
            user_id=current_user.user_id,
            prompt=data.prompt,
            max_tokens=data.max_tokens,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Base Models for Training ====================

@router.get("/for-training/base")
async def get_base_models_for_training(
    current_user = Depends(get_current_user),
):
    """Get available base models for training"""
    from ..repositories.model_repository import ModelRepository
    repo = ModelRepository()
    models = await repo.get_base_models(current_user.user_id)
    return {"models": models}


@router.get("/for-training/lora")
async def get_lora_models(
    current_user = Depends(get_current_user),
):
    """Get available LoRA adapters"""
    from ..repositories.model_repository import ModelRepository
    repo = ModelRepository()
    models = await repo.get_lora_models(current_user.user_id)
    return {"models": models}


# ==================== Version Management ====================

class VersionUpdate(BaseModel):
    version_tag: Optional[str] = None
    version_notes: Optional[str] = None
    is_latest: Optional[bool] = None


@router.get("/{model_id}/versions")
async def get_model_versions(
    model_id: UUID,
    current_user = Depends(get_current_user),
):
    """Get all versions of a model (based on parent_model_id)"""
    from ..database import db

    # First get the model to find its parent or use it as parent
    model = await db.fetchrow(
        """
        SELECT model_id, parent_model_id, name
        FROM finetune_models
        WHERE model_id = $1 AND user_id = $2
        """,
        model_id,
        current_user.user_id
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Determine which model_id to use as the parent reference
    parent_id = model["parent_model_id"] or model["model_id"]

    # Get all versions
    versions = await db.fetch(
        """
        SELECT
            m.model_id, m.name, m.version, m.version_tag,
            m.is_latest, m.version_notes, m.status,
            m.created_at, m.file_size_mb,
            j.training_method, j.current_loss as final_loss,
            j.completed_at as trained_at
        FROM finetune_models m
        LEFT JOIN finetune_training_jobs j ON m.training_job_id = j.job_id
        WHERE (m.parent_model_id = $1 OR m.model_id = $1)
          AND m.user_id = $2
        ORDER BY m.version DESC
        """,
        parent_id,
        current_user.user_id
    )

    return {
        "parent_model_id": str(parent_id),
        "parent_name": model["name"],
        "total_versions": len(versions),
        "versions": [
            {
                "model_id": str(v["model_id"]),
                "name": v["name"],
                "version": v["version"],
                "version_tag": v["version_tag"],
                "is_latest": v["is_latest"],
                "version_notes": v["version_notes"],
                "status": v["status"],
                "created_at": v["created_at"].isoformat() if v["created_at"] else None,
                "file_size_mb": v["file_size_mb"],
                "training_method": v["training_method"],
                "final_loss": v["final_loss"],
                "trained_at": v["trained_at"].isoformat() if v["trained_at"] else None,
            }
            for v in versions
        ]
    }


@router.put("/{model_id}/version")
async def update_model_version(
    model_id: UUID,
    data: VersionUpdate,
    current_user = Depends(get_current_user),
):
    """Update version metadata for a model"""
    from ..database import db

    # Verify ownership
    model = await db.fetchrow(
        "SELECT model_id, parent_model_id FROM finetune_models WHERE model_id = $1 AND user_id = $2",
        model_id,
        current_user.user_id
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    updates = []
    params = [model_id]
    param_idx = 2

    if data.version_tag is not None:
        updates.append(f"version_tag = ${param_idx}")
        params.append(data.version_tag)
        param_idx += 1

    if data.version_notes is not None:
        updates.append(f"version_notes = ${param_idx}")
        params.append(data.version_notes)
        param_idx += 1

    if data.is_latest is not None and data.is_latest:
        # If setting as latest, unset others
        parent_id = model["parent_model_id"] or model["model_id"]
        await db.execute(
            "UPDATE finetune_models SET is_latest = FALSE WHERE parent_model_id = $1",
            parent_id
        )
        updates.append("is_latest = TRUE")

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    await db.execute(
        f"UPDATE finetune_models SET {', '.join(updates)} WHERE model_id = $1",
        *params
    )

    return {"message": "Version updated successfully"}


@router.get("/{model_id}/version-compare")
async def compare_model_versions(
    model_id: UUID,
    compare_with: UUID,
    current_user = Depends(get_current_user),
):
    """Compare two model versions"""
    from ..database import db

    models = await db.fetch(
        """
        SELECT
            m.model_id, m.name, m.version, m.version_tag,
            m.created_at, m.file_size_mb,
            j.training_method, j.current_loss, j.best_loss,
            j.total_training_time_seconds,
            j.config
        FROM finetune_models m
        LEFT JOIN finetune_training_jobs j ON m.training_job_id = j.job_id
        WHERE m.model_id IN ($1, $2) AND m.user_id = $3
        """,
        model_id,
        compare_with,
        current_user.user_id
    )

    if len(models) != 2:
        raise HTTPException(status_code=404, detail="One or both models not found")

    def format_model(m):
        return {
            "model_id": str(m["model_id"]),
            "name": m["name"],
            "version": m["version"],
            "version_tag": m["version_tag"],
            "created_at": m["created_at"].isoformat() if m["created_at"] else None,
            "file_size_mb": m["file_size_mb"],
            "training_method": m["training_method"],
            "final_loss": m["current_loss"],
            "best_loss": m["best_loss"],
            "training_time_seconds": m["total_training_time_seconds"],
            "config": m["config"],
        }

    return {
        "model_a": format_model(models[0]),
        "model_b": format_model(models[1]),
    }
