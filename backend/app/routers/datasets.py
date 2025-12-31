"""
Dataset API Router
"""

from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from ..models.dataset import (
    DatasetList,
    DatasetPreview,
    DatasetResponse,
    DatasetType,
    DatasetUpdate,
)
from ..models.user import UserResponse
from ..routers.auth import get_current_user
from ..services.dataset_service import dataset_service

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=DatasetList)
async def list_datasets(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    dataset_type: Optional[str] = Query(None),
):
    """List all datasets for current user"""
    return await dataset_service.list_datasets(
        user_id=current_user.user_id,
        page=page,
        per_page=per_page,
        status=status,
        dataset_type=dataset_type,
    )


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    file: UploadFile = File(...),
    name: str = Form(...),
    dataset_type: DatasetType = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated
):
    """Upload a new dataset file"""
    # Parse tags
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        dataset = await dataset_service.upload_dataset(
            user_id=current_user.user_id,
            file=file,
            name=name,
            description=description,
            dataset_type=dataset_type,
            tags=tag_list,
        )
        return dataset
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Get dataset details"""
    dataset = await dataset_service.get_dataset(dataset_id, current_user.user_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )
    return dataset


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: UUID,
    update_data: DatasetUpdate,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Update dataset metadata"""
    try:
        dataset = await dataset_service.update_dataset(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            name=update_data.name,
            description=update_data.description,
            tags=update_data.tags,
            column_mapping=update_data.column_mapping,
        )
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )
        return dataset
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Delete a dataset"""
    success = await dataset_service.delete_dataset(dataset_id, current_user.user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found",
        )


@router.post("/{dataset_id}/validate", response_model=DatasetResponse)
async def validate_dataset(
    dataset_id: UUID,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Validate a dataset"""
    try:
        dataset = await dataset_service.validate_dataset(dataset_id, current_user.user_id)
        return dataset
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: UUID,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
    limit: int = Query(10, ge=1, le=100),
):
    """Preview dataset rows"""
    try:
        preview = await dataset_service.preview_dataset(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
            limit=limit,
        )
        return preview
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(
    dataset_id: UUID,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Get detailed dataset statistics"""
    try:
        stats = await dataset_service.get_dataset_statistics(
            dataset_id=dataset_id,
            user_id=current_user.user_id,
        )
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
