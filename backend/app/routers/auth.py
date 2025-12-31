"""
Authentication API Router
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..utils.rate_limiter import limiter

from ..models.auth import (
    HFTokenRequest,
    HFTokenResponse,
    HFTokenValidationResponse,
    LoginRequest,
    PasswordChangeRequest,
    RefreshTokenRequest,
    TokenResponse,
)
from ..models.user import UserCreate, UserResponse, UserUpdate
from ..services.auth_service import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> UserResponse:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    payload = auth_service.decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await auth_service.get_user_by_id(UUID(payload.sub))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is deactivated",
        )

    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(request: Request, user_data: UserCreate):
    """Register a new user (rate limited: 5/minute)"""
    try:
        user = await auth_service.create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, login_data: LoginRequest):
    """Login and get JWT tokens (rate limited: 10/minute)"""
    user = await auth_service.authenticate(login_data.email, login_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth_service.create_tokens(user.user_id)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    payload = auth_service.decode_token(request.refresh_token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    user = await auth_service.get_user_by_id(UUID(payload.sub))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    return auth_service.create_tokens(user.user_id)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    """Get current user profile"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_me(
    update_data: UserUpdate,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Update current user profile"""
    try:
        user = await auth_service.update_user(
            current_user.user_id,
            username=update_data.username,
            preferences=update_data.preferences,
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: PasswordChangeRequest,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Change password"""
    try:
        success = await auth_service.change_password(
            current_user.user_id,
            request.current_password,
            request.new_password,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to change password",
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/hf-token", response_model=HFTokenResponse)
async def save_hf_token(
    request: HFTokenRequest,
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Save HuggingFace token (with optional validation)"""
    result = await auth_service.save_hf_token(
        current_user.user_id,
        request.hf_token,
        validate=request.validate,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to save token"),
        )

    validation = result.get("validation")
    return HFTokenResponse(
        has_token=True,
        token_prefix=request.hf_token[:8] + "..." if len(request.hf_token) > 8 else None,
        is_valid=validation.get("valid") if validation else None,
        hf_username=validation.get("username") if validation else None,
        hf_name=validation.get("name") if validation else None,
        can_write=validation.get("can_write") if validation else None,
        error=None,
    )


@router.get("/hf-token", response_model=HFTokenResponse)
async def get_hf_token_status(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Get HuggingFace token status (basic info only)"""
    token = await auth_service.get_hf_token(current_user.user_id)

    return HFTokenResponse(
        has_token=token is not None,
        token_prefix=token[:8] + "..." if token and len(token) > 8 else None,
        is_valid=None,
    )


@router.get("/hf-token/validate", response_model=HFTokenValidationResponse)
async def validate_hf_token(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Validate stored HuggingFace token with HuggingFace API"""
    result = await auth_service.validate_hf_token(current_user.user_id)

    return HFTokenValidationResponse(
        valid=result.get("valid", False),
        username=result.get("username"),
        name=result.get("name"),
        email=result.get("email"),
        orgs=result.get("orgs", []),
        can_write=result.get("can_write", False),
        error=result.get("error"),
    )


@router.delete("/hf-token", status_code=status.HTTP_204_NO_CONTENT)
async def delete_hf_token(
    current_user: Annotated[UserResponse, Depends(get_current_user)],
):
    """Delete HuggingFace token"""
    await auth_service.delete_hf_token(current_user.user_id)
