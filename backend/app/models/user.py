"""
User Pydantic models
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)


class UserCreate(UserBase):
    """Model for user registration"""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """Model for updating user profile"""
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    preferences: Optional[dict[str, Any]] = None


class UserInDB(UserBase):
    """User model as stored in database"""
    user_id: UUID
    password_hash: str
    is_active: bool = True
    is_admin: bool = False
    hf_token_encrypted: Optional[str] = None
    preferences: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    """User response model (no sensitive data)"""
    user_id: UUID
    is_active: bool
    is_admin: bool
    has_hf_token: bool = False
    preferences: dict[str, Any] = {}
    created_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserList(BaseModel):
    """Paginated user list response"""
    users: list[UserResponse]
    total: int
    page: int
    per_page: int
