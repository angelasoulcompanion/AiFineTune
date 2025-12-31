"""
Authentication Pydantic models
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class TokenPayload(BaseModel):
    """JWT token payload"""
    sub: str  # user_id
    exp: int  # expiration timestamp
    type: str  # 'access' or 'refresh'


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class HFTokenRequest(BaseModel):
    """HuggingFace token save request"""
    hf_token: str = Field(..., min_length=10)
    validate_token: bool = True  # Whether to validate with HuggingFace API


class HFTokenResponse(BaseModel):
    """HuggingFace token status response"""
    has_token: bool
    token_prefix: Optional[str] = None  # First few chars for verification
    is_valid: Optional[bool] = None
    hf_username: Optional[str] = None
    hf_name: Optional[str] = None
    can_write: Optional[bool] = None
    error: Optional[str] = None


class HFTokenValidationResponse(BaseModel):
    """HuggingFace token validation response"""
    valid: bool
    username: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    orgs: list[str] = []
    can_write: bool = False
    error: Optional[str] = None
