"""
Authentication service - JWT tokens and password hashing
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import bcrypt
import jwt
from pydantic import EmailStr

from ..config import settings


def parse_jsonb(value) -> dict:
    """Parse JSONB value from database (may be string or dict)"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


from ..database import db
from ..models.auth import TokenPayload, TokenResponse
from ..models.user import UserCreate, UserInDB, UserResponse


class AuthService:
    """Authentication and user management service"""

    # Password hashing
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against a hash"""
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    # JWT tokens
    @staticmethod
    def create_access_token(user_id: UUID) -> str:
        """Create a JWT access token"""
        expires = datetime.now(timezone.utc) + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )
        payload = {
            "sub": str(user_id),
            "exp": int(expires.timestamp()),
            "type": "access",
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def create_refresh_token(user_id: UUID) -> str:
        """Create a JWT refresh token"""
        expires = datetime.now(timezone.utc) + timedelta(
            days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        payload = {
            "sub": str(user_id),
            "exp": int(expires.timestamp()),
            "type": "refresh",
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Optional[TokenPayload]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def create_tokens(self, user_id: UUID) -> TokenResponse:
        """Create both access and refresh tokens"""
        access_token = self.create_access_token(user_id)
        refresh_token = self.create_refresh_token(user_id)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    # User operations
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        # Check if email exists
        existing = await db.fetchrow(
            "SELECT user_id FROM finetune_users WHERE email = $1",
            user_data.email,
        )
        if existing:
            raise ValueError("Email already registered")

        # Check if username exists
        existing = await db.fetchrow(
            "SELECT user_id FROM finetune_users WHERE username = $1",
            user_data.username,
        )
        if existing:
            raise ValueError("Username already taken")

        # Hash password and create user
        password_hash = self.hash_password(user_data.password)

        row = await db.fetchrow(
            """
            INSERT INTO finetune_users (email, username, password_hash)
            VALUES ($1, $2, $3)
            RETURNING user_id, email, username, is_active, is_admin,
                      hf_token_encrypted IS NOT NULL as has_hf_token,
                      preferences, created_at, last_login_at
            """,
            user_data.email,
            user_data.username,
            password_hash,
        )

        return UserResponse(
            user_id=row["user_id"],
            email=row["email"],
            username=row["username"],
            is_active=row["is_active"],
            is_admin=row["is_admin"],
            has_hf_token=row["has_hf_token"],
            preferences=parse_jsonb(row["preferences"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )

    async def authenticate(self, email: EmailStr, password: str) -> Optional[UserResponse]:
        """Authenticate user by email and password"""
        row = await db.fetchrow(
            """
            SELECT user_id, email, username, password_hash, is_active, is_admin,
                   hf_token_encrypted IS NOT NULL as has_hf_token,
                   preferences, created_at, last_login_at
            FROM finetune_users
            WHERE email = $1 AND is_active = TRUE
            """,
            email,
        )

        if not row:
            return None

        if not self.verify_password(password, row["password_hash"]):
            return None

        # Update last login
        await db.execute(
            "UPDATE finetune_users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = $1",
            row["user_id"],
        )

        return UserResponse(
            user_id=row["user_id"],
            email=row["email"],
            username=row["username"],
            is_active=row["is_active"],
            is_admin=row["is_admin"],
            has_hf_token=row["has_hf_token"],
            preferences=parse_jsonb(row["preferences"]),
            created_at=row["created_at"],
            last_login_at=datetime.now(timezone.utc),
        )

    async def get_user_by_id(self, user_id: UUID) -> Optional[UserResponse]:
        """Get user by ID"""
        row = await db.fetchrow(
            """
            SELECT user_id, email, username, is_active, is_admin,
                   hf_token_encrypted IS NOT NULL as has_hf_token,
                   preferences, created_at, last_login_at
            FROM finetune_users
            WHERE user_id = $1
            """,
            user_id,
        )

        if not row:
            return None

        return UserResponse(
            user_id=row["user_id"],
            email=row["email"],
            username=row["username"],
            is_active=row["is_active"],
            is_admin=row["is_admin"],
            has_hf_token=row["has_hf_token"],
            preferences=parse_jsonb(row["preferences"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )

    async def update_user(
        self, user_id: UUID, username: Optional[str] = None, preferences: Optional[dict] = None
    ) -> Optional[UserResponse]:
        """Update user profile"""
        updates = []
        params = [user_id]
        param_idx = 2

        if username is not None:
            # Check if username is taken by another user
            existing = await db.fetchrow(
                "SELECT user_id FROM finetune_users WHERE username = $1 AND user_id != $2",
                username,
                user_id,
            )
            if existing:
                raise ValueError("Username already taken")
            updates.append(f"username = ${param_idx}")
            params.append(username)
            param_idx += 1

        if preferences is not None:
            updates.append(f"preferences = ${param_idx}")
            params.append(preferences)
            param_idx += 1

        if not updates:
            return await self.get_user_by_id(user_id)

        query = f"""
            UPDATE finetune_users
            SET {', '.join(updates)}
            WHERE user_id = $1
            RETURNING user_id, email, username, is_active, is_admin,
                      hf_token_encrypted IS NOT NULL as has_hf_token,
                      preferences, created_at, last_login_at
        """

        row = await db.fetchrow(query, *params)

        if not row:
            return None

        return UserResponse(
            user_id=row["user_id"],
            email=row["email"],
            username=row["username"],
            is_active=row["is_active"],
            is_admin=row["is_admin"],
            has_hf_token=row["has_hf_token"],
            preferences=parse_jsonb(row["preferences"]),
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )

    async def save_hf_token(self, user_id: UUID, hf_token: str, validate: bool = True) -> dict:
        """
        Save HuggingFace token (encrypted)

        Args:
            user_id: User ID
            hf_token: HuggingFace API token
            validate: Whether to validate the token with HuggingFace API

        Returns:
            dict with success status and validation info
        """
        from ..utils.crypto import encrypt_token
        from ..utils.hf_validator import validate_hf_token

        # Validate token if requested
        validation_result = None
        if validate:
            validation_result = await validate_hf_token(hf_token)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "validation": validation_result,
                }

        # Encrypt the token before storing
        encrypted = encrypt_token(hf_token)

        result = await db.execute(
            "UPDATE finetune_users SET hf_token_encrypted = $2 WHERE user_id = $1",
            user_id,
            encrypted,
        )

        success = "UPDATE 1" in result
        return {
            "success": success,
            "error": None if success else "Failed to save token",
            "validation": validation_result,
        }

    async def get_hf_token(self, user_id: UUID) -> Optional[str]:
        """Get HuggingFace token (decrypted)"""
        from ..utils.crypto import decrypt_token, is_encrypted

        encrypted_token = await db.fetchval(
            "SELECT hf_token_encrypted FROM finetune_users WHERE user_id = $1",
            user_id,
        )

        if not encrypted_token:
            return None

        # Check if token is encrypted (for backward compatibility)
        if is_encrypted(encrypted_token):
            return decrypt_token(encrypted_token)
        else:
            # Token was stored before encryption was implemented
            # Return as-is (consider re-encrypting on next save)
            return encrypted_token

    async def validate_hf_token(self, user_id: UUID) -> dict:
        """Validate the stored HuggingFace token"""
        from ..utils.hf_validator import validate_hf_token

        token = await self.get_hf_token(user_id)
        if not token:
            return {
                "valid": False,
                "error": "No HuggingFace token stored",
                "username": None,
            }

        return await validate_hf_token(token)

    async def delete_hf_token(self, user_id: UUID) -> bool:
        """Delete the stored HuggingFace token"""
        result = await db.execute(
            "UPDATE finetune_users SET hf_token_encrypted = NULL WHERE user_id = $1",
            user_id,
        )
        return "UPDATE 1" in result

    async def change_password(
        self, user_id: UUID, current_password: str, new_password: str
    ) -> bool:
        """Change user password"""
        row = await db.fetchrow(
            "SELECT password_hash FROM finetune_users WHERE user_id = $1",
            user_id,
        )

        if not row:
            return False

        if not self.verify_password(current_password, row["password_hash"]):
            raise ValueError("Current password is incorrect")

        new_hash = self.hash_password(new_password)
        result = await db.execute(
            "UPDATE finetune_users SET password_hash = $2 WHERE user_id = $1",
            user_id,
            new_hash,
        )
        return "UPDATE 1" in result


# Global service instance
auth_service = AuthService()
