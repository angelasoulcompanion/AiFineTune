"""
Configuration management for AiFineTune Platform
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "AiFineTune Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # API
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Database
    DATABASE_URL: str = "postgresql://davidsamanyaporn@localhost:5432/AiFineTune"
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10
    DB_COMMAND_TIMEOUT: int = 60

    # JWT Authentication
    JWT_SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # File Storage
    UPLOAD_DIR: str = "./uploads"
    DATASETS_DIR: str = "./uploads/datasets"
    MODELS_DIR: str = "./uploads/models"
    CHECKPOINTS_DIR: str = "./uploads/checkpoints"
    MAX_UPLOAD_SIZE_MB: int = 500

    # HuggingFace
    HF_CACHE_DIR: Optional[str] = None
    HF_TOKEN: Optional[str] = None

    # Training
    DEFAULT_DEVICE: str = "auto"  # 'auto', 'cuda', 'mps', 'cpu'
    MAX_CONCURRENT_JOBS: int = 2

    # Modal.com (Cloud Training)
    MODAL_TOKEN_ID: Optional[str] = None
    MODAL_TOKEN_SECRET: Optional[str] = None

    # Encryption (for storing HF tokens)
    ENCRYPTION_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience access
settings = get_settings()
