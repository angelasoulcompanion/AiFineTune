"""
AiFineTune Platform - FastAPI Application
Fine-tuning platform for LLM models
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import db, init_db
from app.routers import auth, datasets, models, training, dashboard
from app.utils.rate_limiter import limiter

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AiFineTune Platform...")

    # Create upload directories
    for dir_path in [settings.UPLOAD_DIR, settings.DATASETS_DIR, settings.MODELS_DIR, settings.CHECKPOINTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Connect to database
    await db.connect()
    logger.info("Database connected")

    # Initialize database (run migrations)
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database init warning: {e}")

    yield

    # Shutdown
    logger.info("Shutting down AiFineTune Platform...")
    await db.disconnect()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Fine-tuning platform for LLM models. Supports SFT, LoRA, QLoRA, DPO, ORPO.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(datasets.router, prefix=settings.API_PREFIX)
app.include_router(models.router)  # models router has /api/models prefix
app.include_router(training.router)  # training router has /api/training prefix
app.include_router(dashboard.router)  # dashboard router has /api/dashboard prefix


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "running",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": db.pool is not None,
    }


@app.get("/api/methods")
async def get_fine_tuning_methods():
    """Get all supported fine-tuning methods"""
    return {
        "methods": [
            {
                "id": "sft",
                "name": "SFT (Supervised Fine-Tuning)",
                "description": "Train model on instruction-output pairs",
                "use_cases": ["Chatbot", "Code generation", "Task-specific"],
                "complexity": "Easy",
                "memory": "High",
                "data_format": "instruction-input-output",
            },
            {
                "id": "lora",
                "name": "LoRA (Low-Rank Adaptation)",
                "description": "Efficient fine-tuning with low-rank matrices",
                "use_cases": ["Resource-constrained", "Quick adaptation"],
                "complexity": "Easy",
                "memory": "Low",
                "data_format": "instruction-input-output",
            },
            {
                "id": "qlora",
                "name": "QLoRA (Quantized LoRA)",
                "description": "4-bit quantized LoRA for consumer GPUs",
                "use_cases": ["Consumer GPU", "Large models"],
                "complexity": "Easy",
                "memory": "Very Low",
                "data_format": "instruction-input-output",
            },
            {
                "id": "dpo",
                "name": "DPO (Direct Preference Optimization)",
                "description": "Align model with preferences without reward model",
                "use_cases": ["Alignment", "Safety", "Quality improvement"],
                "complexity": "Medium",
                "memory": "Medium",
                "data_format": "prompt-chosen-rejected",
            },
            {
                "id": "orpo",
                "name": "ORPO (Odds Ratio Preference Optimization)",
                "description": "Single-stage preference optimization",
                "use_cases": ["Alignment", "Simpler than DPO"],
                "complexity": "Medium",
                "memory": "Medium",
                "data_format": "prompt-chosen-rejected",
            },
        ]
    }


# WebSocket connection manager for training progress
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

        # Register callback with training service
        from app.services.training_service import training_service
        training_service.register_progress_callback(
            job_id,
            lambda data: self.broadcast(job_id, data)
        )

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            try:
                self.active_connections[job_id].remove(websocket)
            except ValueError:
                pass
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@app.websocket("/ws/training/{job_id}")
async def websocket_training_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training progress"""
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Could handle client commands here (e.g., stop, pause)
            await websocket.send_json({"type": "ack", "job_id": job_id})
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
