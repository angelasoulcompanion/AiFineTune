"""
Base Trainer - Abstract base class for training implementations
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from uuid import UUID
from pathlib import Path
import logging
import time

from ..config import settings
from .utils import (
    get_memory_usage,
    TrainingMetrics,
    estimate_training_duration,
    estimate_model_parameters,
    estimate_memory_required,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for trainers"""

    def __init__(
        self,
        job_id: UUID,
        progress_callback: Optional[Callable[[dict], Any]] = None,
    ):
        self.job_id = job_id
        self.progress_callback = progress_callback
        self.checkpoints_dir = Path(settings.CHECKPOINTS_DIR) / str(job_id)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.is_cancelled = False
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0

        # Metrics collection
        self.metrics = TrainingMetrics()
        self.training_start_time: Optional[float] = None
        self.last_memory_check: float = 0
        self.memory_check_interval: float = 30  # seconds

    @abstractmethod
    async def train(self, job: dict) -> dict:
        """
        Run training

        Args:
            job: Training job configuration

        Returns:
            dict with keys:
                - success: bool
                - output_path: str (path to trained model)
                - output_size_mb: float
                - error: str (if failed)
        """
        pass

    def cancel(self):
        """Cancel training"""
        self.is_cancelled = True

    def _report_progress(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """Report training progress"""
        if self.progress_callback:
            progress_pct = 0
            if total_steps and step:
                progress_pct = (step / total_steps) * 100

            data = {
                "type": "progress",
                "job_id": str(self.job_id),
                "epoch": epoch or self.current_epoch,
                "total_epochs": kwargs.get("total_epochs"),
                "step": step or self.current_step,
                "total_steps": total_steps or self.total_steps,
                "progress": round(progress_pct, 2),
                "loss": round(loss, 6) if loss else None,
                "learning_rate": learning_rate,
                **kwargs
            }
            self.progress_callback(data)

    def _report_status(self, status: str, message: Optional[str] = None):
        """Report status change"""
        if self.progress_callback:
            self.progress_callback({
                "type": "status",
                "job_id": str(self.job_id),
                "status": status,
                "message": message,
            })

    def _report_metrics(self, metrics: dict):
        """Report training metrics"""
        if self.progress_callback:
            self.progress_callback({
                "type": "metrics",
                "job_id": str(self.job_id),
                "metrics": metrics,
            })

    def _report_log(self, level: str, message: str):
        """Report log message"""
        if self.progress_callback:
            self.progress_callback({
                "type": "log",
                "job_id": str(self.job_id),
                "level": level,
                "message": message,
            })

    def _get_device(self) -> str:
        """Get best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _get_compute_dtype(self, device: str):
        """Get appropriate compute dtype for device"""
        import torch

        if device == "cuda":
            # Check for bf16 support
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        elif device == "mps":
            # MPS works best with float16
            return torch.float16
        else:
            return torch.float32

    def _start_training_timer(self):
        """Start tracking training time"""
        self.training_start_time = time.time()

    def _get_elapsed_time(self) -> float:
        """Get elapsed training time in seconds"""
        if self.training_start_time:
            return time.time() - self.training_start_time
        return 0.0

    def _check_and_report_memory(self, force: bool = False):
        """Check and report memory usage periodically"""
        now = time.time()
        if not force and (now - self.last_memory_check) < self.memory_check_interval:
            return

        self.last_memory_check = now
        memory = get_memory_usage()
        if memory:
            self.metrics.add_step(
                step=self.current_step,
                memory_gb=memory.used_gb,
            )
            self._report_memory(memory.to_dict())

    def _report_memory(self, memory_info: dict):
        """Report memory usage via callback"""
        if self.progress_callback:
            self.progress_callback({
                "type": "memory",
                "job_id": str(self.job_id),
                "memory": memory_info,
            })

    def _finalize_metrics(self, batch_size: int = 4):
        """Finalize metrics after training"""
        elapsed = self._get_elapsed_time()
        self.metrics.total_training_time = elapsed

        if elapsed > 0 and self.current_step > 0:
            self.metrics.steps_per_second = self.current_step / elapsed
            self.metrics.samples_per_second = (self.current_step * batch_size) / elapsed

    def estimate_duration(
        self,
        num_samples: int,
        num_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        model_id: str,
        method: str = "lora",
        max_seq_length: int = 2048,
    ) -> dict:
        """Estimate training duration"""
        device = self._get_device()
        model_size = estimate_model_parameters(model_id)

        estimate = estimate_training_duration(
            num_samples=num_samples,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            model_size_b=model_size,
            device=device,
            method=method,
            max_seq_length=max_seq_length,
        )

        # Report estimation
        if self.progress_callback:
            self.progress_callback({
                "type": "estimate",
                "job_id": str(self.job_id),
                "duration": {
                    "formatted": estimate.formatted,
                    "seconds": estimate.estimated_seconds,
                    "confidence": estimate.confidence,
                    "basis": estimate.basis,
                },
            })

        return {
            "formatted": estimate.formatted,
            "seconds": estimate.estimated_seconds,
            "confidence": estimate.confidence,
            "basis": estimate.basis,
        }

    def estimate_memory(
        self,
        model_id: str,
        method: str = "lora",
        batch_size: int = 4,
        max_seq_length: int = 2048,
    ) -> dict:
        """Estimate memory requirements"""
        model_size = estimate_model_parameters(model_id)
        return estimate_memory_required(
            model_size_b=model_size,
            method=method,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )

    def get_metrics(self) -> dict:
        """Get collected training metrics"""
        return self.metrics.to_dict()

    def get_metrics_summary(self) -> dict:
        """Get metrics summary"""
        return self.metrics.get_summary()
