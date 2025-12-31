"""
Training Utilities - Duration estimation, memory monitoring, metrics collection
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ==================== Duration Estimation ====================

@dataclass
class DurationEstimate:
    """Training duration estimation result"""
    estimated_seconds: float
    estimated_minutes: float
    estimated_hours: float
    formatted: str
    confidence: str  # "low", "medium", "high"
    basis: str  # What the estimate is based on


def estimate_training_duration(
    num_samples: int,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    model_size_b: float = 1.0,  # Model size in billions of parameters
    device: str = "cuda",
    method: str = "lora",
    max_seq_length: int = 2048,
) -> DurationEstimate:
    """
    Estimate training duration based on dataset size, config, and hardware.

    These are rough estimates based on typical training scenarios:
    - CUDA A100: ~0.5-1s per step for 7B model with LoRA
    - CUDA A10: ~1-2s per step for 7B model with LoRA
    - MPS M1/M2: ~3-5s per step for 7B model with LoRA
    - CPU: ~30-60s per step for 7B model
    """
    # Calculate total steps
    effective_batch = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, num_samples // effective_batch)
    total_steps = steps_per_epoch * num_epochs

    # Base time per step (seconds) - adjusted by device
    device_multipliers = {
        "cuda": 1.0,   # Baseline
        "mps": 3.0,    # ~3x slower than CUDA
        "cpu": 30.0,   # ~30x slower than CUDA
    }
    device_mult = device_multipliers.get(device, 1.0)

    # Method multipliers
    method_multipliers = {
        "sft": 1.2,    # Full fine-tuning is slower
        "lora": 1.0,   # Baseline
        "qlora": 0.9,  # Slightly faster due to quantization
        "dpo": 1.5,    # DPO needs reference model
        "orpo": 1.3,   # ORPO is slightly simpler than DPO
    }
    method_mult = method_multipliers.get(method, 1.0)

    # Model size multiplier (assuming 7B baseline)
    size_mult = model_size_b / 7.0

    # Sequence length multiplier (assuming 2048 baseline)
    seq_mult = max_seq_length / 2048.0

    # Base time per step for 7B LoRA on A100: ~0.8 seconds
    base_time_per_step = 0.8

    time_per_step = base_time_per_step * device_mult * method_mult * size_mult * seq_mult

    # Total estimated time
    estimated_seconds = total_steps * time_per_step

    # Add overhead (model loading, saving, etc.) - typically 2-5 minutes
    overhead_seconds = 180  # 3 minutes
    estimated_seconds += overhead_seconds

    # Determine confidence based on factors
    confidence = "medium"
    if device == "cuda" and method in ("lora", "qlora"):
        confidence = "high"
    elif device == "cpu" or method == "sft":
        confidence = "low"

    # Format output
    hours = estimated_seconds / 3600
    minutes = estimated_seconds / 60

    if hours >= 1:
        formatted = f"{hours:.1f} hours"
    elif minutes >= 1:
        formatted = f"{minutes:.0f} minutes"
    else:
        formatted = f"{estimated_seconds:.0f} seconds"

    return DurationEstimate(
        estimated_seconds=estimated_seconds,
        estimated_minutes=minutes,
        estimated_hours=hours,
        formatted=formatted,
        confidence=confidence,
        basis=f"{total_steps} steps, {device.upper()}, {method.upper()}, ~{model_size_b}B params"
    )


# ==================== Memory Monitoring ====================

@dataclass
class MemoryUsage:
    """Memory usage snapshot"""
    device: str
    used_gb: float
    total_gb: float
    available_gb: float
    percent_used: float

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "used_gb": round(self.used_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "percent_used": round(self.percent_used, 1),
        }


def get_memory_usage(device: str = "auto") -> Optional[MemoryUsage]:
    """Get current GPU/MPS/CPU memory usage"""
    try:
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if device == "cuda":
            # CUDA memory
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            available = total - used
            return MemoryUsage(
                device="cuda",
                used_gb=used,
                total_gb=total,
                available_gb=available,
                percent_used=(used / total) * 100 if total > 0 else 0,
            )

        elif device == "mps":
            # MPS memory (Apple Silicon)
            # MPS doesn't have direct memory query, use system memory
            import psutil
            mem = psutil.virtual_memory()
            return MemoryUsage(
                device="mps",
                used_gb=mem.used / (1024 ** 3),
                total_gb=mem.total / (1024 ** 3),
                available_gb=mem.available / (1024 ** 3),
                percent_used=mem.percent,
            )

        else:
            # CPU/System memory
            import psutil
            mem = psutil.virtual_memory()
            return MemoryUsage(
                device="cpu",
                used_gb=mem.used / (1024 ** 3),
                total_gb=mem.total / (1024 ** 3),
                available_gb=mem.available / (1024 ** 3),
                percent_used=mem.percent,
            )

    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return None


def get_cuda_memory_stats() -> Optional[dict]:
    """Get detailed CUDA memory statistics"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        return {
            "allocated": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved": torch.cuda.memory_reserved() / (1024 ** 3),
            "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "max_reserved": torch.cuda.max_memory_reserved() / (1024 ** 3),
        }
    except Exception:
        return None


def clear_memory_cache():
    """Clear GPU memory cache"""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        logger.warning(f"Failed to clear memory cache: {e}")


# ==================== Metrics Collection ====================

@dataclass
class TrainingMetrics:
    """Training metrics collection"""
    loss_history: list = field(default_factory=list)
    lr_history: list = field(default_factory=list)
    memory_history: list = field(default_factory=list)
    epoch_losses: list = field(default_factory=list)
    best_loss: Optional[float] = None
    total_training_time: float = 0.0
    steps_per_second: float = 0.0
    samples_per_second: float = 0.0

    def add_step(
        self,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        memory_gb: Optional[float] = None,
    ):
        """Record metrics for a training step"""
        if loss is not None:
            self.loss_history.append({"step": step, "loss": loss})
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss

        if learning_rate is not None:
            self.lr_history.append({"step": step, "lr": learning_rate})

        if memory_gb is not None:
            self.memory_history.append({"step": step, "memory_gb": memory_gb})

    def add_epoch_loss(self, epoch: int, avg_loss: float):
        """Record average loss for an epoch"""
        self.epoch_losses.append({"epoch": epoch, "avg_loss": avg_loss})

    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "loss_history": self.loss_history[-100:],  # Last 100 entries
            "lr_history": self.lr_history[-100:],
            "memory_history": self.memory_history[-100:],
            "epoch_losses": self.epoch_losses,
            "best_loss": self.best_loss,
            "total_training_time": round(self.total_training_time, 2),
            "steps_per_second": round(self.steps_per_second, 3),
            "samples_per_second": round(self.samples_per_second, 3),
        }

    def get_summary(self) -> dict:
        """Get metrics summary"""
        avg_loss = None
        if self.loss_history:
            recent_losses = [x["loss"] for x in self.loss_history[-20:]]
            avg_loss = sum(recent_losses) / len(recent_losses)

        return {
            "best_loss": self.best_loss,
            "current_loss": self.loss_history[-1]["loss"] if self.loss_history else None,
            "avg_recent_loss": round(avg_loss, 6) if avg_loss else None,
            "total_steps": len(self.loss_history),
            "total_epochs": len(self.epoch_losses),
            "training_time_minutes": round(self.total_training_time / 60, 2),
        }


# ==================== Model Size Estimation ====================

def estimate_model_parameters(model_id: str) -> float:
    """
    Estimate model size in billions of parameters from model ID.
    Returns approximate size based on common naming patterns.
    """
    model_lower = model_id.lower()

    # Common size patterns
    size_patterns = [
        ("70b", 70.0), ("65b", 65.0),
        ("34b", 34.0), ("33b", 33.0), ("32b", 32.0),
        ("27b", 27.0), ("26b", 26.0),
        ("14b", 14.0), ("13b", 13.0), ("12b", 12.0),
        ("8b", 8.0), ("7b", 7.0),
        ("3b", 3.0), ("2.7b", 2.7), ("2b", 2.0),
        ("1.5b", 1.5), ("1b", 1.0),
        ("500m", 0.5), ("350m", 0.35), ("125m", 0.125),
    ]

    for pattern, size in size_patterns:
        if pattern in model_lower:
            return size

    # Default to 7B if unknown
    return 7.0


def estimate_memory_required(
    model_size_b: float,
    method: str = "lora",
    batch_size: int = 4,
    max_seq_length: int = 2048,
) -> dict:
    """
    Estimate memory required for training.

    Returns dict with minimum and recommended GB.
    """
    # Base model memory (fp16)
    model_mem_gb = model_size_b * 2  # 2 bytes per param for fp16

    # Method-specific multipliers
    method_mults = {
        "sft": 4.0,     # Full fine-tuning needs optimizer states
        "lora": 1.5,    # LoRA adds ~50% overhead
        "qlora": 0.5,   # 4-bit quantization
        "dpo": 3.0,     # DPO needs reference model
        "orpo": 2.0,    # ORPO is more efficient
    }
    method_mult = method_mults.get(method, 1.5)

    # Activation memory (rough estimate)
    # Scales with batch size and sequence length
    activation_mem = (batch_size * max_seq_length * model_size_b * 0.002)

    # Total estimate
    min_mem = model_mem_gb * method_mult + activation_mem
    recommended = min_mem * 1.3  # 30% buffer

    return {
        "minimum_gb": round(min_mem, 1),
        "recommended_gb": round(recommended, 1),
        "model_memory_gb": round(model_mem_gb, 1),
        "method": method,
        "notes": get_memory_notes(method, model_size_b),
    }


def get_memory_notes(method: str, model_size_b: float) -> str:
    """Get memory-related notes for training configuration"""
    if method == "qlora" and model_size_b <= 8:
        return "QLoRA with 4-bit should fit on 16GB GPU"
    elif method == "qlora" and model_size_b <= 14:
        return "QLoRA with 4-bit should fit on 24GB GPU"
    elif method == "lora" and model_size_b <= 7:
        return "LoRA should fit on 16GB GPU with gradient checkpointing"
    elif method in ("dpo", "orpo") and model_size_b > 7:
        return "Consider using QLoRA for preference training on large models"
    elif method == "sft":
        return "Full SFT requires significant memory; consider LoRA/QLoRA"
    return ""
