"""
Modal.com Cloud Trainer - Serverless GPU training
Supports A10G, A100, H100 GPUs on demand
"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Callable, Any
from uuid import UUID
from datetime import datetime

from .base_trainer import BaseTrainer
from ..config import settings

logger = logging.getLogger(__name__)

# Modal GPU configurations with pricing (approximate $/hour)
MODAL_GPU_CONFIGS = {
    "T4": {
        "gpu": "T4",
        "memory": 16,
        "cost_per_hour": 0.59,
        "best_for": ["7B models with QLoRA", "Small models"],
    },
    "A10G": {
        "gpu": "A10G",
        "memory": 24,
        "cost_per_hour": 1.10,
        "best_for": ["7B-13B models", "LoRA training"],
    },
    "A100-40GB": {
        "gpu": "A100",
        "memory": 40,
        "cost_per_hour": 3.40,
        "best_for": ["13B-34B models", "Full fine-tuning"],
    },
    "A100-80GB": {
        "gpu": "A100-80GB",
        "memory": 80,
        "cost_per_hour": 4.50,
        "best_for": ["34B-70B models", "Large batch sizes"],
    },
    "H100": {
        "gpu": "H100",
        "memory": 80,
        "cost_per_hour": 6.00,
        "best_for": ["70B+ models", "Maximum performance"],
    },
}


def get_recommended_gpu(model_size_b: float, method: str) -> str:
    """Recommend GPU based on model size and training method"""
    if method == "qlora":
        if model_size_b <= 7:
            return "T4"
        elif model_size_b <= 13:
            return "A10G"
        elif model_size_b <= 34:
            return "A100-40GB"
        else:
            return "A100-80GB"
    elif method in ("lora", "dpo", "orpo"):
        if model_size_b <= 7:
            return "A10G"
        elif model_size_b <= 13:
            return "A100-40GB"
        else:
            return "A100-80GB"
    else:  # SFT
        if model_size_b <= 7:
            return "A100-40GB"
        else:
            return "A100-80GB"


def estimate_modal_cost(
    model_size_b: float,
    num_samples: int,
    num_epochs: int,
    method: str,
    gpu_type: Optional[str] = None,
) -> dict:
    """Estimate Modal.com training cost"""
    from .utils import estimate_training_duration

    # Get recommended GPU if not specified
    if not gpu_type:
        gpu_type = get_recommended_gpu(model_size_b, method)

    gpu_config = MODAL_GPU_CONFIGS.get(gpu_type, MODAL_GPU_CONFIGS["A10G"])

    # Estimate duration (Modal is typically faster than local due to better GPUs)
    duration = estimate_training_duration(
        num_samples=num_samples,
        num_epochs=num_epochs,
        batch_size=8,  # Modal can handle larger batches
        gradient_accumulation_steps=2,
        model_size_b=model_size_b,
        device="cuda",  # Modal uses CUDA
        method=method,
    )

    # Modal overhead (cold start, setup, etc.) - about 2-5 minutes
    overhead_minutes = 3
    total_minutes = (duration.estimated_seconds / 60) + overhead_minutes

    # Calculate cost
    cost_per_minute = gpu_config["cost_per_hour"] / 60
    estimated_cost = total_minutes * cost_per_minute

    return {
        "gpu_type": gpu_type,
        "gpu_memory_gb": gpu_config["memory"],
        "estimated_minutes": round(total_minutes, 1),
        "estimated_cost_usd": round(estimated_cost, 2),
        "cost_per_hour": gpu_config["cost_per_hour"],
        "best_for": gpu_config["best_for"],
        "confidence": duration.confidence,
    }


class ModalTrainer(BaseTrainer):
    """Cloud trainer using Modal.com serverless GPUs"""

    def __init__(
        self,
        job_id: UUID,
        progress_callback: Optional[Callable[[dict], Any]] = None,
        gpu_type: str = "A10G",
    ):
        super().__init__(job_id, progress_callback)
        self.gpu_type = gpu_type
        self.modal_function_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.cost_tracker = {
            "gpu_type": gpu_type,
            "start_time": None,
            "end_time": None,
            "duration_minutes": 0,
            "estimated_cost_usd": 0,
        }

    async def train(self, job: dict) -> dict:
        """Run training on Modal.com"""
        self._report_status("preparing", "Connecting to Modal.com...")

        try:
            # Check for Modal token
            modal_token = os.environ.get("MODAL_TOKEN_ID")
            if not modal_token:
                return {
                    "success": False,
                    "error": "Modal token not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                }

            # Import Modal
            try:
                import modal
            except ImportError:
                return {
                    "success": False,
                    "error": "Modal package not installed. Run: pip install modal",
                }

            self._report_log("info", f"Starting Modal training with {self.gpu_type} GPU")
            self.start_time = datetime.now()
            self.cost_tracker["start_time"] = self.start_time.isoformat()

            # Run training on Modal
            result = await self._run_modal_training(job)

            # Calculate final cost
            if self.start_time:
                end_time = datetime.now()
                self.cost_tracker["end_time"] = end_time.isoformat()
                duration = (end_time - self.start_time).total_seconds() / 60
                self.cost_tracker["duration_minutes"] = round(duration, 2)

                gpu_config = MODAL_GPU_CONFIGS.get(self.gpu_type, MODAL_GPU_CONFIGS["A10G"])
                self.cost_tracker["estimated_cost_usd"] = round(
                    duration * (gpu_config["cost_per_hour"] / 60), 2
                )

            if result.get("success"):
                result["cost"] = self.cost_tracker

            return result

        except Exception as e:
            logger.exception(f"Modal training failed for job {self.job_id}")
            return {
                "success": False,
                "error": str(e),
                "cost": self.cost_tracker,
            }

    async def _run_modal_training(self, job: dict) -> dict:
        """Execute training on Modal serverless GPU"""
        import modal

        # Create Modal app for this training job
        app = modal.App(f"aifinetune-{self.job_id}")

        # Define the training image
        training_image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install(
                "torch>=2.1.0",
                "transformers>=4.36.0",
                "datasets>=2.16.0",
                "accelerate>=0.25.0",
                "peft>=0.7.0",
                "trl>=0.7.0",
                "bitsandbytes>=0.42.0",
                "sentencepiece>=0.1.99",
                "huggingface-hub>=0.20.0",
            )
        )

        # Get GPU type
        gpu_map = {
            "T4": modal.gpu.T4(),
            "A10G": modal.gpu.A10G(),
            "A100-40GB": modal.gpu.A100(size="40GB"),
            "A100-80GB": modal.gpu.A100(size="80GB"),
            "H100": modal.gpu.H100(),
        }
        gpu = gpu_map.get(self.gpu_type, modal.gpu.A10G())

        config = job.get("config", {})
        training_config = config.get("training", {})
        lora_config = config.get("lora", {})
        method = job.get("training_method", "lora")
        model_id = job.get("hf_model_id") or job.get("model_path")
        dataset_path = job.get("dataset_path")

        # Define the training function
        @app.function(
            image=training_image,
            gpu=gpu,
            timeout=3600 * 4,  # 4 hour timeout
            secrets=[modal.Secret.from_name("huggingface-token")],
        )
        def train_on_modal(
            model_id: str,
            dataset_content: str,
            method: str,
            training_config: dict,
            lora_config: dict,
            job_id: str,
        ) -> dict:
            """Training function that runs on Modal GPU"""
            import json
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
            from trl import SFTTrainer

            # Parse dataset
            data = [json.loads(line) for line in dataset_content.strip().split("\n")]
            dataset = Dataset.from_list(data)

            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Determine quantization
            load_in_4bit = method == "qlora"

            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

            # Add LoRA
            if method in ("lora", "qlora"):
                peft_config = LoraConfig(
                    r=lora_config.get("lora_r", 16),
                    lora_alpha=lora_config.get("lora_alpha", 32),
                    lora_dropout=lora_config.get("lora_dropout", 0.05),
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, peft_config)

            # Format dataset
            def formatting_func(example):
                instruction = example.get("instruction", example.get("prompt", ""))
                input_text = example.get("input", "")
                output = example.get("output", example.get("response", ""))

                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

                return {"text": text + tokenizer.eos_token}

            dataset = dataset.map(formatting_func)

            # Training arguments
            output_dir = "/tmp/output"
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=training_config.get("num_train_epochs", 3),
                per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
                gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
                learning_rate=training_config.get("learning_rate", 2e-4),
                warmup_ratio=training_config.get("warmup_ratio", 0.03),
                weight_decay=training_config.get("weight_decay", 0.01),
                logging_steps=10,
                save_steps=100,
                bf16=True,
                optim="adamw_8bit",
                lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
                gradient_checkpointing=True,
                report_to="none",
            )

            # Train
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=training_config.get("max_seq_length", 2048),
                args=training_args,
            )

            trainer.train()

            # Save model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Get final loss
            final_loss = trainer.state.log_history[-1].get("loss", 0) if trainer.state.log_history else 0

            return {
                "success": True,
                "final_loss": final_loss,
                "output_dir": output_dir,
            }

        # Read dataset content
        self._report_status("preparing", "Uploading dataset to Modal...")
        with open(dataset_path, "r") as f:
            dataset_content = f.read()

        # Run training on Modal
        self._report_status("training", f"Training on Modal {self.gpu_type}...")

        with app.run():
            result = train_on_modal.remote(
                model_id=model_id,
                dataset_content=dataset_content,
                method=method,
                training_config=training_config,
                lora_config=lora_config,
                job_id=str(self.job_id),
            )

        if result.get("success"):
            self._report_status("completed", "Modal training completed!")
            return {
                "success": True,
                "final_loss": result.get("final_loss"),
                "output_path": result.get("output_dir"),
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Training failed"),
            }

    def cancel(self):
        """Cancel Modal training"""
        super().cancel()
        # Modal jobs auto-terminate, but we can add explicit cancellation
        self._report_log("warning", "Cancellation requested - Modal job will terminate")


# Standalone Modal app for deployment
def create_modal_training_app():
    """Create a Modal app that can be deployed separately"""
    import modal

    app = modal.App("aifinetune-training")

    training_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "datasets>=2.16.0",
            "accelerate>=0.25.0",
            "peft>=0.7.0",
            "trl>=0.7.0",
            "bitsandbytes>=0.42.0",
            "sentencepiece>=0.1.99",
            "huggingface-hub>=0.20.0",
        )
    )

    @app.function(image=training_image, gpu=modal.gpu.A10G(), timeout=3600 * 4)
    def health_check():
        """Check Modal connectivity and GPU availability"""
        import torch
        return {
            "status": "ok",
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        }

    return app
