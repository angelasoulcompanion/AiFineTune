"""
Local Trainer - Training on local hardware (MPS/CUDA/CPU)
Supports SFT, LoRA, QLoRA using Unsloth for efficiency
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Callable, Any
from uuid import UUID
import logging

from .base_trainer import BaseTrainer
from ..config import settings

logger = logging.getLogger(__name__)


class LocalTrainer(BaseTrainer):
    """Local trainer using Unsloth/Transformers"""

    def __init__(
        self,
        job_id: UUID,
        progress_callback: Optional[Callable[[dict], Any]] = None,
    ):
        super().__init__(job_id, progress_callback)

    async def train(self, job: dict) -> dict:
        """Run training locally"""
        self._report_status("preparing", "Loading libraries...")

        try:
            # Run training in thread pool to not block event loop
            result = await asyncio.to_thread(self._run_training_sync, job)
            return result
        except Exception as e:
            logger.exception(f"Training failed for job {self.job_id}")
            return {
                "success": False,
                "error": str(e),
            }

    def _run_training_sync(self, job: dict) -> dict:
        """Synchronous training implementation"""
        config = job.get("config", {})
        training_config = config.get("training", {})
        lora_config = config.get("lora", {})
        method = job.get("training_method", "lora")

        # Get paths
        dataset_path = job.get("dataset_path")
        model_path = job.get("model_path")
        hf_model_id = job.get("hf_model_id")

        if not dataset_path:
            return {"success": False, "error": "Dataset path not found"}

        # Use HF model ID if local path not available
        model_id = model_path if model_path and os.path.exists(model_path) else hf_model_id

        if not model_id:
            return {"success": False, "error": "Model not found"}

        self._report_log("info", f"Training method: {method}")
        self._report_log("info", f"Model: {model_id}")
        self._report_log("info", f"Dataset: {dataset_path}")

        # Route to appropriate trainer based on method
        if method in ("dpo",):
            return self._train_dpo(job, model_id, dataset_path, config)
        elif method in ("orpo",):
            return self._train_orpo(job, model_id, dataset_path, config)
        else:
            # SFT, LoRA, QLoRA
            try:
                # Try to use Unsloth first (much faster)
                return self._train_with_unsloth(job, model_id, dataset_path, config)
            except ImportError:
                self._report_log("warning", "Unsloth not available, using standard transformers")
                return self._train_with_transformers(job, model_id, dataset_path, config)

    def _train_with_unsloth(self, job: dict, model_id: str, dataset_path: str, config: dict) -> dict:
        """Train using Unsloth (optimized for speed)"""
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
        import torch

        training_config = config.get("training", {})
        lora_config = config.get("lora", {})
        method = job.get("training_method", "lora")

        self._report_status("preparing", "Loading model with Unsloth...")

        # Determine dtype
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

        # Load model
        max_seq_length = training_config.get("max_seq_length", 2048)
        load_in_4bit = method == "qlora" or lora_config.get("load_in_4bit", False)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        self._report_log("info", f"Model loaded: {model_id}")

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get("lora_r", 16),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        self._report_log("info", "LoRA adapters added")
        self._report_status("preparing", "Loading dataset...")

        # Load dataset
        dataset = self._load_dataset(dataset_path, job.get("dataset_type", "sft"))

        # Format dataset for training
        def formatting_prompts_func(examples):
            texts = []
            for i in range(len(examples.get("instruction", examples.get("prompt", [])))):
                instruction = examples.get("instruction", examples.get("prompt", []))[i]
                input_text = examples.get("input", [""] * len(examples.get("instruction", [])))[i]
                output = examples.get("output", examples.get("response", []))[i]

                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

                texts.append(text + tokenizer.eos_token)
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)

        self._report_log("info", f"Dataset loaded: {len(dataset)} examples")
        self._report_status("training", "Starting training...")

        # Training arguments
        output_dir = str(self.checkpoints_dir / "output")
        num_epochs = training_config.get("num_train_epochs", 3)

        # Calculate total steps for progress
        batch_size = training_config.get("per_device_train_batch_size", 4)
        grad_accum = training_config.get("gradient_accumulation_steps", 4)
        effective_batch = batch_size * grad_accum
        steps_per_epoch = len(dataset) // effective_batch
        self.total_steps = steps_per_epoch * num_epochs

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=training_config.get("learning_rate", 2e-4),
            warmup_ratio=training_config.get("warmup_ratio", 0.03),
            weight_decay=training_config.get("weight_decay", 0.01),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=training_config.get("optim", "adamw_8bit"),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            seed=42,
            report_to="none",  # Disable wandb etc
        )

        # Create trainer with callback
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=training_args,
            callbacks=[self._create_progress_callback(num_epochs)],
        )

        # Train
        trainer.train()

        self._report_status("saving", "Saving model...")

        # Save model
        final_output = str(self.checkpoints_dir / "final")
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)

        # Calculate output size
        output_size = sum(
            f.stat().st_size for f in Path(final_output).rglob("*") if f.is_file()
        )
        output_size_mb = round(output_size / (1024 * 1024), 2)

        self._report_log("info", f"Model saved to {final_output}")

        return {
            "success": True,
            "output_path": final_output,
            "output_size_mb": output_size_mb,
        }

    def _train_with_transformers(self, job: dict, model_id: str, dataset_path: str, config: dict) -> dict:
        """Train using standard transformers (fallback)"""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
        import torch

        training_config = config.get("training", {})
        lora_config = config.get("lora", {})
        method = job.get("training_method", "lora")

        self._report_status("preparing", "Loading model...")

        # Determine device and dtype
        device = self._get_device()
        dtype = self._get_compute_dtype(device)

        self._report_log("info", f"Using device: {device}, dtype: {dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else None,
        }

        if method == "qlora":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=lora_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=lora_config.get("bnb_4bit_use_double_quant", True),
            )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if device == "mps":
            model = model.to(device)

        self._report_log("info", "Model loaded")

        # Add LoRA
        if method in ("lora", "qlora"):
            peft_config = LoraConfig(
                r=lora_config.get("lora_r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                target_modules=lora_config.get("target_modules"),
                bias=lora_config.get("bias", "none"),
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        self._report_status("preparing", "Loading dataset...")

        # Load and process dataset
        dataset = self._load_dataset(dataset_path, job.get("dataset_type", "sft"))

        def tokenize_function(examples):
            # Build prompts
            texts = []
            for i in range(len(examples.get("instruction", examples.get("prompt", [])))):
                instruction = examples.get("instruction", examples.get("prompt", []))[i]
                input_text = examples.get("input", [""] * len(examples.get("instruction", [])))[i] if "input" in examples else ""
                output = examples.get("output", examples.get("response", []))[i]

                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

                texts.append(text + tokenizer.eos_token)

            return tokenizer(
                texts,
                truncation=True,
                max_length=training_config.get("max_seq_length", 2048),
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        self._report_log("info", f"Dataset tokenized: {len(tokenized_dataset)} examples")
        self._report_status("training", "Starting training...")

        # Training
        output_dir = str(self.checkpoints_dir / "output")
        num_epochs = training_config.get("num_train_epochs", 3)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 2e-4),
            warmup_ratio=training_config.get("warmup_ratio", 0.03),
            weight_decay=training_config.get("weight_decay", 0.01),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            fp16=device == "cuda" and dtype == torch.float16,
            bf16=device == "cuda" and dtype == torch.bfloat16,
            optim=training_config.get("optim", "adamw_torch"),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[self._create_progress_callback(num_epochs)],
        )

        trainer.train()

        self._report_status("saving", "Saving model...")

        # Save
        final_output = str(self.checkpoints_dir / "final")
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)

        output_size = sum(
            f.stat().st_size for f in Path(final_output).rglob("*") if f.is_file()
        )

        return {
            "success": True,
            "output_path": final_output,
            "output_size_mb": round(output_size / (1024 * 1024), 2),
        }

    def _load_dataset(self, path: str, dataset_type: str):
        """Load dataset from file"""
        from datasets import Dataset
        import pandas as pd

        path = Path(path)

        if path.suffix == ".jsonl":
            data = []
            with open(path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            return Dataset.from_list(data)

        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return Dataset.from_list(data)
            return Dataset.from_dict(data)

        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            return Dataset.from_pandas(df)

        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return Dataset.from_pandas(df)

        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def _create_progress_callback(self, total_epochs: int):
        """Create a training callback for progress reporting"""
        from transformers import TrainerCallback

        trainer_self = self

        class ProgressCallback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.epoch_losses = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    loss = logs.get("loss")
                    lr = logs.get("learning_rate")

                    # Report progress
                    trainer_self._report_progress(
                        epoch=int(state.epoch) if state.epoch else 0,
                        total_epochs=total_epochs,
                        step=state.global_step,
                        total_steps=state.max_steps,
                        loss=loss,
                        learning_rate=lr,
                    )

                    # Collect metrics
                    trainer_self.current_step = state.global_step
                    trainer_self.metrics.add_step(
                        step=state.global_step,
                        loss=loss,
                        learning_rate=lr,
                    )

                    # Track losses for epoch average
                    if loss is not None:
                        self.epoch_losses.append(loss)

                    # Periodic memory check
                    trainer_self._check_and_report_memory()

            def on_epoch_end(self, args, state, control, **kwargs):
                trainer_self.current_epoch = int(state.epoch)
                trainer_self._report_log("info", f"Epoch {trainer_self.current_epoch}/{total_epochs} completed")

                # Calculate and record epoch average loss
                if self.epoch_losses:
                    avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
                    trainer_self.metrics.add_epoch_loss(trainer_self.current_epoch, avg_loss)
                    self.epoch_losses = []  # Reset for next epoch

                # Report memory at end of each epoch
                trainer_self._check_and_report_memory(force=True)

            def on_train_begin(self, args, state, control, **kwargs):
                trainer_self.total_steps = state.max_steps
                trainer_self._start_training_timer()
                trainer_self._report_log("info", f"Training started: {state.max_steps} total steps")

                # Initial memory check
                trainer_self._check_and_report_memory(force=True)

            def on_train_end(self, args, state, control, **kwargs):
                # Final metrics
                batch_size = args.per_device_train_batch_size
                trainer_self._finalize_metrics(batch_size)

                # Report final memory
                trainer_self._check_and_report_memory(force=True)

                # Log training stats
                elapsed = trainer_self._get_elapsed_time()
                if elapsed > 0:
                    mins = elapsed / 60
                    trainer_self._report_log("info", f"Training completed in {mins:.1f} minutes")
                    trainer_self._report_log("info", f"Speed: {trainer_self.metrics.steps_per_second:.2f} steps/sec")

        return ProgressCallback()

    def _train_dpo(self, job: dict, model_id: str, dataset_path: str, config: dict) -> dict:
        """Train using DPO (Direct Preference Optimization)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
        import torch

        training_config = config.get("training", {})
        lora_config = config.get("lora", {})
        dpo_config = config.get("dpo", {})

        self._report_status("preparing", "Loading model for DPO training...")

        # Determine device and dtype
        device = self._get_device()
        dtype = self._get_compute_dtype(device)

        self._report_log("info", f"Using device: {device}, dtype: {dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else None,
        }

        # QLoRA quantization if specified
        if lora_config.get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=lora_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=lora_config.get("bnb_4bit_use_double_quant", True),
            )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if device == "mps":
            model = model.to(device)

        self._report_log("info", "Model loaded")

        # Add LoRA
        peft_config = LoraConfig(
            r=lora_config.get("lora_r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        self._report_status("preparing", "Loading DPO dataset...")

        # Load dataset (expects prompt, chosen, rejected columns)
        dataset = self._load_dataset(dataset_path, "dpo")

        # Validate DPO dataset format
        required_cols = ["prompt", "chosen", "rejected"]
        missing = [c for c in required_cols if c not in dataset.column_names]
        if missing:
            return {"success": False, "error": f"DPO dataset missing columns: {missing}"}

        self._report_log("info", f"DPO dataset loaded: {len(dataset)} examples")
        self._report_status("training", "Starting DPO training...")

        # Training arguments
        output_dir = str(self.checkpoints_dir / "output")
        num_epochs = training_config.get("num_train_epochs", 1)

        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 5e-5),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            weight_decay=training_config.get("weight_decay", 0.01),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            fp16=device == "cuda" and dtype == torch.float16,
            bf16=device == "cuda" and dtype == torch.bfloat16,
            optim=training_config.get("optim", "adamw_8bit"),
            beta=dpo_config.get("beta", 0.1),
            loss_type=dpo_config.get("loss_type", "sigmoid"),
            max_length=training_config.get("max_seq_length", 1024),
            max_prompt_length=training_config.get("max_prompt_length", 512),
            report_to="none",
        )

        # Create DPO trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Will use implicit reference model
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[self._create_progress_callback(num_epochs)],
        )

        # Train
        trainer.train()

        self._report_status("saving", "Saving DPO model...")

        # Save model
        final_output = str(self.checkpoints_dir / "final")
        trainer.save_model(final_output)
        tokenizer.save_pretrained(final_output)

        output_size = sum(
            f.stat().st_size for f in Path(final_output).rglob("*") if f.is_file()
        )

        self._report_log("info", f"DPO model saved to {final_output}")

        return {
            "success": True,
            "output_path": final_output,
            "output_size_mb": round(output_size / (1024 * 1024), 2),
        }

    def _train_orpo(self, job: dict, model_id: str, dataset_path: str, config: dict) -> dict:
        """Train using ORPO (Odds Ratio Preference Optimization)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import ORPOTrainer, ORPOConfig
        from datasets import Dataset
        import torch

        training_config = config.get("training", {})
        lora_config = config.get("lora", {})
        orpo_config = config.get("orpo", {})

        self._report_status("preparing", "Loading model for ORPO training...")

        # Determine device and dtype
        device = self._get_device()
        dtype = self._get_compute_dtype(device)

        self._report_log("info", f"Using device: {device}, dtype: {dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else None,
        }

        # QLoRA quantization if specified
        if lora_config.get("load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=lora_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=lora_config.get("bnb_4bit_use_double_quant", True),
            )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if device == "mps":
            model = model.to(device)

        self._report_log("info", "Model loaded")

        # Add LoRA
        peft_config = LoraConfig(
            r=lora_config.get("lora_r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        self._report_status("preparing", "Loading ORPO dataset...")

        # Load dataset (expects prompt, chosen, rejected columns)
        dataset = self._load_dataset(dataset_path, "orpo")

        # Validate ORPO dataset format
        required_cols = ["prompt", "chosen", "rejected"]
        missing = [c for c in required_cols if c not in dataset.column_names]
        if missing:
            return {"success": False, "error": f"ORPO dataset missing columns: {missing}"}

        self._report_log("info", f"ORPO dataset loaded: {len(dataset)} examples")
        self._report_status("training", "Starting ORPO training...")

        # Training arguments
        output_dir = str(self.checkpoints_dir / "output")
        num_epochs = training_config.get("num_train_epochs", 1)

        training_args = ORPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 5e-5),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            weight_decay=training_config.get("weight_decay", 0.01),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            fp16=device == "cuda" and dtype == torch.float16,
            bf16=device == "cuda" and dtype == torch.bfloat16,
            optim=training_config.get("optim", "adamw_8bit"),
            beta=orpo_config.get("beta", 0.1),
            max_length=training_config.get("max_seq_length", 1024),
            max_prompt_length=training_config.get("max_prompt_length", 512),
            report_to="none",
        )

        # Create ORPO trainer
        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[self._create_progress_callback(num_epochs)],
        )

        # Train
        trainer.train()

        self._report_status("saving", "Saving ORPO model...")

        # Save model
        final_output = str(self.checkpoints_dir / "final")
        trainer.save_model(final_output)
        tokenizer.save_pretrained(final_output)

        output_size = sum(
            f.stat().st_size for f in Path(final_output).rglob("*") if f.is_file()
        )

        self._report_log("info", f"ORPO model saved to {final_output}")

        return {
            "success": True,
            "output_path": final_output,
            "output_size_mb": round(output_size / (1024 * 1024), 2),
        }
