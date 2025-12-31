"""
Model Service - Business logic for model management
Including HuggingFace integration for search/download
"""
import os
import asyncio
import shutil
from typing import Optional
from uuid import UUID
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, list_models
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from ..repositories.model_repository import ModelRepository
from ..config import settings


# Popular models for suggestions
POPULAR_BASE_MODELS = [
    {
        "model_id": "unsloth/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B Instruct",
        "params": "1B",
        "family": "Llama",
        "quantization": None,
    },
    {
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B Instruct",
        "params": "3B",
        "family": "Llama",
        "quantization": None,
    },
    {
        "model_id": "unsloth/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "params": "1.5B",
        "family": "Qwen",
        "quantization": None,
    },
    {
        "model_id": "unsloth/Qwen2.5-3B-Instruct",
        "name": "Qwen 2.5 3B Instruct",
        "params": "3B",
        "family": "Qwen",
        "quantization": None,
    },
    {
        "model_id": "unsloth/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B Instruct",
        "params": "7B",
        "family": "Qwen",
        "quantization": None,
    },
    {
        "model_id": "unsloth/gemma-2-2b-it",
        "name": "Gemma 2 2B Instruct",
        "params": "2B",
        "family": "Gemma",
        "quantization": None,
    },
    {
        "model_id": "unsloth/Phi-3.5-mini-instruct",
        "name": "Phi 3.5 Mini Instruct",
        "params": "3.8B",
        "family": "Phi",
        "quantization": None,
    },
    {
        "model_id": "unsloth/mistral-7b-instruct-v0.3",
        "name": "Mistral 7B Instruct v0.3",
        "params": "7B",
        "family": "Mistral",
        "quantization": None,
    },
]


class ModelService:
    """Service for model operations"""

    def __init__(self):
        self.repo = ModelRepository()
        self.models_dir = Path(settings.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_api = HfApi()

    # ==================== HuggingFace Search ====================

    async def search_huggingface(
        self,
        query: str,
        limit: int = 20,
        filter_task: Optional[str] = "text-generation",
    ) -> list[dict]:
        """Search models on HuggingFace Hub"""
        try:
            # Run in thread pool (HfApi is sync)
            models = await asyncio.to_thread(
                lambda: list(list_models(
                    search=query,
                    task=filter_task,
                    sort="downloads",
                    direction=-1,
                    limit=limit,
                ))
            )

            results = []
            for model in models:
                results.append({
                    "model_id": model.id,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags[:5] if model.tags else [],
                    "pipeline_tag": model.pipeline_tag,
                    "last_modified": model.last_modified.isoformat() if model.last_modified else None,
                })

            return results

        except Exception as e:
            raise ValueError(f"Failed to search HuggingFace: {str(e)}")

    async def get_model_info(self, hf_model_id: str, hf_token: Optional[str] = None) -> dict:
        """Get detailed info about a HuggingFace model"""
        try:
            info = await asyncio.to_thread(
                lambda: self.hf_api.model_info(hf_model_id, token=hf_token)
            )

            # Calculate approximate size
            size_bytes = 0
            files = []
            if info.siblings:
                for sibling in info.siblings:
                    if sibling.size:
                        size_bytes += sibling.size
                    files.append({
                        "filename": sibling.rfilename,
                        "size": sibling.size,
                    })

            return {
                "model_id": info.id,
                "author": info.author,
                "sha": info.sha,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else None,
                "files": files[:20],  # Limit files shown
                "gated": info.gated,
                "private": info.private,
            }

        except RepositoryNotFoundError:
            raise ValueError(f"Model not found: {hf_model_id}")
        except GatedRepoError:
            raise ValueError(f"Gated model - HuggingFace token required: {hf_model_id}")
        except Exception as e:
            raise ValueError(f"Failed to get model info: {str(e)}")

    def get_popular_models(self) -> list[dict]:
        """Get list of popular/recommended base models"""
        return POPULAR_BASE_MODELS

    # ==================== Download ====================

    async def download_model(
        self,
        user_id: UUID,
        hf_model_id: str,
        name: str,
        hf_token: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """Download a model from HuggingFace and register it"""
        # Check if already exists
        models, _ = await self.repo.get_by_user(user_id)
        for m in models:
            if m["base_model_id"] == hf_model_id:
                raise ValueError(f"Model already registered: {hf_model_id}")

        # Create record with downloading status
        model = await self.repo.create(
            user_id=user_id,
            name=name,
            model_type="base",
            base_model_id=hf_model_id,
            description=description,
            hf_repo_id=hf_model_id,
        )

        try:
            # Update to downloading
            await self.repo.update_status(model["model_id"], "downloading")

            # Download model files
            model_dir = self.models_dir / str(user_id) / hf_model_id.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download snapshot (all model files)
            local_dir = await asyncio.to_thread(
                lambda: snapshot_download(
                    hf_model_id,
                    local_dir=str(model_dir),
                    token=hf_token,
                    ignore_patterns=["*.md", "*.txt", ".gitattributes"],
                )
            )

            # Calculate size
            total_size = sum(
                f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()
            )
            size_mb = round(total_size / (1024 * 1024), 2)

            # Update record
            model = await self.repo.update(
                model["model_id"],
                file_path=str(local_dir),
                file_size_mb=size_mb,
                status="ready",
            )

            return model

        except Exception as e:
            # Mark as error
            await self.repo.update_status(
                model["model_id"],
                "error",
                str(e)
            )
            raise ValueError(f"Failed to download model: {str(e)}")

    # ==================== CRUD ====================

    async def create_model(
        self,
        user_id: UUID,
        name: str,
        model_type: str,
        base_model_id: str,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        file_size_mb: Optional[float] = None,
        hf_repo_id: Optional[str] = None,
    ) -> dict:
        """Create a model record (for locally trained models)"""
        model = await self.repo.create(
            user_id=user_id,
            name=name,
            model_type=model_type,
            base_model_id=base_model_id,
            description=description,
            file_path=file_path,
            file_size_mb=file_size_mb,
            hf_repo_id=hf_repo_id,
        )
        return model

    async def get_model(self, model_id: UUID, user_id: UUID) -> dict:
        """Get model by ID (with ownership check)"""
        model = await self.repo.get_by_id(model_id)
        if not model:
            raise ValueError("Model not found")
        if model["user_id"] != user_id:
            raise ValueError("Access denied")
        return model

    async def list_models(
        self,
        user_id: UUID,
        page: int = 1,
        per_page: int = 20,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """List user's models with pagination"""
        models, total = await self.repo.get_by_user(
            user_id, page, per_page, model_type, status
        )
        return {
            "models": models,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
        }

    async def delete_model(self, model_id: UUID, user_id: UUID) -> bool:
        """Delete a model"""
        model = await self.get_model(model_id, user_id)

        # Delete local files if exists
        if model.get("file_path") and os.path.exists(model["file_path"]):
            try:
                shutil.rmtree(model["file_path"])
            except Exception:
                pass  # Ignore deletion errors

        return await self.repo.delete(model_id)

    # ==================== HuggingFace Push ====================

    async def push_to_huggingface(
        self,
        model_id: UUID,
        user_id: UUID,
        repo_name: str,
        hf_token: str,
        private: bool = False,
    ) -> dict:
        """Push model to HuggingFace Hub"""
        model = await self.get_model(model_id, user_id)

        if not model.get("file_path"):
            raise ValueError("Model has no local files to push")

        try:
            # Create repo
            repo_url = await asyncio.to_thread(
                lambda: self.hf_api.create_repo(
                    repo_id=repo_name,
                    token=hf_token,
                    private=private,
                    exist_ok=True,
                )
            )

            # Upload folder
            await asyncio.to_thread(
                lambda: self.hf_api.upload_folder(
                    folder_path=model["file_path"],
                    repo_id=repo_name,
                    token=hf_token,
                )
            )

            # Update record
            model = await self.repo.update_hf_push(
                model_id, repo_name, True
            )

            return {
                **model,
                "hf_url": f"https://huggingface.co/{repo_name}",
            }

        except Exception as e:
            raise ValueError(f"Failed to push to HuggingFace: {str(e)}")

    # ==================== Ollama Integration ====================

    async def import_to_ollama(
        self,
        model_id: UUID,
        user_id: UUID,
        ollama_name: str,
    ) -> dict:
        """Import model to Ollama (create Modelfile and run ollama create)"""
        model = await self.get_model(model_id, user_id)

        if not model.get("file_path"):
            raise ValueError("Model has no local files")

        model_path = Path(model["file_path"])

        # Find safetensors or bin file
        model_file = None
        for ext in ["*.safetensors", "*.bin", "*.gguf"]:
            files = list(model_path.glob(ext))
            if files:
                model_file = files[0]
                break

        if not model_file:
            raise ValueError("No model weights found (safetensors/bin/gguf)")

        try:
            # Create Modelfile
            modelfile_content = f"""FROM {model_file}
TEMPLATE \"\"\"{{{{ if .System }}}}{{ .System }}{{{{ end }}}}{{{{ if .Prompt }}}}{{ .Prompt }}{{{{ end }}}}\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
            modelfile_path = model_path / "Modelfile"
            modelfile_path.write_text(modelfile_content)

            # Run ollama create
            process = await asyncio.create_subprocess_exec(
                "ollama", "create", ollama_name, "-f", str(modelfile_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise ValueError(f"Ollama error: {stderr.decode()}")

            # Update record
            model = await self.repo.update_ollama(model_id, ollama_name, True)

            return model

        except FileNotFoundError:
            raise ValueError("Ollama not installed. Please install from https://ollama.ai")
        except Exception as e:
            raise ValueError(f"Failed to import to Ollama: {str(e)}")

    # ==================== Test Model ====================

    async def test_model(
        self,
        model_id: UUID,
        user_id: UUID,
        prompt: str,
        max_tokens: int = 256,
    ) -> dict:
        """Test model with a prompt (via Ollama if available)"""
        model = await self.get_model(model_id, user_id)

        if not model.get("ollama_model_name"):
            raise ValueError("Model not imported to Ollama. Import first to test.")

        try:
            # Use ollama run
            process = await asyncio.create_subprocess_exec(
                "ollama", "run", model["ollama_model_name"],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode()),
                timeout=60.0
            )

            if process.returncode != 0:
                raise ValueError(f"Ollama error: {stderr.decode()}")

            return {
                "model": model["ollama_model_name"],
                "prompt": prompt,
                "response": stdout.decode().strip(),
            }

        except asyncio.TimeoutError:
            raise ValueError("Model response timed out")
        except FileNotFoundError:
            raise ValueError("Ollama not installed")
        except Exception as e:
            raise ValueError(f"Failed to test model: {str(e)}")
