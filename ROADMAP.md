# AiFineTune - Development Roadmap

> Fine-tuning platform for LLM models
> Last Updated: 2025-12-31 (Phase 1, 2, 3, 4, & 5 Complete!)

---

## Project Overview

**AiFineTune** is a web platform for fine-tuning Large Language Models with support for:
- **Training Methods**: SFT, LoRA, QLoRA, DPO, ORPO
- **Execution**: Local (MPS/CUDA/CPU), Modal.com, HuggingFace Spaces
- **Integrations**: HuggingFace Hub, Ollama

---

## Current Status

### Completed Features (90%)

| Feature | Backend | Frontend | Status |
|---------|---------|----------|--------|
| Authentication (JWT) | ✅ | ✅ | Working |
| User Registration/Login | ✅ | ✅ | Working |
| Dataset Upload | ✅ | ✅ | Working |
| Dataset Validation | ✅ | ✅ | Working |
| Dataset Preview/Stats | ✅ | ✅ | Working |
| Model Search (HuggingFace) | ✅ | ✅ | Working |
| Model Download from HF | ✅ | ✅ | Working |
| Model Push to HF | ✅ | ✅ | Working |
| Ollama Import | ✅ | ✅ | Working |
| Model Testing | ✅ | ✅ | Working |
| Training Job CRUD | ✅ | ✅ | Working |
| Training Templates | ✅ | ✅ | Working |
| WebSocket Progress | ✅ | ✅ | Need Testing |

### Completed Features (Phase 1, 2, & 3)

| Feature | Status | Notes |
|---------|--------|-------|
| LocalTrainer Execution | ✅ Done | All 5 methods working |
| HF Token Encryption | ✅ Done | Fernet AES-128-CBC |
| HF Token Validation | ✅ Done | HuggingFace API |
| Settings Page | ✅ Done | Profile, Token, Password |
| API Rate Limiting | ✅ Done | SlowAPI |
| DPO/ORPO Trainers | ✅ Done | All methods available |
| Training Duration Estimation | ✅ Done | Auto-estimate before training |
| GPU/MPS Memory Monitoring | ✅ Done | Real-time memory tracking |
| Training Loss Visualization | ✅ Done | Recharts loss/LR/memory charts |
| LR Scheduler Options | ✅ Done | 7 scheduler types |
| Advanced Hyperparameter UI | ✅ Done | Expandable config panel |
| Modal.com Cloud Training | ✅ Done | GPU on demand (T4-H100) |
| Cloud Cost Estimation | ✅ Done | Pre-training cost preview |
| Execution Environment UI | ✅ Done | Local/Cloud selection |

### Completed Features (Phase 5)

| Feature | Status | Notes |
|---------|--------|-------|
| Docker Deployment | ✅ Done | Multi-stage builds, docker-compose |
| Training Analytics Dashboard | ✅ Done | Stats, charts, recent jobs |
| Model Versioning System | ✅ Done | Auto-increment triggers, compare |
| API Documentation | ✅ Done | OpenAPI auto-generated via FastAPI |

### Remaining (Optional)

| Feature | Status | Impact |
|---------|--------|--------|
| Dataset Splitting UI | Future | train/val/test split |
| Batch Dataset Upload | Future | Multiple files at once |
| User Documentation | Future | Usage guides |

---

## Development Phases

### Phase 1: Training Core ✅ COMPLETED
> Make training actually work

**Priority**: P0 - Blocking
**Status**: ✅ DONE!

- [x] Complete LocalTrainer implementation
  - [x] Unsloth integration (optimized path)
  - [x] Standard transformers fallback
  - [x] SFT trainer
  - [x] LoRA/QLoRA trainer
  - [x] DPO trainer
  - [x] ORPO trainer
- [x] Checkpoint saving/loading
- [x] Output model creation after training
- [x] WebSocket progress integration
- [x] Error handling and recovery
- [x] Training cancellation
- [x] Fixed API field naming (`name` vs `job_name`)
- [x] Fixed database integration issues

**Files modified**:
- `backend/app/training/local_trainer.py` - Added DPO/ORPO trainers
- `backend/app/services/training_service.py` - Fixed integrations
- `backend/app/repositories/training_job_repository.py` - Fixed column names
- `backend/app/routers/training.py` - Fixed API models
- `frontend/src/hooks/useTraining.ts` - Fixed field names
- `frontend/src/components/training/*` - Fixed field references

---

### Phase 2: Security & Settings ✅ COMPLETED
> Security hardening and user experience

**Priority**: P1 - High
**Status**: ✅ DONE!

- [x] HF Token encryption
  - [x] Add cryptography library (Fernet AES-128-CBC)
  - [x] Encrypt before storing
  - [x] Decrypt when using (with backward compatibility)
- [x] HF Token validation with HuggingFace API
  - [x] Validate on save
  - [x] Separate validate endpoint
  - [x] Show username/permissions
- [x] Settings page UI
  - [x] User profile editing
  - [x] HF token management (save/validate/delete)
  - [x] Password change
- [x] API rate limiting (slowapi)
  - [x] Login: 10/minute
  - [x] Register: 5/minute
  - [x] Default: 100/minute

**Files created/modified**:
- `backend/app/utils/crypto.py` - Fernet encryption utilities
- `backend/app/utils/hf_validator.py` - HuggingFace API validation
- `backend/app/utils/rate_limiter.py` - SlowAPI rate limiting
- `backend/app/services/auth_service.py` - Updated HF token methods
- `backend/app/routers/auth.py` - New validation endpoint + rate limits
- `backend/app/models/auth.py` - Extended response models
- `frontend/src/hooks/useSettings.ts` - Settings hooks
- `frontend/src/pages/SettingsPage.tsx` - Settings UI
- `frontend/src/App.tsx` - Added settings route

---

### Phase 3: Advanced Training ✅ COMPLETED
> More training methods and monitoring

**Priority**: P2 - Medium
**Status**: ✅ DONE!

- [x] DPO trainer implementation (done in Phase 1)
- [x] ORPO trainer implementation (done in Phase 1)
- [x] Training duration estimation
- [x] GPU/MPS memory monitoring
- [x] Evaluation metrics collection
- [x] Training loss visualization (Recharts)
- [x] Learning rate scheduling options (7 schedulers)
- [x] Advanced hyperparameter tuning UI

**Files created/modified**:
- `backend/app/training/utils.py` - Duration/memory estimation utilities
- `backend/app/training/base_trainer.py` - Added metrics collection
- `backend/app/training/local_trainer.py` - Enhanced callback with metrics
- `backend/app/routers/training.py` - New estimation endpoints
- `frontend/src/components/training/LossChart.tsx` - Loss visualization
- `frontend/src/components/training/AdvancedConfigPanel.tsx` - Advanced UI
- `frontend/src/pages/TrainingDetailPage.tsx` - Charts tab, metrics display

---

### Phase 4: Cloud Training ✅ COMPLETED
> Scale beyond local machine

**Priority**: P2 - Medium
**Status**: ✅ DONE!

- [x] Modal.com trainer implementation
  - [x] GPU instance provisioning (T4, A10G, A100-40GB, A100-80GB, H100)
  - [x] Training execution
  - [x] Progress streaming via WebSocket
  - [x] Cost tracking per job
- [x] Cost estimation UI
- [x] Cloud job monitoring
- [x] Auto-shutdown on completion (Modal auto-terminates)
- [ ] HuggingFace Spaces trainer (optional, future)

**Files created/modified**:
- `backend/app/training/modal_trainer.py` - Modal.com trainer with GPU configs
- `backend/app/routers/training.py` - Added cloud endpoints
- `backend/requirements.txt` - Added modal>=0.64.0
- `frontend/src/components/training/CloudConfigPanel.tsx` - Cloud config UI
- `frontend/src/components/training/TrainingWizard.tsx` - Added environment step
- `frontend/src/hooks/useTraining.ts` - Added cloud hooks

---

### Phase 5: Polish & Production ✅ COMPLETED
> Production readiness

**Priority**: P3 - Low
**Status**: ✅ DONE!

- [x] Docker deployment
  - [x] Backend Dockerfile (multi-stage Python 3.11)
  - [x] Frontend Dockerfile (multi-stage Node + nginx)
  - [x] docker-compose.yml (PostgreSQL, Backend, Frontend)
  - [x] nginx.conf with API/WebSocket proxy
  - [x] .dockerignore files
  - [x] .env.example template
- [x] Training analytics dashboard
  - [x] Dashboard stats API (datasets, models, jobs counts)
  - [x] Analytics API (jobs/day, training time, loss by method)
  - [x] React dashboard with Recharts visualization
  - [x] Real-time job status display
- [x] Model versioning system
  - [x] Version columns (version, version_tag, is_latest, version_notes)
  - [x] Auto-increment trigger for versions
  - [x] GET /{model_id}/versions endpoint
  - [x] PUT /{model_id}/version endpoint
  - [x] GET /{model_id}/version-compare endpoint
- [x] API documentation (OpenAPI auto-generated via FastAPI /docs)
- [ ] Dataset splitting UI (optional, future)
- [ ] User documentation (optional, future)

**Files created/modified**:
- `backend/Dockerfile` - Multi-stage Python build
- `frontend/Dockerfile` - Multi-stage Node + nginx build
- `frontend/nginx.conf` - nginx config with API proxy
- `docker-compose.yml` - Full stack orchestration
- `backend/.dockerignore` - Build exclusions
- `frontend/.dockerignore` - Build exclusions
- `.env.example` - Environment template
- `backend/app/routers/dashboard.py` - Dashboard API
- `backend/main.py` - Added dashboard router
- `frontend/src/hooks/useDashboard.ts` - Dashboard hooks
- `frontend/src/pages/Dashboard.tsx` - Enhanced dashboard
- `backend/migrations/002_model_versioning.sql` - Version migration
- `backend/app/routers/models.py` - Version endpoints

---

## Tech Stack

### Backend
- **Framework**: FastAPI
- **Database**: PostgreSQL (asyncpg)
- **Auth**: JWT (PyJWT + bcrypt)
- **ML**: transformers, peft, trl, unsloth (optional)
- **Cloud**: Modal.com SDK

### Frontend
- **Framework**: React 19 + TypeScript
- **Build**: Vite 7
- **Styling**: Tailwind CSS v4
- **State**: TanStack Query (React Query)
- **Routing**: React Router v7
- **Icons**: Lucide React

### Database Schema
- `finetune_users` - User accounts
- `finetune_datasets` - Training datasets
- `finetune_models` - Base & fine-tuned models
- `finetune_training_jobs` - Training job tracking
- `finetune_training_templates` - Pre-configured templates

---

## API Endpoints Summary

### Auth (`/api/auth`)
- POST `/register` - User registration
- POST `/login` - Get JWT tokens
- POST `/refresh` - Refresh access token
- GET `/me` - Current user profile
- PUT `/me` - Update profile
- POST `/change-password` - Change password
- POST `/hf-token` - Save HF token
- GET `/hf-token` - Check HF token status
- DELETE `/hf-token` - Remove HF token

### Datasets (`/api/datasets`)
- GET `/` - List datasets
- POST `/upload` - Upload dataset
- GET `/{id}` - Get dataset details
- PUT `/{id}` - Update dataset
- DELETE `/{id}` - Delete dataset
- POST `/{id}/validate` - Validate dataset
- GET `/{id}/preview` - Preview rows
- GET `/{id}/statistics` - Get stats

### Models (`/api/models`)
- GET `/` - List models
- POST `/` - Create model record
- GET `/{id}` - Get model details
- PUT `/{id}` - Update model
- DELETE `/{id}` - Delete model
- GET `/huggingface/search` - Search HF Hub
- GET `/huggingface/info/{model_id}` - Model info
- GET `/huggingface/popular` - Popular models
- POST `/huggingface/download` - Download model
- POST `/{id}/push-hf` - Push to HuggingFace
- POST `/{id}/import-ollama` - Import to Ollama
- POST `/{id}/test` - Test model
- GET `/{id}/versions` - Get version history
- PUT `/{id}/version` - Update version metadata
- GET `/{id}/version-compare` - Compare two versions

### Training (`/api/training`)
- GET `/` - List training jobs
- POST `/` - Create training job
- GET `/templates` - Get templates
- GET `/{id}` - Get job details
- POST `/{id}/start` - Start job
- POST `/{id}/cancel` - Cancel job
- DELETE `/{id}` - Delete job
- GET `/{id}/status` - Get status
- GET `/{id}/metrics` - Get metrics

### Dashboard (`/api/dashboard`)
- GET `/stats` - Dashboard statistics (counts, recent jobs)
- GET `/analytics` - Training analytics (jobs/day, time by method)

### WebSocket
- WS `/ws/training/{job_id}` - Real-time progress

---

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Copy environment file
cp .env.example .env

# Edit .env with your settings (DB password, JWT secret, etc.)

# Start all services
docker-compose up -d

# Access: http://localhost:3000
```

### Option 2: Local Development

#### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

#### Database
```bash
createdb AiFineTune
psql -d AiFineTune -f backend/migrations/001_initial_schema.sql
psql -d AiFineTune -f backend/migrations/002_model_versioning.sql
```

---

## Contributing

1. Pick a task from current phase
2. Create feature branch
3. Implement with tests
4. Submit PR

---

## Notes

- Training requires GPU (CUDA) or Apple Silicon (MPS) for reasonable speed
- CPU training possible but very slow
- Unsloth provides 2x speedup if installed
- Modal.com requires account and API keys

---

*Created with 💜 by David & Angela*
