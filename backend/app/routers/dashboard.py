"""
Dashboard Router - Analytics and statistics endpoints
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends

from ..database import db
from .auth import get_current_user


router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats")
async def get_dashboard_stats(
    current_user=Depends(get_current_user),
):
    """Get dashboard statistics for current user"""
    user_id = current_user.user_id

    # Get counts
    datasets_count = await db.fetchval(
        "SELECT COUNT(*) FROM finetune_datasets WHERE user_id = $1",
        user_id
    )

    models_count = await db.fetchval(
        "SELECT COUNT(*) FROM finetune_models WHERE user_id = $1",
        user_id
    )

    jobs_count = await db.fetchval(
        "SELECT COUNT(*) FROM finetune_training_jobs WHERE user_id = $1",
        user_id
    )

    # Training job status breakdown
    status_counts = await db.fetch(
        """
        SELECT status, COUNT(*) as count
        FROM finetune_training_jobs
        WHERE user_id = $1
        GROUP BY status
        """,
        user_id
    )

    status_breakdown = {row["status"]: row["count"] for row in status_counts}
    completed = status_breakdown.get("completed", 0)
    failed = status_breakdown.get("failed", 0)
    running = status_breakdown.get("training", 0) + status_breakdown.get("preparing", 0)

    # Success rate
    total_finished = completed + failed
    success_rate = (completed / total_finished * 100) if total_finished > 0 else None

    # Training methods usage
    method_counts = await db.fetch(
        """
        SELECT training_method, COUNT(*) as count
        FROM finetune_training_jobs
        WHERE user_id = $1
        GROUP BY training_method
        ORDER BY count DESC
        """,
        user_id
    )

    # Recent jobs
    recent_jobs = await db.fetch(
        """
        SELECT
            j.job_id, j.name, j.status, j.training_method,
            j.progress_percentage, j.current_loss, j.created_at,
            j.started_at, j.completed_at, j.execution_env,
            d.name as dataset_name, m.name as model_name
        FROM finetune_training_jobs j
        LEFT JOIN finetune_datasets d ON j.dataset_id = d.dataset_id
        LEFT JOIN finetune_models m ON j.base_model_id = m.model_id
        WHERE j.user_id = $1
        ORDER BY j.created_at DESC
        LIMIT 5
        """,
        user_id
    )

    return {
        "stats": {
            "datasets": datasets_count or 0,
            "models": models_count or 0,
            "training_jobs": jobs_count or 0,
            "success_rate": round(success_rate, 1) if success_rate else None,
            "running_jobs": running,
            "completed_jobs": completed,
            "failed_jobs": failed,
        },
        "status_breakdown": status_breakdown,
        "method_usage": [
            {"method": row["training_method"], "count": row["count"]}
            for row in method_counts
        ],
        "recent_jobs": [
            {
                "job_id": str(job["job_id"]),
                "name": job["name"],
                "status": job["status"],
                "training_method": job["training_method"],
                "execution_env": job["execution_env"],
                "progress": job["progress_percentage"],
                "loss": job["current_loss"],
                "dataset_name": job["dataset_name"],
                "model_name": job["model_name"],
                "created_at": job["created_at"].isoformat() if job["created_at"] else None,
                "started_at": job["started_at"].isoformat() if job["started_at"] else None,
                "completed_at": job["completed_at"].isoformat() if job["completed_at"] else None,
            }
            for job in recent_jobs
        ],
    }


@router.get("/analytics")
async def get_training_analytics(
    days: int = 30,
    current_user=Depends(get_current_user),
):
    """Get training analytics for charts"""
    user_id = current_user.user_id
    start_date = datetime.utcnow() - timedelta(days=days)

    # Jobs per day
    jobs_per_day = await db.fetch(
        """
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM finetune_training_jobs
        WHERE user_id = $1 AND created_at >= $2
        GROUP BY DATE(created_at)
        ORDER BY date
        """,
        user_id,
        start_date
    )

    # Training time per method
    training_time = await db.fetch(
        """
        SELECT
            training_method,
            SUM(EXTRACT(EPOCH FROM (completed_at - started_at))) / 60 as total_minutes
        FROM finetune_training_jobs
        WHERE user_id = $1
            AND status = 'completed'
            AND started_at IS NOT NULL
            AND completed_at IS NOT NULL
        GROUP BY training_method
        """,
        user_id
    )

    # Average loss by method
    avg_loss = await db.fetch(
        """
        SELECT training_method, AVG(current_loss) as avg_loss
        FROM finetune_training_jobs
        WHERE user_id = $1 AND status = 'completed' AND current_loss IS NOT NULL
        GROUP BY training_method
        """,
        user_id
    )

    # Execution environment usage
    env_usage = await db.fetch(
        """
        SELECT execution_env, COUNT(*) as count
        FROM finetune_training_jobs
        WHERE user_id = $1
        GROUP BY execution_env
        """,
        user_id
    )

    return {
        "jobs_per_day": [
            {"date": row["date"].isoformat(), "count": row["count"]}
            for row in jobs_per_day
        ],
        "training_time_by_method": [
            {"method": row["training_method"], "minutes": round(row["total_minutes"] or 0, 1)}
            for row in training_time
        ],
        "avg_loss_by_method": [
            {"method": row["training_method"], "loss": round(row["avg_loss"], 4) if row["avg_loss"] else None}
            for row in avg_loss
        ],
        "env_usage": [
            {"env": row["execution_env"], "count": row["count"]}
            for row in env_usage
        ],
    }
