"""
Database connection management using asyncpg
Following the pattern from AngelaAI
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg

from .config import settings

logger = logging.getLogger(__name__)


class Database:
    """Async PostgreSQL database connection manager"""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None
        self._connect_lock = asyncio.Lock()

    async def connect(self, max_retries: int = 5, initial_wait: float = 2.0) -> None:
        """
        Create connection pool with retry logic
        """
        if self.pool is not None:
            return

        async with self._connect_lock:
            if self.pool is not None:
                return

            wait_time = initial_wait
            for attempt in range(max_retries):
                try:
                    self.pool = await asyncpg.create_pool(
                        self.database_url,
                        min_size=settings.DB_POOL_MIN_SIZE,
                        max_size=settings.DB_POOL_MAX_SIZE,
                        command_timeout=settings.DB_COMMAND_TIMEOUT,
                    )
                    logger.info("Database connection pool created successfully")
                    return
                except Exception as e:
                    logger.warning(
                        f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        wait_time *= 2  # Exponential backoff
                    else:
                        raise ConnectionError(
                            f"Failed to connect to database after {max_retries} attempts"
                        ) from e

    async def disconnect(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args) -> str:
        """Execute a query (INSERT, UPDATE, DELETE)"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def executemany(self, query: str, args: list) -> None:
        """Execute a query with multiple parameter sets"""
        async with self.acquire() as conn:
            await conn.executemany(query, args)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# Global database instance
db = Database()


async def get_db() -> Database:
    """Dependency for FastAPI endpoints"""
    if not db.pool:
        await db.connect()
    return db


async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool (for repositories)"""
    if not db.pool:
        await db.connect()
    return db.pool


async def init_db() -> None:
    """Initialize database (run migrations)"""
    import os

    migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
    migration_file = os.path.join(migrations_dir, "001_initial_schema.sql")

    if os.path.exists(migration_file):
        async with db.acquire() as conn:
            with open(migration_file, "r") as f:
                sql = f.read()
            try:
                await conn.execute(sql)
                logger.info("Database migrations applied successfully")
            except asyncpg.exceptions.DuplicateObjectError:
                logger.info("Database tables already exist")
            except Exception as e:
                logger.error(f"Error applying migrations: {e}")
                raise
    else:
        logger.warning(f"Migration file not found: {migration_file}")
