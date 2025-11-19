import json
from contextlib import contextmanager
from typing import Any, Optional

from loguru import logger
from peewee_migrate import Router
from sqlalchemy import Dialect, MetaData, create_engine, types
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.sql.type_api import _T
from typing_extensions import Self

from configs import BACKEND_DIR, app_config
from core.services.wrappers import register_connection


class JSONField(types.TypeDecorator):
    impl = types.Text
    cache_ok = True

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        return json.dumps(value)

    def process_result_value(self, value: Optional[_T], dialect: Dialect) -> Any:
        if value is not None:
            return json.loads(value)

    def copy(self, **kw: Any) -> Self:
        return JSONField(self.impl.length)

    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


# Workaround to handle the peewee migration
# This is required to ensure the peewee migration is handled before the alembic migration
def handle_peewee_migration(DATABASE_URL):
    # db = None
    try:
        # Replace the postgresql:// with postgres:// to handle the peewee migration
        db = register_connection(DATABASE_URL.replace("postgresql://", "postgres://"))
        migrate_dir = BACKEND_DIR / "migrations"
        router = Router(db, logger=logger, migrate_dir=migrate_dir)
        router.run()
        db.close()

    except Exception as e:
        logger.error(f"Failed to initialize the database connection: {e}")
        logger.warning(
            "Hint: If your database password contains special characters, you may need to URL-encode it."
        )
        raise
    finally:
        # Properly closing the database connection
        if db and not db.is_closed():
            db.close()

        # Assert if db connection has been closed
        assert db.is_closed(), "Database connection is still open."


handle_peewee_migration(app_config.SQLALCHEMY_DATABASE_URI)


SQLALCHEMY_DATABASE_URL = app_config.SQLALCHEMY_DATABASE_URI


if isinstance(app_config.SQLALCHEMY_POOL_SIZE, int):
    if app_config.SQLALCHEMY_POOL_SIZE > 0:
        engine = create_engine(
            SQLALCHEMY_DATABASE_URL,
            pool_size=app_config.SQLALCHEMY_POOL_SIZE,
            max_overflow=app_config.SQLALCHEMY_MAX_OVERFLOW,
            pool_timeout=app_config.SQLALCHEMY_POOL_TIMEOUT,
            pool_recycle=app_config.SQLALCHEMY_POOL_RECYCLE,
            pool_pre_ping=True,
            poolclass=QueuePool,
        )
    else:
        engine = create_engine(
            SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, poolclass=NullPool
        )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)


SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)
metadata_obj = MetaData(None)
Base = declarative_base(metadata=metadata_obj)
Session = scoped_session(SessionLocal)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


get_db = contextmanager(get_session)
