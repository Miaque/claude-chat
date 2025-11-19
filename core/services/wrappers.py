from contextvars import ContextVar

from loguru import logger
from peewee import *
from peewee import InterfaceError as PeeWeeInterfaceError
from peewee import PostgresqlDatabase
from playhouse.db_url import connect, parse
from playhouse.shortcuts import ReconnectMixin

db_state_default = {"closed": None, "conn": None, "ctx": None, "transactions": None}
db_state = ContextVar("db_state", default=db_state_default.copy())


class PeeweeConnectionState(object):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        self._state.get()[name] = value

    def __getattr__(self, name):
        value = self._state.get()[name]
        return value


class CustomReconnectMixin(ReconnectMixin):
    reconnect_errors = (
        # psycopg2
        (OperationalError, "termin"),
        (InterfaceError, "closed"),
        # peewee
        (PeeWeeInterfaceError, "closed"),
    )


class ReconnectingPostgresqlDatabase(CustomReconnectMixin, PostgresqlDatabase):
    pass


def register_connection(db_url):
    # Standard database connection (existing logic)
    db = connect(db_url, unquote_user=True, unquote_password=True)
    if isinstance(db, PostgresqlDatabase):
        # Enable autoconnect for SQLite databases, managed by Peewee
        db.autoconnect = True
        db.reuse_if_open = True
        logger.info("Connected to PostgreSQL database")

        # Get the connection details
        connection = parse(db_url, unquote_user=True, unquote_password=True)

        # Use our custom database class that supports reconnection
        db = ReconnectingPostgresqlDatabase(**connection)
        db.connect(reuse_if_open=True)
    else:
        raise ValueError("Unsupported database connection")

    return db
