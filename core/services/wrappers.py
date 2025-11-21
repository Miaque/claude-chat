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
    db = connect(db_url, unquote_user=True, unquote_password=True)
    if isinstance(db, PostgresqlDatabase):
        db.autoconnect = True
        db.reuse_if_open = True
        logger.info("已连接到 PostgreSQL 数据库")

        # 获取连接详情
        connection = parse(db_url, unquote_user=True, unquote_password=True)

        # 使用支持重连的自定义数据库类
        db = ReconnectingPostgresqlDatabase(**connection)
        db.connect(reuse_if_open=True)
    else:
        raise ValueError("不支持的数据库连接类型")

    return db
