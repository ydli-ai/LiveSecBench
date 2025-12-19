"""
存储模块

提供统一的数据存储接口，支持多种数据库后端（SQLite、MySQL等）
"""

from livesecbench.storage.base_storage import BaseStorage
from livesecbench.storage.sqlite_storage import SQLiteStorage
from livesecbench.storage.storage_factory import create_storage, create_storage_simple

try:
    from livesecbench.storage.mysql_storage import MySQLStorage
    __all__ = [
        'BaseStorage',
        'SQLiteStorage',
        'MySQLStorage',
        'create_storage',
        'create_storage_simple',
    ]
except ImportError:
    __all__ = [
        'BaseStorage',
        'SQLiteStorage',
        'create_storage',
        'create_storage_simple',
    ]

