"""
存储工厂类

根据配置动态创建不同类型的存储实例
"""
from typing import Optional
from livesecbench.storage.base_storage import BaseStorage
from livesecbench.utils.logger import get_logger

logger = get_logger(__name__)


def create_storage(
    config_manager,
    task_id: Optional[str] = None,
) -> BaseStorage:
    """
    根据配置创建存储实例
    """
    storage_type = config_manager.get_storage_type()
    storage_tables = config_manager.get_storage_tables()
    
    logger.info(f"正在创建存储实例，类型: {storage_type}")
    
    if storage_type == "sqlite":
        from livesecbench.storage.sqlite_storage import SQLiteStorage
        
        db_path = config_manager.get_storage_db_path()
        return SQLiteStorage(
            db_path=db_path,
            model_outputs_table=storage_tables.get('model_outputs_table', 'model_outputs'),
            pk_results_table=storage_tables.get('pk_results_table', 'pk_results'),
            tasks_table=storage_tables.get('tasks_table', 'evaluation_tasks'),
            task_id=task_id,
        )
    
    elif storage_type == "mysql":
        from livesecbench.storage.mysql_storage import MySQLStorage
        
        mysql_config = config_manager.get_mysql_config()
        return MySQLStorage(
            host=mysql_config.get('host', 'localhost'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'livesecbench'),
            charset=mysql_config.get('charset', 'utf8mb4'),
            pool_size=mysql_config.get('pool_size', 10),
            max_overflow=mysql_config.get('max_overflow', 20),
            pool_timeout=mysql_config.get('pool_timeout', 30),
            model_outputs_table=storage_tables.get('model_outputs_table', 'model_outputs'),
            pk_results_table=storage_tables.get('pk_results_table', 'pk_results'),
            tasks_table=storage_tables.get('tasks_table', 'evaluation_tasks'),
            task_id=task_id,
        )
    
    else:
        raise ValueError(
            f"不支持的存储类型: {storage_type}。"
            f"支持的类型: 'sqlite', 'mysql'"
        )


def create_storage_simple(
    storage_type: str = "sqlite",
    db_path: Optional[str] = None,
    mysql_config: Optional[dict] = None,
    task_id: Optional[str] = None,
    **kwargs
) -> BaseStorage:
    """
    简化版存储创建函数（不依赖 ConfigManager）
    """
    if storage_type == "sqlite":
        if not db_path:
            raise ValueError("SQLite 存储需要提供 db_path 参数")
        
        from livesecbench.storage.sqlite_storage import SQLiteStorage
        return SQLiteStorage(
            db_path=db_path,
            task_id=task_id,
            **kwargs
        )
    
    elif storage_type == "mysql":
        if not mysql_config:
            raise ValueError("MySQL 存储需要提供 mysql_config 参数")
        
        from livesecbench.storage.mysql_storage import MySQLStorage
        return MySQLStorage(
            host=mysql_config.get('host', 'localhost'),
            port=mysql_config.get('port', 3306),
            user=mysql_config.get('user', 'root'),
            password=mysql_config.get('password', ''),
            database=mysql_config.get('database', 'livesecbench'),
            charset=mysql_config.get('charset', 'utf8mb4'),
            pool_size=mysql_config.get('pool_size', 10),
            max_overflow=mysql_config.get('max_overflow', 20),
            pool_timeout=mysql_config.get('pool_timeout', 30),
            task_id=task_id,
            **kwargs
        )
    
    else:
        raise ValueError(f"不支持的存储类型: {storage_type}")

