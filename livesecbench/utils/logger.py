"""
全局日志配置模块
提供统一的日志配置和管理功能
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "livesecbench" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

_log_configured = False


def setup_logger(
    name: Optional[str] = None,
    level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file_name: Optional[str] = None,
    encoding: str = 'utf-8'
) -> logging.Logger:
    """设置并返回一个配置好的日志记录器"""
    global _log_configured
    
    logger = logging.getLogger(name) if name else logging.getLogger()
    
    if logger.handlers:
        return logger
    
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_to_file:
        if log_file_name is None:
            today = time.strftime('%Y_%m_%d', time.localtime())
            log_file_name = f"{today}.log"
        
        log_file_path = LOGS_DIR / log_file_name
        file_handler = logging.FileHandler(log_file_path, encoding=encoding, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器（如果未配置则使用默认配置初始化）"""
    logger = logging.getLogger(name) if name else logging.getLogger()
    
    if not logger.handlers:
        return setup_logger(name=name)
    
    return logger


def configure_root_logger(
    level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file_name: Optional[str] = None
) -> None:
    """配置根日志记录器（应在程序启动时调用一次）"""
    global _log_configured
    
    if _log_configured:
        return
    
    setup_logger(
        name=None,
        level=level,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
        log_file_name=log_file_name
    )
    
    _log_configured = True

