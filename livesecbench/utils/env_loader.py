"""统一的环境变量加载工具"""

from pathlib import Path
from dotenv import load_dotenv
from livesecbench.utils.logger import get_logger

logger = get_logger(__name__)


def load_project_env() -> None:
    """加载项目根目录的.env文件"""
    project_root = Path(__file__).resolve().parent.parent.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path, override=True)
        logger.info(f"已加载环境变量文件: {env_path}")
    else:
        load_dotenv(env_path, override=False)
        logger.warning(f"未找到环境变量文件: {env_path}，将使用系统环境变量")

