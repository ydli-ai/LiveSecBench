# encoding: utf-8
"""文生图多后端适配器：SiliconFlow、SD WebUI、ComfyUI"""

from livesecbench.core.image_generation.base import BaseImageGenAdapter
from livesecbench.core.image_generation.siliconflow_adapter import SiliconFlowAdapter
from livesecbench.core.image_generation.sd_webui_adapter import SDWebUIAdapter
from livesecbench.core.image_generation.comfyui_adapter import ComfyUIAdapter

__all__ = [
    "BaseImageGenAdapter",
    "SiliconFlowAdapter",
    "SDWebUIAdapter",
    "ComfyUIAdapter",
    "get_adapter",
]


def get_adapter(api_provider: str) -> BaseImageGenAdapter:
    """根据 api_provider 返回对应适配器实例"""
    provider = (api_provider or "").strip().lower()
    if provider == "siliconflow":
        return SiliconFlowAdapter()
    if provider == "sd_webui":
        return SDWebUIAdapter()
    if provider == "comfyui":
        return ComfyUIAdapter()
    raise ValueError(f"不支持的文生图 api_provider: {api_provider}")
