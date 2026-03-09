# encoding: utf-8
"""文生图 API 适配器基类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseImageGenAdapter(ABC):
    """文生图后端适配器抽象基类"""

    @abstractmethod
    def build_request(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        api_config: Dict[str, Any],
        image_generation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造该后端的请求体。"""
        pass

    @abstractmethod
    def get_endpoint(self, api_config: Dict[str, Any]) -> str:
        """返回请求路径（相对 base_url）。"""
        pass

    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析响应，返回统一结构的图片列表。
        每项至少包含: b64 (base64 字符串) 或 url 或 path。
        若有 b64，后续由调用方落盘。
        """
        pass
