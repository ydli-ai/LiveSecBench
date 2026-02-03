# encoding: utf-8
"""SiliconFlow 文生图 API 适配器"""

from typing import Any, Dict, List

from livesecbench.core.image_generation.base import BaseImageGenAdapter


class SiliconFlowAdapter(BaseImageGenAdapter):
    """SiliconFlow: POST images/generations, 响应 data[].b64_json 或 data[].url"""

    def build_request(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        api_config: Dict[str, Any],
        image_generation: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_id = api_config.get("model_id", "")
        req = {
            "model": model_id,
            "prompt": prompt,
            "image_size": image_generation.get("image_size", "1024x1024"),
            "batch_size": image_generation.get("batch_size", 1),
            "num_inference_steps": image_generation.get("num_inference_steps", 20),
            "guidance_scale": image_generation.get("guidance_scale", 7.5),
        }
        return req

    def get_endpoint(self, api_config: Dict[str, Any]) -> str:
        return (api_config.get("end_point") or "images/generations").strip().lstrip("/")

    def parse_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = []
        data = response.get("data") or []
        for item in data:
            out = {}
            if item.get("b64_json"):
                out["b64"] = item["b64_json"]
            if item.get("url"):
                out["url"] = item["url"]
            if out:
                items.append(out)
        return items
