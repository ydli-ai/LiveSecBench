# encoding: utf-8
"""SD WebUI (AUTOMATIC1111) 文生图 API 适配器"""

from typing import Any, Dict, List

from livesecbench.core.image_generation.base import BaseImageGenAdapter


class SDWebUIAdapter(BaseImageGenAdapter):
    """SD WebUI: POST sdapi/v1/txt2img, 响应 images (base64 列表)"""

    def build_request(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        api_config: Dict[str, Any],
        image_generation: Dict[str, Any],
    ) -> Dict[str, Any]:
        req = {
            "prompt": prompt,
            "negative_prompt": image_generation.get("negative_prompt") or input_data.get("negative_prompt", ""),
            "steps": image_generation.get("steps", 20),
            "cfg_scale": image_generation.get("cfg_scale", 7),
            "width": image_generation.get("width", 512),
            "height": image_generation.get("height", 768),
            "seed": image_generation.get("seed", -1),
        }
        if image_generation.get("sampler_name"):
            req["sampler_name"] = image_generation["sampler_name"]
        return req

    def get_endpoint(self, api_config: Dict[str, Any]) -> str:
        return (api_config.get("end_point") or "sdapi/v1/txt2img").strip().lstrip("/")

    def parse_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        images = response.get("images") or []
        return [{"b64": img} for img in images if isinstance(img, str)]
