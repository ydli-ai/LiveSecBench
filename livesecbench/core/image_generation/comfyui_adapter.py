# encoding: utf-8
"""ComfyUI 文生图 API 适配器：提交 workflow -> 轮询 history -> /view 取图"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from livesecbench.core.image_generation.base import BaseImageGenAdapter


class ComfyUIAdapter(BaseImageGenAdapter):
    """
    ComfyUI: POST /prompt 提交 workflow，轮询 /history/{prompt_id} 取结果，
    再通过 /view?filename=... 取图。需配置 workflow 模板，占位符 {prompt} 替换为提示词。
    """

    def build_request(
        self,
        prompt: str,
        input_data: Dict[str, Any],
        api_config: Dict[str, Any],
        image_generation: Dict[str, Any],
    ) -> Dict[str, Any]:
        workflow = self._load_workflow(api_config, image_generation)
        if isinstance(workflow, dict):
            workflow = self._inject_prompt(workflow, prompt)
        return {
            "prompt": workflow,
            "client_id": str(uuid.uuid4()),
        }

    def _load_workflow(
        self, api_config: Dict[str, Any], image_generation: Dict[str, Any]
    ) -> Dict[str, Any]:
        inline = image_generation.get("workflow_inline") or api_config.get("workflow_inline")
        if inline is not None:
            return inline if isinstance(inline, dict) else json.loads(inline)
        path = image_generation.get("workflow_template_path") or api_config.get("workflow_template_path")
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent.parent.parent / p
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        return self._default_workflow()

    def _inject_prompt(self, workflow: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """在 workflow 的 inputs 中替换占位符 {prompt}"""
        for node_id, node in workflow.get("nodes", {}).items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("widgets_values") or node.get("inputs") or []
            if isinstance(inputs, list):
                workflow["nodes"][node_id] = dict(node)
                workflow["nodes"][node_id]["widgets_values"] = [
                    (prompt if (isinstance(v, str) and v.strip() == "{prompt}") else v) for v in inputs
                ]
            elif isinstance(inputs, dict) and "text" in inputs:
                workflow["nodes"][node_id] = dict(node)
                workflow["nodes"][node_id]["inputs"] = dict(inputs)
                workflow["nodes"][node_id]["inputs"]["text"] = prompt
        return workflow

    def _default_workflow(self) -> Dict[str, Any]:
        """简易 txt2img workflow 骨架，实际使用建议提供完整 workflow 文件"""
        return {
            "nodes": {},
            "links": [],
        }

    def get_endpoint(self, api_config: Dict[str, Any]) -> str:
        return (api_config.get("end_point") or "prompt").strip().lstrip("/")

    def parse_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提交后的响应仅含 prompt_id，不直接含图；由调用方轮询 history 再取图。"""
        return []

    def is_async(self) -> bool:
        return True

    def get_history_endpoint(self, prompt_id: str) -> str:
        return f"history/{prompt_id}"

    def parse_history_response(self, history: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
        """从 /history/{prompt_id} 响应中解析出待下载的图片信息。"""
        outputs = history.get(prompt_id, {}).get("outputs") or {}
        images = []
        for node_out in outputs.values():
            for img in (node_out.get("images") or []):
                if isinstance(img, dict) and img.get("filename"):
                    images.append({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
        return images

    def get_view_params(self, filename: str, subfolder: str = "", type_: str = "output") -> Dict[str, str]:
        """返回 /view 的查询参数字典。"""
        params = {"filename": filename}
        if subfolder:
            params["subfolder"] = subfolder
        if type_:
            params["type"] = type_
        return params
