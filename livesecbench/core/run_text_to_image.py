# encoding: utf-8
"""文生图单题调用：按 api_provider 选适配器，请求生成，落盘，返回统一 payload 供 caption 与存储。"""

import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from livesecbench.core.image_generation import get_adapter
from livesecbench.core.image_generation.comfyui_adapter import ComfyUIAdapter
from livesecbench.infra.http_client import RetryableHTTPClient
from livesecbench.utils.logger import get_logger

logger = get_logger(__name__)


async def run_single_text_to_image_call(
    http_client: RetryableHTTPClient,
    semaphore: asyncio.Semaphore,
    model_name: str,
    model_id: str,
    input_data: Dict[str, Any],
    api_config: Dict[str, Any],
    image_generation: Dict[str, Any],
    artifacts_base: Path,
    task_id: Optional[str] = None,
    identifier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    单题文生图：构造请求 -> 调用 API -> 解析响应 -> 落盘 -> 返回统一结构。
    返回含 status, image_outputs (path/list), prompt, category, consume_time 等，供后续 caption 与存储。
    """
    prompt = (input_data.get("question_text") or input_data.get("prompt") or "").strip()
    category = input_data.get("dimension", "text_to_image")
    question_id = input_data.get("question_id", "")
    risk_tags = input_data.get("risk_tags") or []
    if isinstance(risk_tags, list):
        risk_tags = ", ".join(str(t) for t in risk_tags)

    provider = (api_config.get("api_provider") or "siliconflow").strip().lower()
    adapter = get_adapter(provider)
    endpoint = adapter.get_endpoint(api_config)
    req_body = adapter.build_request(prompt, input_data, api_config, image_generation)

    start = time.time()
    try:
        async with semaphore:
            if getattr(adapter, "is_async", lambda: False)() and isinstance(adapter, ComfyUIAdapter):
                image_items = await _run_comfyui_flow(
                    http_client, adapter, api_config, req_body, endpoint, identifier
                )
            else:
                raw = await http_client.post(
                    endpoint=endpoint,
                    json_data=req_body,
                    context_name=f"文生图 {model_name}",
                    task_type="answer",
                    identifier=identifier or {"question_id": question_id, "dimension": category},
                )
                image_items = adapter.parse_response(raw)
    except Exception as e:
        logger.error(f"文生图请求失败 {model_name}: {e}")
        return {
            "status": "error",
            "model_name": model_name,
            "model": model_id,
            "category": category,
            "prompt": prompt,
            "error": str(e),
            "consume_time": time.time() - start,
            "image_outputs": [],
        }

    consume_time = time.time() - start
    if not image_items:
        return {
            "status": "error",
            "model_name": model_name,
            "model": model_id,
            "category": category,
            "prompt": prompt,
            "error": "响应中无图片",
            "consume_time": consume_time,
            "image_outputs": [],
        }

    # 落盘：artifacts_base / task_id / model_id / question_id / 0.png, 1.png, ...
    out_dir = artifacts_base
    if task_id:
        out_dir = out_dir / str(task_id)
    out_dir = out_dir / (model_id or "unknown") / (question_id or "unknown")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, item in enumerate(image_items):
        b64 = item.get("b64")
        url = item.get("url")
        path = item.get("path")
        if path and Path(path).exists():
            saved.append({"path": str(Path(path).resolve()), "index": i})
            continue
        if b64:
            raw_bytes = base64.b64decode(b64)
            ext = "png"
            fname = f"{i}.{ext}"
            out_path = out_dir / fname
            out_path.write_bytes(raw_bytes)
            saved.append({"path": str(out_path), "index": i})
            continue
        if url:
            saved.append({"url": url, "path": "", "index": i})

    # 统一 payload，供 caption 与 storage
    payload = {
        "status": "success",
        "model_name": model_name,
        "model": model_id,
        "category": category,
        "prompt": prompt,
        "prompt_hash": _prompt_hash(prompt, question_id, saved),
        "consume_time": consume_time,
        "image_outputs": saved,
        "question_id": question_id,
        "risk_tags": risk_tags,
        "input_data": input_data,
    }
    return payload


def _prompt_hash(prompt: str, question_id: str, image_outputs: List[Dict]) -> str:
    raw = f"{question_id}|{prompt}|{len(image_outputs)}"
    for o in image_outputs:
        raw += "|" + (o.get("path") or o.get("url") or "")
    return hashlib.md5(raw.encode()).hexdigest()


async def _run_comfyui_flow(
    http_client: RetryableHTTPClient,
    adapter: ComfyUIAdapter,
    api_config: Dict[str, Any],
    req_body: Dict[str, Any],
    endpoint: str,
    identifier: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """ComfyUI: POST /prompt -> 轮询 /history/{prompt_id} -> GET /view 取图 -> 转 b64 列表。"""
    raw = await http_client.post(
        endpoint=endpoint,
        json_data=req_body,
        context_name="ComfyUI 文生图",
        task_type="answer",
        identifier=identifier,
    )
    prompt_id = raw.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI 未返回 prompt_id")
    hist_endpoint = adapter.get_history_endpoint(prompt_id)
    max_wait = 300
    step = 2.0
    elapsed = 0
    while elapsed < max_wait:
        text, _ = await http_client.get(hist_endpoint)
        hist = json.loads(text) if isinstance(text, str) else text
        if hist.get(prompt_id):
            break
        await asyncio.sleep(step)
        elapsed += step
    if not hist.get(prompt_id):
        raise RuntimeError("ComfyUI history 超时未返回结果")
    image_infos = adapter.parse_history_response(hist, prompt_id)
    if not image_infos:
        return []
    results = []
    for info in image_infos:
        params = adapter.get_view_params(
            info["filename"],
            info.get("subfolder", ""),
            info.get("type", "output"),
        )
        _, content = await http_client.get("view", params=params)
        b64 = base64.b64encode(content).decode("utf-8")
        results.append({"b64": b64})
    return results
