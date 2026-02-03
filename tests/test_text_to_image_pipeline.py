# -*- coding: utf-8 -*-
"""文生图流水线：适配器请求体、响应解析与落盘路径。"""

import base64
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from livesecbench.core.image_generation import get_adapter
from livesecbench.core.image_generation.siliconflow_adapter import SiliconFlowAdapter
from livesecbench.core.image_generation.sd_webui_adapter import SDWebUIAdapter
from livesecbench.core.image_generation.comfyui_adapter import ComfyUIAdapter


class TestGetAdapter:
    def test_siliconflow(self):
        adapter = get_adapter("siliconflow")
        assert isinstance(adapter, SiliconFlowAdapter)

    def test_sd_webui(self):
        adapter = get_adapter("sd_webui")
        assert isinstance(adapter, SDWebUIAdapter)

    def test_comfyui(self):
        adapter = get_adapter("comfyui")
        assert isinstance(adapter, ComfyUIAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="不支持的文生图 api_provider"):
            get_adapter("unknown")


class TestSiliconFlowAdapter:
    def test_build_request(self):
        adapter = SiliconFlowAdapter()
        api_config = {"model_id": "Kwai-Kolors/Kolors"}
        image_generation = {"image_size": "1024x1024", "batch_size": 1, "num_inference_steps": 20, "guidance_scale": 7.5}
        req = adapter.build_request("a beach", {}, api_config, image_generation)
        assert req["model"] == "Kwai-Kolors/Kolors"
        assert req["prompt"] == "a beach"
        assert req["image_size"] == "1024x1024"
        assert req["batch_size"] == 1
        assert req["num_inference_steps"] == 20
        assert req["guidance_scale"] == 7.5

    def test_get_endpoint(self):
        adapter = SiliconFlowAdapter()
        assert adapter.get_endpoint({"end_point": "images/generations"}) == "images/generations"

    def test_parse_response_data_b64(self):
        adapter = SiliconFlowAdapter()
        b64 = base64.b64encode(b"fake-png").decode("utf-8")
        out = adapter.parse_response({"data": [{"b64_json": b64}]})
        assert len(out) == 1
        assert out[0]["b64"] == b64

    def test_parse_response_data_url(self):
        adapter = SiliconFlowAdapter()
        out = adapter.parse_response({"data": [{"url": "https://example.com/img.png"}]})
        assert len(out) == 1
        assert out[0]["url"] == "https://example.com/img.png"


class TestSDWebUIAdapter:
    def test_build_request(self):
        adapter = SDWebUIAdapter()
        image_generation = {"width": 512, "height": 768, "steps": 20, "cfg_scale": 7}
        req = adapter.build_request("a cat", {}, {}, image_generation)
        assert req["prompt"] == "a cat"
        assert req["width"] == 512
        assert req["height"] == 768
        assert req["steps"] == 20
        assert req["cfg_scale"] == 7

    def test_get_endpoint(self):
        adapter = SDWebUIAdapter()
        assert adapter.get_endpoint({"end_point": "sdapi/v1/txt2img"}) == "sdapi/v1/txt2img"

    def test_parse_response_images(self):
        adapter = SDWebUIAdapter()
        b64 = base64.b64encode(b"x").decode("utf-8")
        out = adapter.parse_response({"images": [b64]})
        assert len(out) == 1
        assert out[0]["b64"] == b64


class TestComfyUIAdapter:
    def test_build_request_client_id(self):
        adapter = ComfyUIAdapter()
        req = adapter.build_request("a dog", {}, {}, {})
        assert "prompt" in req
        assert "client_id" in req
        assert len(req["client_id"]) > 0

    def test_get_endpoint(self):
        adapter = ComfyUIAdapter()
        assert adapter.get_endpoint({"end_point": "prompt"}) == "prompt"

    def test_is_async(self):
        adapter = ComfyUIAdapter()
        assert adapter.is_async() is True

    def test_parse_history_response(self):
        adapter = ComfyUIAdapter()
        hist = {"pid1": {"outputs": {"node_1": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}}
        imgs = adapter.parse_history_response(hist, "pid1")
        assert len(imgs) == 1
        assert imgs[0]["filename"] == "out.png"

    def test_get_view_params(self):
        adapter = ComfyUIAdapter()
        params = adapter.get_view_params("out.png", "sub", "output")
        assert params["filename"] == "out.png"
        assert params["subfolder"] == "sub"
        assert params["type"] == "output"


class TestRunTextToImageArtifactsPath:
    """落盘路径：artifacts / task_id / model_id / question_id / 0.png"""

    @pytest.mark.asyncio
    async def test_artifacts_path_structure(self):
        from livesecbench.core.run_text_to_image import run_single_text_to_image_call, _prompt_hash
        artifacts_base = Path("/tmp/livesecbench_artifacts_test")
        artifacts_base.mkdir(parents=True, exist_ok=True)
        try:
            http_client = MagicMock()
            http_client.post = AsyncMock(return_value={"data": [{"b64_json": base64.b64encode(b"\x89PNG").decode("utf-8")}]})
            api_config = {"api_provider": "siliconflow", "model_id": "test-model", "end_point": "images/generations"}
            image_generation = {"image_size": "1024x1024", "batch_size": 1}
            model_item = {"model_name": "Test", "model": "test-model", "api_config": api_config, "image_generation": image_generation}
            input_data = {"question_text": "a tree", "dimension": "text_to_image", "question_id": "q1"}
            semaphore = __import__("asyncio").Semaphore(2)
            result = await run_single_text_to_image_call(
                http_client=http_client,
                semaphore=semaphore,
                model_name="Test",
                model_id="test-model",
                input_data=input_data,
                api_config=api_config,
                image_generation=image_generation,
                artifacts_base=artifacts_base,
                task_id="task1",
            )
            assert result.get("status") == "success"
            assert result.get("image_outputs")
            out_dir = artifacts_base / "task1" / "test-model" / "q1"
            assert out_dir.exists()
            assert any(out_dir.glob("*.png"))
        finally:
            import shutil
            if artifacts_base.exists():
                shutil.rmtree(artifacts_base, ignore_errors=True)
