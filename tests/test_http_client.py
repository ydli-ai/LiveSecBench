"""
HTTP客户端单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from livesecbench.infra.http_client import RetryableHTTPClient, RateLimiter


class TestRateLimiter:
    """RateLimiter测试类"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_no_limit(self):
        """测试无限制的速率限制器"""
        limiter = RateLimiter(per_second=0, per_minute=0)
        # 应该立即返回，不等待
        await limiter.acquire()
        assert True  # 如果没有抛出异常，测试通过
    
    @pytest.mark.asyncio
    async def test_rate_limiter_per_second(self, monkeypatch):
        """测试每秒限制（通过模拟时间避免真实等待）"""
        limiter = RateLimiter(per_second=2, per_minute=0)
        module_path = "livesecbench.infra.http_client"
        fake_now = 0.0

        def fake_time():
            return fake_now

        async def fake_sleep(duration):
            nonlocal fake_now
            fake_now += duration

        monkeypatch.setattr(f"{module_path}.time.time", fake_time)
        monkeypatch.setattr(f"{module_path}.asyncio.sleep", fake_sleep)

        # 前两次应该立即通过
        await limiter.acquire()
        await limiter.acquire()

        # 第三次需要等待，模拟时间推进
        await limiter.acquire()

        assert fake_now >= 1.0  # 确认触发了等待逻辑


class TestRetryableHTTPClient:
    """RetryableHTTPClient测试类"""
    
    @pytest.fixture
    def mock_client(self):
        """创建模拟的HTTP客户端"""
        client = RetryableHTTPClient(
            base_url="https://api.test.com",
            api_key="test_key",
            timeout=10,
            max_retries=3,
            retry_delay=0.1
        )
        return client
    
    @pytest.mark.asyncio
    async def test_post_success(self, mock_client):
        """测试成功的POST请求"""
        mock_response = {
            'choices': [{'message': {'content': 'test response'}}]
        }
        
        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.text = '{"choices": [{"message": {"content": "test response"}}]}'
        
        mock_httpx_client = MagicMock()
        mock_httpx_client.post = AsyncMock(return_value=mock_response_obj)
        mock_httpx_client.aclose = AsyncMock()
        mock_client._client = mock_httpx_client
        
        result = await mock_client.post(
            endpoint="test",
            json_data={"test": "data"},
            context_name="测试"
        )
        
        assert 'choices' in result
        assert mock_httpx_client.post.await_count == 1
    
    @pytest.mark.asyncio
    async def test_post_retry_on_429(self, mock_client):
        """测试429错误时的重试"""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.text = '{"error": "rate limit"}'
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.text = '{"choices": [{"message": {"content": "success"}}]}'
        
        mock_httpx_client = MagicMock()
        mock_httpx_client.post = AsyncMock(side_effect=[mock_response_429, mock_response_200])
        mock_httpx_client.aclose = AsyncMock()
        mock_client._client = mock_httpx_client
        
        result = await mock_client.post(
            endpoint="test",
            json_data={"test": "data"},
            context_name="测试"
        )
        
        assert 'choices' in result
        assert mock_httpx_client.post.await_count == 2
    
    @pytest.mark.asyncio
    async def test_post_max_retries_exceeded(self, mock_client):
        """测试超过最大重试次数"""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = '{"error": "rate limit"}'
        
        mock_httpx_client = MagicMock()
        mock_httpx_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client.aclose = AsyncMock()
        mock_client._client = mock_httpx_client
        
        with pytest.raises(RuntimeError):
            await mock_client.post(
                endpoint="test",
                json_data={"test": "data"},
                context_name="测试"
            )
