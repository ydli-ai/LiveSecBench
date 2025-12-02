"""
统一的HTTP客户端封装
提供带重试机制、错误处理的HTTP请求功能
"""

import asyncio
import atexit
import json
import time
from collections import deque
from typing import Dict, Optional, Any

import httpx
from json_repair import repair_json

from livesecbench.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """速率限制器，支持每秒请求数、每分钟请求数和每分钟Token数（TPM）限制
    
    TPM限流采用预估策略：
    - acquire 时预先占用估算的 token 额度
    - register_tokens 时用实际消耗替换预估值
    """
    
    def __init__(self, per_second: int = 0, per_minute: int = 0, tokens_per_minute: int = 0, estimated_tokens_per_request: int = 4000):
        self.per_second = per_second
        self.per_minute = per_minute
        self.tokens_per_minute = tokens_per_minute
        self.estimated_tokens_per_request = estimated_tokens_per_request
        self.second_timestamps = deque()
        self.minute_timestamps = deque()
        self.token_timestamps = deque()  # (timestamp, tokens, is_estimated)
        self.token_sum = 0
        self.lock = asyncio.Lock()
        self.pending_estimates = 0

    async def acquire(self) -> None:
        """获取令牌，如果需要等待则在锁外等待，避免阻塞其他协程
        
        对于TPM限流，采用预估策略：
        - 预先占用 estimated_tokens_per_request 的额度
        - 防止高并发时瞬间超过限制
        """
        if self.per_second == 0 and self.per_minute == 0 and self.tokens_per_minute == 0:
            return

        while True:
            async with self.lock:
                now = time.time()

                while self.second_timestamps and now - self.second_timestamps[0] >= 1.0:
                    self.second_timestamps.popleft()
                while self.minute_timestamps and now - self.minute_timestamps[0] >= 60.0:
                    self.minute_timestamps.popleft()
                while self.token_timestamps and now - self.token_timestamps[0][0] >= 60.0:
                    ts, tokens, is_estimated = self.token_timestamps.popleft()
                    self.token_sum -= tokens
                    if is_estimated:
                        self.pending_estimates -= 1

                wait_time = 0.0
                if 0 < self.per_second <= len(self.second_timestamps):
                    wait_time = max(wait_time, 1.0 - (now - self.second_timestamps[0]))
                if 0 < self.per_minute <= len(self.minute_timestamps):
                    wait_time = max(wait_time, 60.0 - (now - self.minute_timestamps[0]))
                if self.tokens_per_minute > 0:
                    estimated_total = self.token_sum + self.estimated_tokens_per_request
                    if estimated_total > self.tokens_per_minute and self.token_timestamps:
                        oldest_ts, _, _ = self.token_timestamps[0]
                        wait_time = max(wait_time, 60.0 - (now - oldest_ts))

                if wait_time == 0:
                    if self.per_second > 0:
                        self.second_timestamps.append(now)
                    if self.per_minute > 0:
                        self.minute_timestamps.append(now)
                    
                    if self.tokens_per_minute > 0:
                        self.token_timestamps.append((now, self.estimated_tokens_per_request, True))
                        self.token_sum += self.estimated_tokens_per_request
                        self.pending_estimates += 1
                    
                    return

            await asyncio.sleep(wait_time)

    async def register_tokens(self, tokens: int) -> None:
        """在请求完成后登记本次消耗的tokens，用于TPM限流"""
        if self.tokens_per_minute <= 0:
            return
        
        async with self.lock:
            now = time.time()
            while self.token_timestamps and now - self.token_timestamps[0][0] >= 60.0:
                ts, old_tokens, is_estimated = self.token_timestamps.popleft()
                self.token_sum -= old_tokens
                if is_estimated:
                    self.pending_estimates -= 1
            
            if self.pending_estimates > 0:
                for i in range(len(self.token_timestamps) - 1, -1, -1):
                    ts, old_tokens, is_estimated = self.token_timestamps[i]
                    if is_estimated:
                        self.token_sum -= old_tokens
                        self.token_sum += int(tokens)
                        self.token_timestamps[i] = (ts, int(tokens), False)
                        self.pending_estimates -= 1
                        break
            else:
                if tokens > 0:
                    self.token_timestamps.append((now, int(tokens), False))
                    self.token_sum += int(tokens)


class RetryableHTTPClient:
    """带重试机制的HTTP客户端: 支持自动重试、限流处理、速率限制、JSON修复"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 210,
        max_retries: int = 5,
        retry_delay: int = 1,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = rate_limiter
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._timeout = httpx.Timeout(self.timeout, connect=30.0)
        self._atexit_registered = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is not None:
                return self._client

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                http2=True,
            )
            return self._client

    async def aclose(self) -> None:
        """在异步上下文关闭底层HTTP客户端"""
        client = self._client
        if client is None:
            return
        self._client = None
        await client.aclose()
        
    def _extract_error_info(self, output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从响应中提取错误信息，支持多种错误格式"""
        error_info = None
        if isinstance(output, dict):
            if 'error' in output and output['error']:
                error_info = output['error']
            elif 'errors' in output and output['errors']:
                error_info = output['errors']
        
        if not error_info:
            return None
        
        error_code = ''
        error_message = error_info
        
        if isinstance(error_info, dict):
            error_code = str(error_info.get('code') or error_info.get('status') or '')
            error_message = error_info.get('message') or error_info
        elif isinstance(error_info, list) and error_info:
            first_error = error_info[0]
            if isinstance(first_error, dict):
                error_code = str(first_error.get('code') or first_error.get('status') or '')
                error_message = first_error.get('message') or first_error
            else:
                error_message = first_error
        
        return {'code': error_code, 'message': error_message}
    
    async def post(
        self,
        endpoint: str,
        json_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        context_name: str = "请求",
    ) -> Dict[str, Any]:
        """发送POST请求（带重试机制）"""
        client = await self._get_client()
        url = endpoint.lstrip('/')

        default_headers = {
            "Authorization": f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        if headers:
            default_headers.update(headers)
        
        for attempt in range(self.max_retries):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                
                response = await client.post(
                    url=url,
                    headers=default_headers,
                    json=json_data,
                )
                status_code = response.status_code
                
                if status_code == 429:
                    wait_time = self.retry_delay * (2 ** (attempt + 1))
                    logger.warning(
                        f"{context_name}遭遇限流(429)，第 {attempt + 1}/{self.max_retries} 次重试，"
                        f"将在 {wait_time:.2f}s 后再次尝试。"
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"{context_name}达到频率限制的最大重试次数。")
                        raise RuntimeError(f"{context_name}达到频率限制的最大重试次数")
                
                if status_code == 400:
                    error_text = response.text
                    logger.error(f"{context_name}请求参数错误(400)，直接退出，不重试。body={error_text}")
                    output = json.loads(error_text, strict=False)
                    if 'error' in output and 'code' in output['error']:
                        error_code = output['error']['code']
                        if error_code and error_code in ["data_inspection_failed", 18, "no_response_from_channel"]:
                            return output['error']
                    response.raise_for_status()
                
                if 401 <= status_code < 500:
                    error_text = response.text
                    logger.error(f"{context_name}请求失败 status={status_code}, body={error_text}")
                    output = json.loads(error_text, strict=False)
                    if 'error' in output and 'code' in output['error']:
                        error_code = output['error']['code']
                        if error_code and error_code == 403:
                            return output['error']
                    if 'error' in output and 'type' in output['error']:
                        error_type = output['error']['type']
                        if error_type and error_type == "censorship_blocked":
                            return output['error']
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"等待 {wait_time:.2f}s 后重试{context_name}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                if status_code >= 500:
                    error_text = response.text
                    logger.warning(f"{context_name}服务器错误 status={status_code}, body={error_text}")
                    output = json.loads(error_text, strict=False)
                    if 'error' in output and 'code' in output['error']:
                        error_code = output['error']['code']
                        if error_code and error_code == "10013":
                            return output['error']
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"等待 {wait_time:.2f}s 后重试{context_name}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                # 解析响应，优先直接解析，失败时再尝试JSON修复
                output_content = response.text
                content_to_parse = output_content
                output: Dict[str, Any]
                if not content_to_parse.strip():
                    content_to_parse = '{}'
                try:
                    output = json.loads(content_to_parse, strict=False)
                except json.JSONDecodeError:
                    repaired = repair_json(output_content, ensure_ascii=False)
                    if not repaired.strip():
                        repaired = '{}'
                    output = json.loads(repaired, strict=False)
                
                error_info = self._extract_error_info(output)
                if error_info:
                    error_code = error_info['code']
                    error_message = error_info['message']
                    
                    if error_code == '400' or error_code == 400:
                        logger.error(
                            f"{context_name}返回错误码400（请求参数错误），直接退出，不重试。"
                            f"错误信息: {error_message}"
                        )
                        raise RuntimeError(f"{context_name}返回400错误：{error_message}")
                    
                    if error_code == '429' or error_code == 429:
                        wait_time = self.retry_delay * (2 ** (attempt + 1))
                        logger.warning(
                            f"{context_name}返回错误码429，准备降频重试，等待 {wait_time:.2f}s。"
                            f"错误信息: {error_message}"
                        )
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError(f"{context_name}返回429错误：{error_message}")

                    logger.info(f"response output: {output}")
                    logger.error(f"{context_name}返回错误：code={error_code}, message={error_message}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"等待 {wait_time:.2f}s 后重试{context_name}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"{context_name}返回错误：{error_message}")
                
                if self.rate_limiter and getattr(self.rate_limiter, "tokens_per_minute", 0) > 0:
                    usage = output.get('usage', {}) if isinstance(output, dict) else {}
                    tokens_used = usage.get('total_tokens', 0)
                    if tokens_used > 0:
                        await self.rate_limiter.register_tokens(tokens_used)
                        estimated = getattr(self.rate_limiter, "estimated_tokens_per_request", 0)
                        if estimated > 0:
                            diff_percent = ((tokens_used - estimated) / estimated) * 100
                            if abs(diff_percent) > 20:  # 偏差超过20%时记录警告
                                logger.debug(
                                    f"{context_name}token消耗: 实际={tokens_used}, "
                                    f"预估={estimated}, 偏差={diff_percent:+.1f}%"
                                )

                return output
                
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                logger.warning(
                    f'{context_name}网络异常 (尝试 {attempt + 1}/{self.max_retries}): '
                    f'{type(e).__name__} - {str(e)}'
                )
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f'等待 {wait_time} 秒后重试...')
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f'{context_name}重试次数已用尽')
                    raise
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    logger.warning(
                        f'{context_name}服务器错误 (尝试 {attempt + 1}/{self.max_retries}): '
                        f'{e.response.status_code}'
                    )
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f'等待 {wait_time} 秒后重试...')
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
                    
            except Exception as e:
                logger.error(f'{context_name}请求异常: {type(e).__name__} - {str(e)}')
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f'等待 {wait_time} 秒后重试...')
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise RuntimeError(f"{context_name}失败，已重试 {self.max_retries} 次仍未成功")

