# encoding: utf-8
import asyncio
import base64
import datetime
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

from livesecbench.infra.config import ConfigManager
from livesecbench.infra.http_client import RetryableHTTPClient, RateLimiter
from livesecbench.storage.sqlite_storage import SQLiteStorage
from livesecbench.utils.env_loader import load_project_env
from livesecbench.utils.logger import get_logger

load_project_env()
logger = get_logger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def single_question_call(
        http_client: RetryableHTTPClient,
        semaphore: asyncio.Semaphore,
        model_name: str,
        model: str,
        input_data: dict,
        storage: SQLiteStorage,
        reasoning_enabled: bool = True,
        is_reasoning_model: bool = False,
        image_path: Optional[str] = None,
        model_error_handlers: Optional[Dict[str, str]] = None,
        provider_ignore: Optional[list] = None,
        endpoint: str = "chat/completions",
        use_structured_content: bool = False,
) -> Dict[str, Any]:
    """测试单个问题，获取模型的回答"""
    prompt = input_data.get('question_text') or input_data.get('prompt', '')
    category = input_data.get('dimension', '')
    existing = await storage.aget_model_output(model, category, prompt)
    if existing and existing.get('status') == 'success':
        return existing
    
    model_error_handlers = model_error_handlers or {}

    try:
        
        if image_path:
            if image_path.startswith('http://') or image_path.startswith('https://'):
                data_url = image_path
            else:
                base64_image = encode_image_to_base64(image_path)
                image_format = Path(image_path).suffix.replace('.', '')
                if image_path.endswith('jpg') or image_path.endswith('jpeg'):
                    image_format = 'jpeg'
                if image_format not in ['png', 'jpeg', 'gif', 'webp']:
                    raise Exception("不支持的图片格式，仅支持 png, jpeg, gif, webp")
                data_url = f"data:image/{image_format};base64,{base64_image}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
        elif use_structured_content:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        req_data = {
            "model": model,
            "messages": messages
        }

        if reasoning_enabled and is_reasoning_model:
            req_data['reasoning'] = {"enabled": True}
        
        if provider_ignore:
            if isinstance(provider_ignore, str):
                req_data['provider'] = {'ignore': provider_ignore.split(',')}
            elif isinstance(provider_ignore, list):
                req_data['provider'] = {'ignore': provider_ignore}

        async with semaphore:
            start = time.time()
            output = await http_client.post(
                endpoint=endpoint,
                json_data=req_data,
                context_name=f"模型 {model_name}"
            )

        end = time.time()
        consume_time = end - start

        answer = None
        reasoning = None
        provider = None
        prompt_tokens = None
        completion_tokens = None
        server_model = None
        
        if 'choices' not in output or len(output['choices']) == 0:
            if 'error' in output:
                error_info = output['error']
                if isinstance(error_info, dict):
                    error_code = error_info.get('code')
                    error_type = error_info.get('type')
                    error_message = error_info.get('message', '')
                    
                    if error_code == '204' or error_type == 'content_filtered':
                        answer = error_message or '内容审查：该问题触发了安全过滤'
    
            if answer is None:
                if model in model_error_handlers:
                    answer = model_error_handlers[model]
                elif any(model.startswith(prefix) for prefix in ['gemini-2.0-flash']):
                    answer = model_error_handlers.get('gemini-2.0-flash', 'ext. PROHIBITED_CONTENT')
                elif any(model.startswith(prefix) for prefix in ['x-ai/grok-4-fast', 'x-ai/grok-3-mini']):
                    answer = model_error_handlers.get('x-ai/grok-4-fast', 
                        'The caller does not have permission to execute the specified operation. Content violates usage guidelines.')
                elif any(model.startswith(prefix) for prefix in ['Doubao-1.5-pro-32k', 'doubao-seed-1-6']):
                    answer = model_error_handlers.get('Doubao-1.5-pro-32k', '换个问题试试')
        else:
            answer = output['choices'][0]['message']['content']
            if 'reasoning' in output['choices'][0]['message']:
                reasoning = output['choices'][0]['message']['reasoning']
            elif 'reasoning_content' in output['choices'][0]['message']:
                reasoning = output['choices'][0]['message']['reasoning_content']

            if reasoning is not None and not isinstance(reasoning, str):
                reasoning = None

            if 'provider' in output:
                provider = output['provider']
            prompt_tokens = output.get('usage', {}).get('prompt_tokens')
            completion_tokens = output.get('usage', {}).get('completion_tokens')
            server_model = output.get('model')

        if answer is None:
            return {
                "status": "error",
                "model_name": model_name,
                "model": model,
                "category": category,
                "prompt": prompt,
                "error": "响应中没有choices且未配置错误处理器",
                "created_at": int(time.time()),
                "current_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        pattern = r'^(<think\b[^>]*>.*?<\/think>)\s*(.*)$'
        match = re.match(pattern, answer, re.DOTALL)
        if match:
            think_content = match.group(1)
            answer = match.group(2).strip()

            content_pattern = r'^<think\b[^>]*>(.*?)<\/think>$'
            content_match = re.match(content_pattern, think_content, re.DOTALL)
            think_text = content_match.group(1).strip() if content_match else think_content
            if reasoning is None:
                reasoning = think_text

        resp = {
            "status": "success",
            "model_name": model_name,
            "model": model,
            "server_model": server_model,
            "category": category,
            "prompt": prompt,
            "prompt_hash": input_data.get('question_id', ''),
            "reasoning": reasoning,
            "answer": answer,
            "provider": provider,
            "consume_time": consume_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "created_at": int(time.time()),
            "current_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if category != '事实性':
            metadata = input_data.get('metadata', {})
            resp.update({
                "prompt_type": metadata.get('type', ''),
                "prompt_difficulty": metadata.get('difficulty', ''),
                "prompt_category": metadata.get('category', ''),
            })
        else:
            reference_answer = input_data.get('reference_answer', [])
            metadata = input_data.get('metadata', {})
            resp.update({
                "true_answer": reference_answer[0] if reference_answer else "",
                "delusion_type": metadata.get('type', ''),
            })
        return resp
    except Exception as e:
        logger.error("异常文件: {}，所在行: {}，异常信息: {}".format(e.__traceback__.tb_frame.f_globals.get("__file__", "NULL"), e.__traceback__.tb_lineno, e.args))
        logger.error(f"模型 {model_name} LLM 请求失败: {type(e).__name__} - {e}")
        logger.info(f'model_name: {model_name}')
        logger.info(f'model: {model}')
        logger.info(f'prompt: {prompt}')
        return {
            "status": "error",
            "model_name": model_name,
            "model": model,
            "category": category,
            "prompt": prompt,
            "error": str(e),
            "created_at": int(time.time()),
            "current_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


async def batch_test_model(
        model_item: Dict,
        questions: List[dict],
        storage: SQLiteStorage,
        timeout: int = 210,
        max_concurrent: int = 10,
        reasoning_enabled: bool = True,
        max_retries: int = 5,
        retry_delay: int = 1,
        global_rate_limit_per_second: int = 0,
        global_rate_limit_per_minute: int = 0,
        global_tpm: int = 0,
        model_error_handlers: Optional[Dict] = None,
) -> None:
    """批量测试模型"""
    api_config = model_item.get('api_config', {})
    if not api_config:
        logger.error(f"模型 {model_item.get('model_name')} 缺少 api_config 配置")
        return
    
    base_url = api_config.get('base_url')
    api_key = api_config.get('api_key')
    model_id = api_config.get('model_id')
    provider_ignore = api_config.get('provider_ignore')
    endpoint = api_config.get('end_point', 'chat/completions')
    model_max_concurrent = api_config.get('max_concurrent') or max_concurrent
    model_rpm = api_config.get('rpm') or global_rate_limit_per_minute
    model_tpm = api_config.get('tpm') or global_tpm
    model_estimated_tokens = api_config.get('estimated_tokens_per_request', 4000)
    
    if not base_url:
        logger.error(f"模型 {model_item.get('model_name')} 的 api_config 缺少 base_url")
        return
    if not api_key:
        logger.error(f"模型 {model_item.get('model_name')} 的 api_config 缺少 api_key")
        return
    if not model_id:
        logger.error(f"模型 {model_item.get('model_name')} 的 api_config 缺少 model_id")
        return
    
    model_item['model'] = model_id

    rate_limiter = None
    if global_rate_limit_per_second > 0 or model_rpm > 0 or model_tpm > 0:
        rate_limiter = RateLimiter(
            per_second=global_rate_limit_per_second,
            per_minute=model_rpm,
            tokens_per_minute=model_tpm,
            estimated_tokens_per_request=model_estimated_tokens,
        )
        limit_info = []
        if global_rate_limit_per_second > 0:
            limit_info.append(f"每秒={global_rate_limit_per_second}请求")
        if model_rpm > 0:
            limit_info.append(f"每分钟={model_rpm}请求")
        if model_tpm > 0:
            limit_info.append(f"每分钟={model_tpm}tokens (预估每请求={model_estimated_tokens}tokens)")
        logger.info(
            f"模型 {model_item.get('model_name')} 启用速率限制: {', '.join(limit_info)}"
        )

    http_client = RetryableHTTPClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limiter=rate_limiter,
    )

    logger.info(f"开始测试模型: {model_item['model_name']} ({model_item['model']}), 题目: {len(questions)}, 并发: {max_concurrent}")

    success_count = 0
    error_count = 0
    total_time = 0
    pending_questions = []
    skipped_count = 0

    try:
        for item in questions:
            prompt_text = item.get('question_text')
            category = item.get('dimension')
            existing = await storage.aget_model_output(model_item['model'], category, prompt_text)
            if existing and existing.get('status') == 'success':
                skipped_count += 1
                success_count += 1
                continue
            pending_questions.append(item)

        if skipped_count:
            logger.info(f"模型 {model_item['model_name']} 已命中缓存 {skipped_count} 条，跳过重复处理。")

        if not pending_questions:
            logger.info(f"模型 {model_item['model_name']} 所有题目均已缓存，跳过请求。")
            return

        semaphore = asyncio.Semaphore(model_max_concurrent)
        is_reasoning_model = model_item.get('is_reasoning', False)
        enable_image_text = model_item.get('image_text_input', False)
        use_structured_content = model_item.get('use_structured_content', False)
        
        task_to_info = {}
        tasks = []
        for idx, item in enumerate(pending_questions, 1):
            image_path = None
            if enable_image_text:
                image_path = item.get('image_path') or item.get('image_url')
                if not image_path:
                    metadata = item.get('metadata') or {}
                    image_path = metadata.get('image_path') or metadata.get('image_url')
            coro = single_question_call(
                http_client=http_client,
                semaphore=semaphore,
                model_name=model_item['model_name'],
                model=model_item['model'],
                input_data=item,
                storage=storage,
                reasoning_enabled=reasoning_enabled,
                is_reasoning_model=is_reasoning_model,
                image_path=image_path,
                model_error_handlers=model_error_handlers,
                provider_ignore=provider_ignore,
                endpoint=endpoint,
                use_structured_content=use_structured_content,
            )
            task = asyncio.create_task(coro)
            tasks.append(task)
            task_to_info[task] = (idx, item)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task, result in zip(tasks, results):
            try:
                if isinstance(result, Exception):
                    idx = task_to_info.get(task, (0, {}))[0]
                    category = task_to_info.get(task, (0, {}))[1].get('dimension', 'unknown')
                    error_count += 1
                    logger.error(f"[{idx}/{len(pending_questions)}] ✗ 异常 | 分类: {category} | 错误: {str(result)}")
                    logger.error("异常文件: {}，所在行: {}，异常信息: {}".format(result.__traceback__.tb_frame.f_globals.get("__file__", "NULL"), result.__traceback__.tb_lineno, result.args))
                    continue
                
                if task not in task_to_info:
                    logger.warning(f"任务不在task_to_info中，跳过")
                    continue
                    
                idx, question = task_to_info[task]
                category = question.get('dimension')

                if not result:
                    logger.warning(f"[{idx}/{len(pending_questions)}] 结果为空，跳过 | 分类: {category}")
                    logger.debug(f"结果内容: {result}")
                    continue
                
                if result.get('status') == 'success':
                    answer_preview = str(result.get('answer', ''))[:200]
                    logger.info(f"[{idx}] 模型回答预览: {answer_preview}...")
                
                if result.get('status') == 'error':
                    error_count += 1
                    logger.info(f"[{idx}/{len(pending_questions)}] ✗ 失败 | 分类: {category} | "
                          f"错误: {result.get('error', 'Unknown')}")
                    continue

                await storage.asave_model_output(result)

                if result.get('status') == 'success':
                    success_count += 1
                    total_time += result.get('consume_time', 0)
                    logger.info(f"[{idx}/{len(pending_questions)}] ✓ 成功 | 分类: {category} | "
                          f"耗时: {result.get('consume_time', 0):.2f}s | "
                          f"已写入SQLite")
                else:
                    error_count += 1
                    logger.info(f"[{idx}/{len(pending_questions)}] ✗ 失败 | 分类: {category} | "
                          f"错误: {result.get('error', 'Unknown')}")

            except Exception as e:
                idx = task_to_info.get(task, (0, {}))[0]
                category = task_to_info.get(task, (0, {}))[1].get('dimension', 'unknown')
                error_count += 1
                logger.error(f"[{idx}/{len(pending_questions)}] ✗ 异常 | 分类: {category} | 错误: {str(e)}")
                logger.error("异常文件: {}，所在行: {}，异常信息: {}".format(e.__traceback__.tb_frame.f_globals.get("__file__", "NULL"), e.__traceback__.tb_lineno, e.args))

    finally:
        pass

    logger.info(f"测试完成: {model_item['model_name']}")
    logger.info(f"总数: {len(questions)} | 成功: {success_count} | 失败: {error_count}")
    if success_count > 0:
        logger.info(f"平均耗时: {total_time / success_count:.2f}s")


async def batch_gen_llm_answer(
    target_model_list: List[Dict[str, Any]],
    all_questions: List[Dict[str, Any]],
    config_manager: Optional[ConfigManager] = None,
    task_id: Optional[str] = None,
) -> None:
    """批量生成所有模型的回答"""
    if config_manager is None:
        from livesecbench.infra.config import ConfigManager
        config_path = Path(__file__).resolve().parent.parent / "configs" / "run_custom_safety_benchmark.yaml"
        config_manager = ConfigManager(str(config_path))
    
    storage_tables = config_manager.get_storage_tables()
    storage = SQLiteStorage(
        db_path=config_manager.get_storage_db_path(),
        model_outputs_table=storage_tables['model_outputs_table'],
        pk_results_table=storage_tables['pk_results_table'],
        task_id=task_id,
    )
    
    api_call_settings = config_manager.get_api_call_settings()
    timeout = api_call_settings.get('timeout', 210)
    max_concurrent = api_call_settings.get('max_concurrent', 10)
    reasoning_enabled = api_call_settings.get('reasoning_enabled', True)
    max_retries = api_call_settings.get('max_retries', 5)
    retry_delay = api_call_settings.get('retry_delay', 1)
    rate_limit_per_second = api_call_settings.get('rate_limit_per_second', 0)
    rate_limit_per_minute = api_call_settings.get('rate_limit_per_minute', 0)
    tpm = api_call_settings.get('tpm', 0)
    
    model_error_handlers = config_manager.get_model_error_handlers()
    
    for model_item in target_model_list:
        if 'api_config' not in model_item:
            logger.warning(f"模型 {model_item.get('model_name', 'unknown')} 缺少 api_config 配置，跳过")
            continue

        await batch_test_model(
            model_item,
            all_questions,
            storage=storage,
            timeout=timeout,
            max_concurrent=max_concurrent,
            reasoning_enabled=reasoning_enabled,
            max_retries=max_retries,
            retry_delay=retry_delay,
            global_rate_limit_per_second=rate_limit_per_second,
            global_rate_limit_per_minute=rate_limit_per_minute,
            global_tpm=tpm,
            model_error_handlers=model_error_handlers,
        )
