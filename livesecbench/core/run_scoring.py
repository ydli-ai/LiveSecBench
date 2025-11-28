import importlib
import time
from typing import Optional, Tuple, Dict, Any, Callable

from livesecbench.infra.config import ConfigManager
from livesecbench.infra.http_client import RetryableHTTPClient
from livesecbench.storage.sqlite_storage import SQLiteStorage
from livesecbench.utils.env_loader import load_project_env
from livesecbench.utils.logger import get_logger

load_project_env()
logger = get_logger(__name__)


def _build_pk_payload(
    evaluation_dimension: str,
    category: str,
    question: str,
    model_a: str,
    model_b: str,
    reasoning_a: Optional[str],
    reasoning_b: Optional[str],
    answer_a: str,
    answer_b: str,
    output_a: str,
    output_b: str,
    prompt: str,
    winner: str,
    content: str,
    current_time: str,
    consume_time: float,
    true_answer: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "A": model_a,
        "B": model_b,
        "winner": winner,
        "获胜模型": winner,
        "evaluation_dimension": evaluation_dimension,
        "category": category,
        "测试题目": question,
        "模型A的思维链": reasoning_a,
        "模型A的回答": answer_a,
        "模型A的思维链&回答": output_a,
        "模型B的思维链": reasoning_b,
        "模型B的回答": answer_b,
        "模型B的思维链&回答": output_b,
        "PK判别提示词": prompt,
        "PK判别结果": content,
        "consume": consume_time,
        "timestamp": time.time(),
        "current_time": current_time,
    }
    if true_answer is not None:
        payload["事实性正确答案"] = true_answer
    return payload


def _swap_pk_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """交换PK结果 payload 中的模型A/B字段，用于调整缓存结果的方向"""
    swapped = dict(payload)
    swapped['A'] = payload['B']
    swapped['B'] = payload['A']
    swapped['模型A的思维链'] = payload.get('模型B的思维链')
    swapped['模型A的回答'] = payload.get('模型B的回答')
    swapped['模型A的思维链&回答'] = payload.get('模型B的思维链&回答')
    swapped['模型B的思维链'] = payload.get('模型A的思维链')
    swapped['模型B的回答'] = payload.get('模型A的回答')
    swapped['模型B的思维链&回答'] = payload.get('模型A的思维链&回答')

    if payload.get('winner') == payload.get('A'):
        new_winner = payload.get('B')
    else:
        new_winner = payload.get('A')
    swapped['winner'] = new_winner
    swapped['获胜模型'] = new_winner
    return swapped


def _canonical_pair(model_a: str, model_b: str) -> Tuple[str, str, bool]:
    """返回排序后的模型对以及是否与原始顺序相反，用于确保缓存键的一致性"""
    if model_a <= model_b:
        return model_a, model_b, False
    return model_b, model_a, True


async def pk(
    http_client: RetryableHTTPClient,
    evaluation_dimension: str,
    evaluate_prompt_template: str,
    judge_model: str,
    storage: SQLiteStorage,
    category: str,
    question: str,
    model_A: str,
    model_B: str,
    reasoning_A: Optional[str],
    reasoning_B: Optional[str],
    answer_A: str,
    answer_B: str,
    true_answer: Optional[str] = None,
    endpoint: str = "chat/completions",
) -> Tuple[Optional[str], bool, float, Optional[str], Dict[str, Any]]:
    """进行模型A和模型B的PK，返回获胜模型"""
    output_A = f'<think>{reasoning_A}</think>\n\n{answer_A}' if reasoning_A is not None else answer_A
    output_B = f'<think>{reasoning_B}</think>\n\n{answer_B}' if reasoning_B is not None else answer_B

    if evaluation_dimension in ('事实性', 'factuality'):
        prompt = evaluate_prompt_template.format(question, true_answer, output_A, output_B)
    elif evaluation_dimension in ('reasoning', '推理安全'):
        prompt = evaluate_prompt_template.format(question, reasoning_A, reasoning_B)
    else:
        prompt = evaluate_prompt_template.format(question, output_A, output_B)

    key_model_a, key_model_b, swap_for_storage = _canonical_pair(model_A, model_B)

    cached = storage.get_pk_result(category, question, key_model_a, key_model_b)
    if cached:
        if swap_for_storage:
            cached = _swap_pk_payload(cached)
        return cached["winner"], False, 0, None, cached

    req_data = {
        "model": judge_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    start_time = time.time()
    
    try:
        output = await http_client.post(
            endpoint=endpoint,
            json_data=req_data,
            context_name="PK判别模型"
        )
        
        content = output['choices'][0]['message']['content']
        
    except Exception as e:
        logger.error(f'PK请求失败: {str(e)}')
        return None, None, None, None, {}
    
    end_time = time.time()
    consume_time = end_time - start_time

    if ('A' not in content and 'B' not in content) or ('A' in content and 'B' in content):
        logger.warning('模型未能按照预期格式输出A或B，AB都不在或者AB都在')
        return None, None, None, None, {}
    content = content.strip()

    if content == 'A' or 'A' in content:
        winner = model_A
    else:
        winner = model_B

    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    result_payload = _build_pk_payload(
        evaluation_dimension,
        category,
        question,
        model_A,
        model_B,
        reasoning_A,
        reasoning_B,
        answer_A,
        answer_B,
        output_A,
        output_B,
        prompt,
        winner,
        content,
        current_time,
        consume_time,
        true_answer,
    )

    storage_payload = result_payload if not swap_for_storage else _swap_pk_payload(result_payload)
    storage.save_pk_result(evaluation_dimension, category, question, key_model_a, key_model_b, storage_payload)

    return winner, True, consume_time, content.strip(), result_payload


def create_pk_runner(
    criteria_template: str,
    judge_model: str,
    storage: SQLiteStorage,
    judge_api_config: Dict[str, Any],
) -> Callable:
    """创建PK运行器，返回一个异步函数包装器"""
    base_url = judge_api_config.get('base_url', 'https://api.deepseek.com/v1')
    api_key = judge_api_config.get('api_key', '')
    timeout = judge_api_config.get('timeout', 120)
    max_retries = judge_api_config.get('max_retries', 5)
    retry_delay = judge_api_config.get('retry_delay', 1)
    endpoint = judge_api_config.get('end_point', 'chat/completions')
    
    if not base_url or not api_key:
        raise ValueError("judge_model_api 缺少必要的 base_url 或 api_key 配置")
    
    http_client = RetryableHTTPClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    
    async def pk_wrapper(
        evaluation_dimension: str,
        category: str,
        question: str,
        model_A: str,
        model_B: str,
        reasoning_A: Optional[str],
        reasoning_B: Optional[str],
        answer_A: str,
        answer_B: str,
        true_answer: Optional[str] = None,
    ):
        return await pk(
            http_client=http_client,
            evaluation_dimension=evaluation_dimension,
            evaluate_prompt_template=criteria_template,
            judge_model=judge_model,
            storage=storage,
            category=category,
            question=question,
            model_A=model_A,
            model_B=model_B,
            reasoning_A=reasoning_A,
            reasoning_B=reasoning_B,
            answer_A=answer_A,
            answer_B=answer_B,
            true_answer=true_answer,
            endpoint=endpoint,
        )
    return pk_wrapper


def build_model_result_fetcher(storage: SQLiteStorage) -> Callable[[str, str, str], Optional[Dict[str, Any]]]:
    """模型结果获取函数"""
    def _fetch(model: str, category: str, prompt: str) -> Optional[Dict[str, Any]]:
        return storage.get_model_output(model, category, prompt)

    return _fetch


def merge_elo_settings(global_settings: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = dict(global_settings or {})
    if overrides:
        settings.update(overrides)

    # 核心参数
    required_keys = [
        "init_rating",
        "init_k",
        "logistic_constant",
        "swiss_group_num",
        "max_workers",
        "result_output_dir",
    ]
    
    missing = [key for key in required_keys if key not in settings]
    if missing:
        raise ValueError(f"ELO配置缺少必要参数: {missing}")
    
    default_templates = {
        "history_filename_template": "{dimension}_elo_history_{timestamp}.csv",
        "result_filename_template": "{dimension}_elo_raw_{timestamp}.csv",
        "record_filename_template": "{dimension}_pk_details_{timestamp}.xlsx",
    }
    
    for key, default_value in default_templates.items():
        if key not in settings:
            settings[key] = default_value
    
    if "history_output_dir" not in settings:
        settings["history_output_dir"] = settings.get("result_output_dir", "elo_results")
    if "record_output_dir" not in settings:
        settings["record_output_dir"] = settings.get("result_output_dir", "elo_results")
    
    return settings

def load_scorer(scorer_name: str) -> Callable:
    """动态加载评分器模块"""
    try:
        scorer_module = importlib.import_module(f"livesecbench.scorers.{scorer_name}")
        return getattr(scorer_module, 'score')
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"无法加载名为 '{scorer_name}' 的评分器。请确保 'livesecbench/scorers/{scorer_name}.py' 文件存在且包含 'score' 函数。错误详情: {e}")


async def launch_evaluation(
    config_manager: ConfigManager, 
    dimension_questions: Dict[str, list],
        task_manager: Optional[Any] = None,
) -> None:
    """启动评测流程: 以题集维度为准，定向关联对应维度的评分器配置"""
    scoring_config = config_manager.get_scoring_config()
    global_elo_settings = config_manager.get_elo_settings()
    
    dimension_to_scorer_config = {}
    for item in scoring_config:
        dimension = item.get('dimension')
        if dimension:
            dimension_to_scorer_config[dimension] = item
    
    # 加载存储配置
    storage_tables = config_manager.get_storage_tables()
    task_id = task_manager.task_id if task_manager else None
    storage = SQLiteStorage(
        db_path=config_manager.get_storage_db_path(),
        model_outputs_table=storage_tables['model_outputs_table'],
        pk_results_table=storage_tables['pk_results_table'],
        task_id=task_id,
    )
    
    all_models = config_manager.get_all_model_ids()
    reasoning_models = config_manager.get_reasoning_model_ids()
    judge_api_config = config_manager.get_judge_model_api()
    
    fetch_model_result = build_model_result_fetcher(storage)

    for dimension, dimension_questions_list in dimension_questions.items():
        if not dimension_questions_list:
            logger.warning(f"{dimension}: 题目列表为空，跳过评分流程。")
            continue
        
        item = dimension_to_scorer_config.get(dimension)
        if not item:
            logger.warning(f"{dimension}: 未找到对应的评分器配置，跳过该维度。")
            continue
        
        scorer_name = item.get('scorer')
        params = item.get('params', {})
        if not scorer_name:
            logger.warning(f"{dimension}: 评分配置缺少scorer字段，跳过该维度。")
            continue

        criteria_template = params.get('criteria_template')
        if not criteria_template:
            logger.warning(f"{dimension}: 缺少criteria_template，跳过该维度。")
            continue

        try:
            elo_settings = merge_elo_settings(global_elo_settings, params.get('elo'))
        except ValueError as exc:
            logger.error(f"{dimension}: {exc}")
            continue

        dimension_judge_api = params.get('judge_api', {}) or {}
        final_judge_api_config = dict(judge_api_config)
        final_judge_api_config.update(dimension_judge_api)

        judge_model = final_judge_api_config.get('model')
        if not judge_model:
            logger.warning(f"{dimension}: 未配置judge_model，跳过该维度。")
            continue

        pk_runner = create_pk_runner(criteria_template, judge_model, storage, final_judge_api_config)
        runtime_context = {
            'logger': logger,
            'pk_runner': pk_runner,
            'fetch_model_result': fetch_model_result,
            'elo_settings': elo_settings,
            'task_manager': task_manager,  # 传递任务管理器
            'config_manager': config_manager,  # 传递配置管理器，用于获取输出路径
        }

        scorer_fn = load_scorer(scorer_name)

        start = time.time()
        logger.info("开始处理维度: %s (题目数: %d)", dimension, len(dimension_questions_list))

        result = await scorer_fn(
            evaluation_dimension=dimension,
            dimension_questions=dimension_questions_list,
            models=all_models,
            reasoning_models=reasoning_models,
            scorer_params=params,
            runtime_context=runtime_context,
        )

        end = time.time()
        logger.info(f"{dimension}, 处理完成")
        logger.info(f"{dimension}, 总耗时: {end - start:.2f}秒")

        if result:
            logger.info(
                "%s: 结果文件 -> history: %s, ranking: %s, records: %s",
                dimension,
                result.get('history_path'),
                result.get('result_path'),
                result.get('record_path'),
            )
            # 记录结果文件到任务管理器
            if task_manager:
                if result.get('history_path'):
                    task_manager.add_result_file(result.get('history_path'), 'elo_history')
                if result.get('result_path'):
                    task_manager.add_result_file(result.get('result_path'), 'elo_raw')
                if result.get('record_path'):
                    task_manager.add_result_file(result.get('record_path'), 'pk_details')
