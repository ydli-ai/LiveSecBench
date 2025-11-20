import csv
import time
from pathlib import Path
from typing import Optional, Any

from livesecbench.infra.config import ConfigManager
from livesecbench.infra.http_client import RetryableHTTPClient
from livesecbench.utils.env_loader import load_project_env
from livesecbench.utils.logger import get_logger

load_project_env()
logger = get_logger(__name__)


async def generate_report_with_llm(
    prompt: str,
    config_manager: Optional[ConfigManager] = None,
) -> str:
    """使用配置的评估模型生成报告"""
    if config_manager is None:
        default_config_path = Path(__file__).resolve().parent.parent / "configs" / "run_custom_safety_benchmark.yaml"
        config_manager = ConfigManager(str(default_config_path))
    
    judge_api_config = config_manager.get_judge_model_api()
    base_url = judge_api_config.get('base_url', 'https://api.deepseek.com/v1')
    api_key = judge_api_config.get('api_key')
    model = judge_api_config.get('model', 'deepseek-chat')
    timeout = judge_api_config.get('timeout', 300)
    max_retries = judge_api_config.get('max_retries', 5)
    retry_delay = judge_api_config.get('retry_delay', 1)
    
    if not api_key:
        raise ValueError("未找到有效的API密钥配置")
    
    http_client = RetryableHTTPClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    
    req_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        output = await http_client.post(
            endpoint="chat/completions",
            json_data=req_data,
            context_name="报告生成"
        )
        
        if 'choices' not in output or len(output['choices']) == 0:
            raise RuntimeError("API响应中未找到有效内容")
        
        report_content = output['choices'][0]['message']['content']
        return report_content
        
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        raise RuntimeError(f"生成报告失败: {e}") from e


def _build_report_prompt(
    config_manager: ConfigManager,
    dimensions: list,
    task_manager: Optional[Any] = None
) -> str:
    """构建报告提示词"""
    day_tag = time.strftime('%Y_%m_%d', time.localtime())
    month_tag = time.strftime('%Y-%m', time.localtime())
    
    if task_manager:
        models_file = Path(f"results/{day_tag}/{month_tag}-models_{task_manager.task_id}.csv")
        stats_file = Path(f"results/{day_tag}/{month_tag}-stats_{task_manager.task_id}.csv")
    else:
        models_file = Path(f"results/{day_tag}/{month_tag}-models.csv")
        stats_file = Path(f"results/{day_tag}/{month_tag}-stats.csv")

    if not models_file.exists():
        logger.warning("未找到模型评分文件，跳过总结报告生成。")
        return ""
    if not stats_file.exists():
        logger.warning("未找到统计摘要文件，跳过总结报告生成。")
        return ""

    with models_file.open('r', encoding='utf-8') as f:
        models_reader = list(csv.DictReader(f))

    with stats_file.open('r', encoding='utf-8') as f:
        stats_reader = list(csv.DictReader(f))

    if not models_reader or not stats_reader:
        logger.warning("评分文件内容为空，无法生成提示词。")
        return ""

    stats = stats_reader[0]
    top_models = models_reader[:3]
    column_alias = {
        '合法性': 'legality',
        '伦理性': 'ethics',
        '事实性': 'factuality',
        '隐私性': 'privacy',
        '对抗鲁棒性': 'robustness',
        '推理安全': 'reasoning',
    }

    metric_columns = [
        col for col in models_reader[0].keys()
        if col not in {'rank', 'model_name', 'provider', 'overall_score', 'evaluation_date', 'rank_change',
                       'is_open_source', 'publish_time', 'location', 'model_url'}
    ]
    if not metric_columns and dimensions:
        metric_columns = [column_alias.get(dim, dim) for dim in dimensions]

    dimension_summary = []
    for dim_key in metric_columns:
        dim_label = next((alias for alias, target in column_alias.items() if target == dim_key), dim_key)
        scores = []
        for row in models_reader:
            value = row.get(dim_key)
            if value is None or value == '':
                continue
            try:
                scores.append(float(value))
            except ValueError:
                continue
        if not scores:
            continue
        dimension_summary.append(f"{dim_label} 平均得分 {sum(scores) / len(scores):.2f}")

    top_lines = []
    for row in top_models:
        top_lines.append(f"{row.get('rank', '')}. {row.get('model_name', '')}（综合得分 {row.get('overall_score', '')}，提供方 {row.get('provider', '未知')}）")
    all_lines = []
    for row in models_reader:
        all_lines.append(f"{row.get('rank', '')}. {row.get('model_name', '')}（综合得分 {row.get('overall_score', '')}，合法性 {row.get('legality', '')}，伦理性 {row.get('ethics', '')}，事实性 {row.get('factuality', '')}，隐私性 {row.get('privacy', '')}，对抗鲁棒性 {row.get('robustness', '')}，推理安全 {row.get('reasoning', '')}，提供方 {row.get('provider', '未知')}）")

    task_info_section = ""
    if task_manager:
        task_info_section = (
            "【任务信息】\n"
            f"- 任务ID：{task_manager.task_id}\n"
            f"- 评测运行名称：{task_manager.task_info.get('eval_run_name', '未知')}\n"
            f"- 创建时间：{task_manager.task_info.get('created_at', '未知')}\n"
            f"- 评测模型数量：{len(task_manager.task_info.get('models', []))} 个\n"
            f"- 评测维度：{', '.join(task_manager.task_info.get('dimensions', []))}\n\n"
        )
    
    prompt_template = config_manager.get_report_prompt_template()
    if not prompt_template:
        logger.warning("未找到报告提示词模板")
        return ""
    
    prompt = prompt_template.format(
        task_info_section=task_info_section,
        total_models=stats.get('totalModels', '未知'),
        average_score=stats.get('averageScore', '未知'),
        dimension_count=len(metric_columns),
        dimension_list=', '.join(metric_columns) if metric_columns else '无可用数据',
        last_update=stats.get('lastUpdate', '未知'),
        top_models_list=chr(10).join(top_lines) if top_lines else '暂无数据',
        dimension_summary=chr(10).join(dimension_summary) if dimension_summary else '暂无有效维度数据',
        all_models_list=chr(10).join(all_lines) if all_lines else '暂无数据',
    )
    
    return prompt


async def generate_report(
    config_manager: ConfigManager,
    dimensions: list,
    task_manager: Optional[Any] = None
) -> None:
    """生成测试报告"""
    prompt = _build_report_prompt(config_manager, dimensions, task_manager)
    
    if not prompt:
        logger.warning("无法构建报告提示词，跳过报告生成")
        return
    
    try:
        report_content = await generate_report_with_llm(prompt, config_manager=config_manager)
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        raise
    
    day_tag = time.strftime('%Y_%m_%d', time.localtime())
    results_dir = Path("results") / day_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_metadata = ""
    if task_manager:
        report_metadata = (
            "---\n"
            f"任务ID: {task_manager.task_id}\n"
            f"评测运行名称: {task_manager.task_info.get('eval_run_name', '未知')}\n"
            f"创建时间: {task_manager.task_info.get('created_at', '未知')}\n"
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
        )
        
        result_files = task_manager.task_info.get('result_files', [])
        if result_files:
            report_metadata += "关联结果文件:\n"
            for rf in result_files:
                report_metadata += f"  - [{rf.get('type', 'unknown')}] {rf.get('path', '')}\n"
        report_metadata += "---\n\n"
    
    if task_manager:
        report_file = results_dir / f"summary_report_{task_manager.task_id}.md"
    else:
        report_file = results_dir / "summary_report.md"
    
    full_report_content = report_metadata + report_content
    report_file.write_text(full_report_content, encoding='utf-8')
    
    if task_manager:
        task_manager.add_result_file(str(report_file), 'summary_report')
    
    logger.info(f"报告已生成: {report_file}")
