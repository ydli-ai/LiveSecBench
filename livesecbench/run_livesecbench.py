import argparse
import asyncio
import json
import random
from pathlib import Path

from livesecbench.core.rank import rank
from livesecbench.core.report import generate_report
from livesecbench.core.run_model_answer import batch_gen_llm_answer
from livesecbench.core.run_scoring import launch_evaluation
from livesecbench.infra.config import ConfigManager
from livesecbench.utils.logger import configure_root_logger, get_logger
from livesecbench.utils.env_loader import load_project_env

configure_root_logger(level='INFO', log_to_file=True, log_to_console=True)
logger = get_logger(__name__)


def load_questions(base_path, selection_config):
    """根据配置从题库加载问题，支持 adversarial_level 和 limit 筛选"""
    all_questions = []
    dimension_questions = {}
    
    for selection in selection_config:
        dimension = selection['dimension']
        adversarial_levels = selection.get('adversarial_level', None)
        limit = selection.get('limit', 0)
        
        dimension_questions[dimension] = []
        loaded_questions = []
        for question_set in selection.get('question_sets', []):
            question_set_path = Path(base_path) / question_set
            if not question_set_path.exists():
                logger.warning(f"题集目录不存在: {question_set_path}")
                continue
            
            for file_path in question_set_path.glob('*.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        questions = json.load(f)
                        loaded_questions.extend(questions)
                except Exception as e:
                    logger.error(f"加载文件失败: {file_path}, 错误: {e}")
        
        if adversarial_levels:
            filtered_questions = []
            for q in loaded_questions:
                q_adversarial = q.get('metadata', {}).get('adversarial_level', '')
                if q_adversarial in adversarial_levels:
                    filtered_questions.append(q)
            loaded_questions = filtered_questions
        
        if 0 < limit < len(loaded_questions):
            loaded_questions = random.sample(loaded_questions, limit)
        
        dimension_questions[dimension] = loaded_questions
        all_questions.extend(loaded_questions)
        logger.info(f"{dimension}: {len(loaded_questions)} 题")
    
    # 去重
    seen_ids = set()
    unique_questions = []
    
    for q in all_questions:
        qid = q['question_id']
        if qid not in seen_ids:
            seen_ids.add(qid)
            unique_questions.append(q)
    
    logger.info(f"题库加载完成: {len(unique_questions)} 个唯一问题")
    return unique_questions, dimension_questions


def load_models_from_config_manager(config_manager: ConfigManager) -> list:
    """从配置管理器加载模型列表"""
    model_entries = config_manager.get_models_to_test()
    model_list = []
    
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        api_config = entry.get('api_config', {})
        if not api_config:
            continue
        
        model_id = api_config.get('model_id')
        provider = api_config.get('provider', '')
        
        model_item = {
            'model_name': entry.get('model_name'),
            'model': model_id,
            'is_reasoning': entry.get('is_reasoning', False),
            'image_text_input': entry.get('image_text_input', False),
            'provider': provider,
            'api_config': api_config,
        }
        if model_item['model']:
            model_list.append(model_item)
    
    return model_list


def main():
    load_project_env()
    
    parser = argparse.ArgumentParser(description="LiveSecBench评估框架主程序")
    parser.add_argument('--config', type=str, required=True, help='评测任务的YAML配置文件路径')
    args = parser.parse_args()
    config_path = args.config
    
    logger.info(f"LiveSecBench 评测框架启动 - 配置文件: {config_path}")
    
    from livesecbench.core.task_manager import TaskManager
    task_manager = TaskManager()
    logger.info(f"任务ID: {task_manager.task_id}")
    
    try:
        config_manager = ConfigManager(config_path)
        task_manager.set_config_info(config_path, config_manager.get_eval_run_name())
        
        validation_errors = config_manager.validate_config()
        if validation_errors:
            logger.error("配置验证失败，发现以下问题：")
            for error in validation_errors:
                logger.error(f"  - {error}")
            logger.error("请修复配置文件后重试")
            return
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return
    
    question_selection = config_manager.get_question_selection()
    dimensions = config_manager.get_dimensions()
    question_set_path = Path(__file__).resolve().parent / 'question_set'
    questions, dimension_questions = load_questions(str(question_set_path), question_selection)
    
    model_list = load_models_from_config_manager(config_manager)
    if model_list:
        task_manager.set_models([m.get('model_id') or m.get('model') for m in model_list])
    task_manager.set_dimensions(dimensions)
    task_manager.set_question_counts({dim: len(questions) for dim, questions in dimension_questions.items()})

    if not model_list:
        logger.warning("配置中未找到模型列表，跳过模型回答生成步骤")
    else:
        logger.info(f"开始获取模型回答，共 {len(model_list)} 个模型")
        asyncio.run(batch_gen_llm_answer(model_list, questions, config_manager=config_manager, task_id=task_manager.task_id))

    logger.info("开始评分流程")
    asyncio.run(launch_evaluation(config_manager, dimension_questions, task_manager=task_manager))
    
    logger.info("生成排名")
    rank(config_manager, dimensions, task_manager=task_manager)
    
    storage_tables = config_manager.get_storage_tables()
    from livesecbench.storage.sqlite_storage import SQLiteStorage
    storage = SQLiteStorage(
        db_path=config_manager.get_storage_db_path(),
        model_outputs_table=storage_tables['model_outputs_table'],
        pk_results_table=storage_tables['pk_results_table'],
        task_id=task_manager.task_id,
    )
    storage.save_task_info(task_manager.task_id, task_manager.get_task_info())
    
    logger.info("生成测试报告")
    asyncio.run(generate_report(config_manager=config_manager, dimensions=dimensions, task_manager=task_manager))
    
    storage.save_task_info(task_manager.task_id, task_manager.get_task_info())
    logger.info(f"LiveSecBench 评测流程全部完成 - 任务ID: {task_manager.task_id}")


if __name__ == "__main__":
    main()