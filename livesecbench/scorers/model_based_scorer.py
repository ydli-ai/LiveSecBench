"""
基于模型的评分器
使用模型对战（PK）和ELO评分系统进行模型评测
"""

from __future__ import annotations

from typing import List
from pathlib import Path
import time

from livesecbench.infra.scoring import (
    SwissPairingStrategy,
    RoundRobinPairingStrategy,
    RandomPairingStrategy,
    ELORatingAlgorithm,
    ScoringOrchestrator,
    ConvergenceDetector,
    AdaptiveConvergenceDetector,
)


async def score(
    evaluation_dimension: str,
    dimension_questions: List[dict],
    models: List[str],
    reasoning_models: List[str],
    scorer_params: dict,
    runtime_context: dict,
):
    """使用基于ELO的模型对战评分逻辑"""
    
    logger = runtime_context['logger']
    pk_runner = runtime_context['pk_runner']
    fetch_model_result = runtime_context['fetch_model_result']
    elo_settings = runtime_context['elo_settings']
    
    logger.info(f'{evaluation_dimension} - 开始处理评分')
    
    model_pool = models if evaluation_dimension != 'reasoning' else reasoning_models
    if not model_pool:
        logger.warning(f"{evaluation_dimension}: 未找到可用模型，跳过评分")
        return {}
    
    question_cnt = len(dimension_questions)
    logger.info(f"{evaluation_dimension}: 模型数={len(model_pool)}, 题目数={question_cnt}")
    
    if question_cnt == 0:
        logger.warning(f"{evaluation_dimension}: 题目列表为空，跳过评分")
        return {
            'elo_scores': {},
            'history_path': None,
            'result_path': None,
            'record_path': None,
        }
    
    pairing_config = elo_settings.get('pairing', {})
    strategy_name = pairing_config.get('strategy', 'swiss')
    
    if strategy_name == 'swiss':
        pairing_strategy = SwissPairingStrategy()
        num_rounds = elo_settings.get('swiss_group_num', 5)
        logger.info(f"{evaluation_dimension}: 使用瑞士制配对，轮数={num_rounds}")
        
    elif strategy_name == 'round_robin':
        pairing_strategy = RoundRobinPairingStrategy()
        num_rounds = 1
        logger.info(f"{evaluation_dimension}: 使用单循环赛配对")
        
    elif strategy_name == 'random':
        pairing_strategy = RandomPairingStrategy()
        num_rounds = pairing_config.get('rounds', 5)
        logger.info(f"{evaluation_dimension}: 使用随机配对，轮数={num_rounds}")
        
    else:
        logger.warning(f"{evaluation_dimension}: 不支持的配对策略 '{strategy_name}'，使用默认瑞士制")
        pairing_strategy = SwissPairingStrategy()
        num_rounds = elo_settings.get('swiss_group_num', 5)
    
    rating_config = elo_settings.get('rating', {})
    algorithm_name = rating_config.get('algorithm', 'elo')
    
    if algorithm_name == 'elo':
        rating_algorithm = ELORatingAlgorithm(
            init_rating=elo_settings.get('init_rating', 1500),
            k_factor=elo_settings.get('init_k', 32),
            logistic_constant=elo_settings.get('logistic_constant', 400)
        )
        logger.info(f"{evaluation_dimension}: 使用ELO评分算法")
        
    else:
        logger.warning(f"{evaluation_dimension}: 不支持的评分算法 '{algorithm_name}'，使用默认ELO")
        rating_algorithm = ELORatingAlgorithm(
            init_rating=elo_settings.get('init_rating', 1500),
            k_factor=elo_settings.get('init_k', 32),
            logistic_constant=elo_settings.get('logistic_constant', 400)
        )
    
    convergence_config = elo_settings.get('convergence', {})
    convergence_detector = None
    
    if convergence_config.get('enabled', False):
        detector_type = convergence_config.get('type', 'basic')
        
        if detector_type == 'adaptive':
            convergence_detector = AdaptiveConvergenceDetector(
                threshold=convergence_config.get('threshold', 0.01),
                min_stable_rounds=convergence_config.get('min_stable_rounds', 3),
                min_rounds=convergence_config.get('min_rounds', 5)
            )
            logger.info(f"{evaluation_dimension}: 使用自适应收敛性检测")
        else:
            convergence_detector = ConvergenceDetector(
                threshold=convergence_config.get('threshold', 0.01),
                min_stable_rounds=convergence_config.get('min_stable_rounds', 3),
                min_rounds=convergence_config.get('min_rounds', 5)
            )
            logger.info(f"{evaluation_dimension}: 使用基础收敛性检测")
    
    orchestrator = ScoringOrchestrator(
        pairing_strategy=pairing_strategy,
        rating_algorithm=rating_algorithm,
        pk_runner=pk_runner,
        fetch_model_result=fetch_model_result,
        logger=logger,
        convergence_detector=convergence_detector
    )
    
    config_manager = runtime_context.get('config_manager')
    if config_manager:
        base_output_dir = config_manager.get_elo_results_dir()
    else:
        result_dir = elo_settings.get('result_output_dir', None)
        if result_dir is None or result_dir == 'elo_results':
            day_tag = time.strftime('%Y_%m_%d', time.localtime())
            base_output_dir = Path("results") / day_tag / "elo_results"
        else:
            base_output_dir = Path(result_dir)
    
    output_dir = base_output_dir / evaluation_dimension
    output_dir.mkdir(parents=True, exist_ok=True)
    
    task_manager = runtime_context.get('task_manager')
    task_id = task_manager.task_id if task_manager else None
    
    if task_id:
        filename_templates = {
            'history': elo_settings.get('history_filename_template', '{dimension}_elo_history_{task_id}_{timestamp}.csv'),
            'result': elo_settings.get('result_filename_template', '{dimension}_elo_raw_{task_id}_{timestamp}.csv'),
            'record': elo_settings.get('record_filename_template', '{dimension}_pk_details_{task_id}_{timestamp}.xlsx'),
        }
    else:
        filename_templates = {
            'history': elo_settings.get('history_filename_template', '{dimension}_elo_history_{timestamp}.csv'),
            'result': elo_settings.get('result_filename_template', '{dimension}_elo_raw_{timestamp}.csv'),
            'record': elo_settings.get('record_filename_template', '{dimension}_pk_details_{timestamp}.xlsx'),
        }
    
    if task_manager:
        orchestrator._task_id = task_manager.task_id
    
    result = await orchestrator.run_scoring(
        evaluation_dimension=evaluation_dimension,
        models=model_pool,
        questions=dimension_questions,
        num_rounds=num_rounds,
        max_workers=elo_settings.get('max_workers', 10),
        output_dir=output_dir,
        filename_templates=filename_templates
    )
    
    logger.info(f"{evaluation_dimension}: 评分完成，结果已保存")
    
    return result
