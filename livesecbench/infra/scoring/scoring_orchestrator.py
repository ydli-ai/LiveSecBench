"""
评分编排器模块
协调配对策略和评分算法，编排整个评分流程
"""

import asyncio
import random
import time
from pathlib import Path
from typing import List, Dict, Callable, Any, Tuple
from collections import defaultdict

import httpx
import pandas as pd

from .pairing_strategies import PairingStrategy
from .rating_algorithms import RatingAlgorithm
from .convergence_detector import ConvergenceDetector


class ScoringOrchestrator:
    """评分编排器: 协调配对策略、执行PK对战、更新评分、管理历史"""
    
    def __init__(
        self,
        pairing_strategy: PairingStrategy,
        rating_algorithm: RatingAlgorithm,
        pk_runner: Callable,
        fetch_model_result: Callable,
        logger: Any,
        convergence_detector: ConvergenceDetector = None
    ):
        self.pairing_strategy = pairing_strategy
        self.rating_algorithm = rating_algorithm
        self.pk_runner = pk_runner
        self.fetch_model_result = fetch_model_result
        self.logger = logger
        self.convergence_detector = convergence_detector
    
    async def run_scoring(
        self,
        evaluation_dimension: str,
        models: List[str],
        questions: List[dict],
        num_rounds: int,
        max_workers: int = 10,
        output_dir: Path = None,
        filename_templates: Dict[str, str] = None,
    ) -> Dict:
        """运行评分流程"""
        self.logger.info(f"开始 {evaluation_dimension} 维度评分，模型数: {len(models)}, 题目数: {len(questions)}")
        
        if not questions:
            self.logger.warning(f"{evaluation_dimension}: 题目列表为空，跳过评分")
            return {
                'elo_scores': {},
                'history_path': None,
                'result_path': None,
                'record_path': None,
            }
        
        current_ratings = {
            model: self.rating_algorithm.get_initial_rating() 
            for model in models
        }
        match_history = defaultdict(set)
        
        all_pair_results = []
        history_snapshots = []
        question_groups = self._split_questions(questions, num_rounds)
        converged_early = False
        for round_idx, question_set in enumerate(question_groups):
            if not question_set:
                self.logger.info(f"{evaluation_dimension}: 第 {round_idx + 1} 轮题目为空，跳过")
                continue
            
            self.logger.info(f"{evaluation_dimension} 第 {round_idx + 1}/{num_rounds} 轮 [题目数: {len(question_set)}]")
            
            pairs = self.pairing_strategy.generate_pairs(
                models=models,
                current_ratings=current_ratings,
                match_history=match_history
            )
            
            if not pairs:
                self.logger.warning(f"{evaluation_dimension}: 第 {round_idx + 1} 轮无法生成配对，跳过")
                continue
            
            self.logger.info(f"本轮配对数: {len(pairs)}")
            
            round_results = await self._execute_round(
                pairs=pairs,
                questions=question_set,
                evaluation_dimension=evaluation_dimension,
                current_ratings=current_ratings,
                max_workers=max_workers
            )
            
            all_pair_results.extend(round_results)
            history_snapshots.append(pd.Series(current_ratings.copy()))
            self._log_current_ranking(current_ratings, round_idx + 1, evaluation_dimension)
            
            if self.convergence_detector:
                if self.convergence_detector.check_convergence(current_ratings, round_idx):
                    converged_early = True
                    convergence_info = self.convergence_detector.get_convergence_info()
                    self.logger.info(
                        f"{evaluation_dimension} - 评分已收敛，提前结束评测 "
                        f"(总轮数: {round_idx + 1}/{num_rounds}, "
                        f"稳定轮数: {convergence_info['stable_rounds']}, "
                        f"变化率: {convergence_info['rating_change_rate']:.4f}, "
                        f"稳定性: {convergence_info['ranking_stability']:.4f})"
                    )
                    break
        
        result = self._save_results(
            evaluation_dimension=evaluation_dimension,
            final_ratings=current_ratings,
            history_snapshots=history_snapshots,
            all_pair_results=all_pair_results,
            output_dir=output_dir,
            filename_templates=filename_templates
        )
        
        if self.convergence_detector:
            result['convergence_info'] = self.convergence_detector.get_convergence_info()
            result['converged_early'] = converged_early
        
        return result
    
    async def _execute_round(
        self,
        pairs: List[Tuple[str, str]],
        questions: List[dict],
        evaluation_dimension: str,
        current_ratings: Dict[str, float],
        max_workers: int
    ) -> List[dict]:
        """执行一轮PK对战"""
        pk_tasks = []
        for model_a, model_b in pairs:
            for question_data in questions:
                category = question_data.get('category') or question_data.get('dimension')
                prompt = question_data.get('prompt') or question_data.get('question') or question_data.get('question_text')
                
                if not prompt:
                    self.logger.warning(f"题目缺少prompt/question/question_text字段，跳过: {question_data.keys()}")
                    continue
                
                result_a = self.fetch_model_result(model_a, category, prompt) or {}
                result_b = self.fetch_model_result(model_b, category, prompt) or {}
                
                pk_tasks.append({
                    'evaluation_dimension': evaluation_dimension,
                    'category': category,
                    'prompt': prompt,
                    'model_a': model_a,
                    'model_b': model_b,
                    'reasoning_a': result_a.get('reasoning'),
                    'reasoning_b': result_b.get('reasoning'),
                    'answer_a': result_a.get('answer'),
                    'answer_b': result_b.get('answer'),
                    'true_answer': result_a.get('true_answer') if category == '事实性' else None,
                })
        
        pk_results = await self._execute_pks(pk_tasks, max_workers)
        updated_count = 0
        for pk_result in pk_results:
            if not pk_result:
                self.logger.warning(f"PK结果为空，跳过评分更新")
                continue
            
            if 'winner' not in pk_result:
                self.logger.warning(f"PK结果缺少winner字段: {pk_result.keys()}")
                continue
            
            model_a = pk_result.get('A') or pk_result.get('model_a')
            model_b = pk_result.get('B') or pk_result.get('model_b')
            winner = pk_result['winner']
            
            if not model_a or not model_b:
                self.logger.warning(f"PK结果缺少模型ID: A={model_a}, B={model_b}")
                continue
            
            old_rating_a = current_ratings.get(model_a, self.rating_algorithm.get_initial_rating())
            old_rating_b = current_ratings.get(model_b, self.rating_algorithm.get_initial_rating())
            
            new_rating_a, new_rating_b = self.rating_algorithm.update_ratings(
                model_a=model_a,
                model_b=model_b,
                winner=winner,
                current_ratings=current_ratings
            )
            
            current_ratings[model_a] = new_rating_a
            current_ratings[model_b] = new_rating_b
            
            updated_count += 1
            self.logger.debug(
                f"评分更新: {model_a}({old_rating_a:.1f}->{new_rating_a:.1f}) vs "
                f"{model_b}({old_rating_b:.1f}->{new_rating_b:.1f}), 胜者: {winner}"
            )
        
        self.logger.info(f"本轮共更新 {updated_count}/{len(pk_results)} 个PK结果的评分")
        
        return pk_results
    
    async def _execute_pks(
        self,
        pk_tasks: List[dict],
        max_workers: int
    ) -> List[dict]:
        """并发执行PK任务"""
        if not pk_tasks:
            return []
        
        semaphore = asyncio.Semaphore(max_workers)
        timeout_config = httpx.Timeout(120.0, connect=30.0)
        
        async def run_single_pk(task):
            """执行单个PK任务"""
            async with semaphore:
                try:
                    winner, is_new, consume_time, content, pk_result = await self.pk_runner(
                        evaluation_dimension=task['evaluation_dimension'],
                        category=task['category'],
                        question=task['prompt'],
                        model_A=task['model_a'],
                        model_B=task['model_b'],
                        reasoning_A=task['reasoning_a'],
                        reasoning_B=task['reasoning_b'],
                        answer_A=task['answer_a'],
                        answer_B=task['answer_b'],
                        true_answer=task['true_answer'],
                    )
                    if pk_result and isinstance(pk_result, dict):
                        return pk_result
                    elif winner:
                        return {
                            'A': task['model_a'],
                            'B': task['model_b'],
                            'winner': winner,
                        }
                    return None
                except Exception as e:
                    self.logger.error(f'PK执行异常: {e}')
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return None
        
        results = await asyncio.gather(
            *[run_single_pk(task) for task in pk_tasks],
            return_exceptions=True
        )
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f'PK任务异常: {result}')
            elif result:
                valid_results.append(result)
        
        return valid_results
    
    def _split_questions(
        self, 
        questions: List[dict], 
        num_groups: int
    ) -> List[List[dict]]:
        """将题目分组"""
        questions_list = list(questions)
        random.shuffle(questions_list)
        
        if num_groups <= 0:
            return [questions_list]
        
        total = len(questions_list)
        base_size = total // num_groups
        remainder = total % num_groups
        
        groups = []
        start = 0
        for i in range(num_groups):
            end = start + base_size + (1 if i < remainder else 0)
            groups.append(questions_list[start:end])
            start = end
        
        return groups
    
    def _log_current_ranking(
        self, 
        ratings: Dict[str, float], 
        round_num: int,
        dimension: str
    ):
        """记录当前排名"""
        self.logger.info(f"{dimension} - 第 {round_num} 轮后排名:")
        for rank, (model, rating) in enumerate(
            sorted(ratings.items(), key=lambda x: x[1], reverse=True), 
            1
        ):
            self.logger.info(f"  {rank}. {model} - {rating:.2f}")
    
    def _save_results(
        self,
        evaluation_dimension: str,
        final_ratings: Dict[str, float],
        history_snapshots: List[pd.Series],
        all_pair_results: List[dict],
        output_dir: Path = None,
        filename_templates: Dict[str, str] = None
    ) -> Dict:
        """保存评分结果"""
        if output_dir is None:
            day_tag = time.strftime('%Y_%m_%d', time.localtime())
            output_dir = Path("results") / day_tag / "elo_results"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename_templates is None:
            filename_templates = {
                'history': "{dimension}_elo_history_{timestamp}.csv",
                'result': "{dimension}_elo_raw_{timestamp}.csv",
                'record': "{dimension}_pk_details_{timestamp}.xlsx"
            }
        
        timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        
        format_params = {
            'dimension': evaluation_dimension,
            'timestamp': timestamp,
            'task_id': getattr(self, '_task_id', ''),
        }
        
        if history_snapshots:
            history_df = pd.DataFrame(history_snapshots)
            history_filename = filename_templates['history'].format(**format_params)
            history_path = output_dir / history_filename
            history_df.to_csv(history_path, index=False)
            self.logger.info(f"已保存评分历史: {history_path}")
        else:
            history_path = None
        
        ranking_records = []
        for rank, model in enumerate(
            sorted(final_ratings, key=final_ratings.get, reverse=True), 
            1
        ):
            ranking_records.append({
                '排名': rank,
                '模型': model,
                'elo': final_ratings[model],
            })
        
        result_filename = filename_templates['result'].format(**format_params)
        result_path = output_dir / result_filename
        pd.DataFrame(ranking_records).to_csv(result_path, index=False)
        self.logger.info(f"已保存排名结果: {result_path}")
        
        if all_pair_results:
            record_filename = filename_templates['record'].format(**format_params)
            record_path = output_dir / record_filename
            pd.DataFrame(all_pair_results).to_excel(record_path, index=False)
            self.logger.info(f"已保存详细记录: {record_path}")
        else:
            record_path = None
        
        return {
            'elo_scores': final_ratings,
            'history_path': str(history_path) if history_path else None,
            'result_path': str(result_path),
            'record_path': str(record_path) if record_path else None,
            'pair_results': all_pair_results,
        }
