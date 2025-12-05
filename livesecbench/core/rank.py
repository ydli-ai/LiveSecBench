"""
模型排名生成模块
负责从ELO评分结果生成最终的模型排名和统计数据。
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from livesecbench.infra.config import ConfigManager
from livesecbench.utils.logger import get_logger

logger = get_logger(__name__)


class ELOScorer:
    """ELO评分器: 负责ELO分数到百分制的转换"""
    
    def __init__(self, base_elo: int = 1500, k_factor: int = 32):
        self.base_elo = base_elo
        self.k_factor = k_factor

    def elo_to_percentile(self, elo_scores: Dict[str, float], method: str = "win_probability") -> Dict[str, float]:
        """转换ELO分数为百分制，支持 linear/win_probability/percentile_rank 三种方法"""
        if method == "linear":
            return self._linear_normalization(elo_scores)
        elif method == "win_probability":
            return self._win_probability_method(elo_scores)
        elif method == "percentile_rank":
            return self._percentile_rank_method(elo_scores)
        else:
            raise ValueError(f"不支持的转换方法: {method}")

    def _linear_normalization(self, elo_scores: dict) -> dict:
        """线性归一化方法"""
        min_elo = min(elo_scores.values())
        max_elo = max(elo_scores.values())
        range_elo = max_elo - min_elo
        
        if range_elo == 0:
            return {model: 50.0 for model in elo_scores}
        
        return {
            model: (elo - min_elo) / range_elo * 100
            for model, elo in elo_scores.items()
        }

    def _win_probability_method(self, elo_scores: dict) -> dict:
        """基于对平均ELO模型的预期胜率"""
        avg_elo = np.mean(list(elo_scores.values()))
        result = {}
        for model, elo in elo_scores.items():
            expected_win_rate = 1 / (1 + 10 ** ((avg_elo - elo) / 400))
            result[model] = expected_win_rate * 100
        return result

    def _percentile_rank_method(self, elo_scores: dict) -> dict:
        """百分位排名方法"""
        sorted_models = sorted(elo_scores.items(), key=lambda x: x[1])
        n = len(sorted_models)
        result = {}
        for rank, (model, elo) in enumerate(sorted_models, 1):
            percentile = (rank - 0.5) / n * 100
            result[model] = percentile
        return result


def _load_previous_rankings(previous_ranking_file: Optional[Path] = None) -> Dict[str, int]:
    """加载上一期的排名数据"""
    if previous_ranking_file is None:
        # 自动查找最近的排名文件（排除当前月份）
        results_base = Path("results")
        if not results_base.exists():
            logger.info("未找到results目录，无法加载历史排名")
            return {}
        
        current_month = time.strftime('%Y-%m', time.localtime())
        ranking_files = []
        
        for date_dir in results_base.iterdir():
            if not date_dir.is_dir():
                continue
            
            # 查找该目录下的models文件
            models_files = list(date_dir.glob("*-models*.csv"))
            for f in models_files:
                # 从文件名提取月份标签
                file_month = f.name.split('-models')[0]
                if file_month != current_month:
                    ranking_files.append(f)
        
        if not ranking_files:
            logger.info("未找到历史排名文件")
            return {}
        
        # 选择最新的文件
        previous_ranking_file = max(ranking_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"加载历史排名: {previous_ranking_file}")
    
    try:
        df = pd.read_csv(previous_ranking_file)
        if 'rank' not in df.columns or 'model_name' not in df.columns:
            logger.warning(f"历史排名文件缺少必要的列: {previous_ranking_file}")
            return {}
        
        # 建立模型名到排名的映射
        rank_map = {}
        for _, row in df.iterrows():
            model_name = row['model_name']
            rank = row['rank']
            if pd.notna(model_name) and pd.notna(rank):
                rank_map[model_name] = int(rank)
        
        logger.info(f"成功加载 {len(rank_map)} 个模型的历史排名")
        return rank_map
        
    except Exception as e:
        logger.error(f"加载历史排名文件出错: {e}")
        return {}


def rank(
    config_manager: ConfigManager, 
    dimensions: Optional[List[str]] = None,
    task_manager: Optional[Any] = None,
    previous_ranking_file: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """生成模型排名和统计数据"""
    if dimensions is None:
        dimensions = config_manager.get_dimensions()
    
    # 加载上一期的排名数据
    previous_rankings = _load_previous_rankings(previous_ranking_file)
    
    elo_results_dir = config_manager.get_elo_results_dir()
    
    if not elo_results_dir.exists():
        logger.warning(f"ELO结果目录不存在: {elo_results_dir}，创建空结果文件")
        results_dir = Path("results") / time.strftime('%Y_%m_%d', time.localtime())
        results_dir.mkdir(parents=True, exist_ok=True)
        month_tag = time.strftime('%Y-%m', time.localtime())
        models_file = results_dir / f"{month_tag}-models.csv"
        stats_file = results_dir / f"{month_tag}-stats.csv"
        pd.DataFrame(columns=['rank', 'model_name', 'provider', 'overall_score']).to_csv(models_file, index=False)
        pd.DataFrame(columns=['totalModels', 'averageScore', 'dimensions', 'lastUpdate']).to_csv(stats_file, index=False)
        return models_file, stats_file
    
    logger.info(f"读取ELO结果: {elo_results_dir}")
    
    scorer = ELOScorer()
    dimension_name_map = config_manager.get_dimension_name_map()

    all_dimension_data = {}
    
    for dimension_dir in elo_results_dir.iterdir():
        if not dimension_dir.is_dir():
            continue
        
        dimension_name = dimension_dir.name
        elo_raw_files = list(dimension_dir.glob(f"{dimension_name}_elo_raw_*.csv"))
        
        if not elo_raw_files:
            logger.warning(f"维度 {dimension_name} 未找到ELO得分文件")
            continue
        
        file_path = max(elo_raw_files, key=lambda p: p.stat().st_mtime)
        
        try:
            df = pd.read_csv(file_path)
            dimension_en = dimension_name_map.get(dimension_name, dimension_name)
            
            if '模型' not in df.columns or 'elo' not in df.columns:
                logger.warning(f"文件 {file_path.name} 缺少必要的列，跳过")
                continue
            
            elo_results = {row['模型']: row['elo'] for _, row in df.iterrows()}
            percentile_scores = scorer.elo_to_percentile(elo_results, "win_probability")
            
            all_dimension_data[dimension_en] = {
                model: round(score, 2) 
                for model, score in percentile_scores.items()
            }
            
            logger.info(f"{dimension_en}: {len(percentile_scores)} 个模型")
            
        except Exception as e:
            logger.error(f"处理文件 {file_path.name} 时出错: {e}")
            continue
    
    if not all_dimension_data:
        logger.warning("未找到有效的ELO结果文件，创建空结果文件")
        results_dir = Path("results") / time.strftime('%Y_%m_%d', time.localtime())
        results_dir.mkdir(parents=True, exist_ok=True)
        month_tag = time.strftime('%Y-%m', time.localtime())
        models_file = results_dir / f"{month_tag}-models.csv"
        stats_file = results_dir / f"{month_tag}-stats.csv"
        pd.DataFrame(columns=['rank', 'model_name', 'provider', 'overall_score']).to_csv(models_file, index=False)
        pd.DataFrame(columns=['totalModels', 'averageScore', 'dimensions', 'lastUpdate']).to_csv(stats_file, index=False)
        return models_file, stats_file
    
    models_to_test = config_manager.get_models_to_test()
    model_name_map = config_manager.get_model_name_map()
    ranking_data = []
    today = time.strftime('%Y/%m/%d', time.localtime())
    
    for model_config in models_to_test:
        api_config = model_config.get('api_config', {})
        if not api_config:
            continue
        
        model_id = api_config.get('model_id')
        if not model_id:
            continue
        
        provider = api_config.get('provider', '')
        display_name = model_name_map.get(model_id, model_config.get('model_name', model_id))
        model_record = {
            'model_name': display_name,
            'model_id': model_id,
            'provider': provider,
        }
        
        for dimension in dimensions:
            if dimension in all_dimension_data:
                dimension_scores = all_dimension_data[dimension]
                score = None
                candidates = dict.fromkeys([
                    model_id,
                    display_name,
                ]).keys()
                for candidate in candidates:
                    if candidate in dimension_scores:
                        score = dimension_scores[candidate]
                        break
                model_record[dimension] = score if score is not None else ''
            else:
                model_record[dimension] = ''
        
        model_record.update({
            "evaluation_date": today,
            "is_open_source": model_config.get('open_source', False),
            "publish_time": model_config.get('publish_time', ''),
            "location": model_config.get('location', ''),
            "model_url": model_config.get('model_detail_url', ''),
        })
        ranking_data.append(model_record)
    
    df = pd.DataFrame(ranking_data)
    score_columns = [d for d in dimensions if d != 'reasoning' and d in df.columns]
    
    if score_columns:
        df['overall_score'] = df[score_columns].apply(
            lambda row: row[row != ''].astype(float).mean() if any(row != '') else 0, 
            axis=1
        ).round(2)
    else:
        df['overall_score'] = 0.0
    
    df = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    # 计算排名变化（基于两次排名的交集）
    if previous_rankings:
        current_models = set(df['model_name'].tolist())
        previous_models = set(previous_rankings.keys())
        common_models = current_models & previous_models
        
        if common_models:
            logger.info(f"两次排名的交集包含 {len(common_models)} 个模型")
            
            # 在交集中，按本次overall_score排序，得到交集内排名
            df_common = df[df['model_name'].isin(common_models)].copy()
            df_common = df_common.sort_values('overall_score', ascending=False).reset_index(drop=True)
            current_intersection_rank = {
                row['model_name']: idx + 1 
                for idx, row in df_common.iterrows()
            }
            
            previous_common = [(model, previous_rankings[model]) for model in common_models]
            previous_common.sort(key=lambda x: x[1])
            previous_intersection_rank = {
                model: idx + 1 
                for idx, (model, _) in enumerate(previous_common)
            }
            
            rank_changes = []
            for _, row in df.iterrows():
                model_name = row['model_name']
                
                if model_name in common_models:
                    prev_rank = previous_intersection_rank[model_name]
                    curr_rank = current_intersection_rank[model_name]
                    rank_change = prev_rank - curr_rank
                    rank_changes.append(rank_change)
                    
                    if rank_change != 0:
                        logger.debug(
                            f"{model_name}: 交集内排名 第{prev_rank}名 -> 第{curr_rank}名，"
                            f"变化: {rank_change:+d}"
                        )
                else:
                    rank_changes.append(0)
                    logger.debug(f"{model_name}: 新上榜模型")
            
            df['rank_change'] = rank_changes
            changed_count = sum(1 for x in rank_changes if x != 0)
            logger.info(f"已计算排名变化: {changed_count} 个模型排名变化, {len(current_models - common_models)} 个新上榜")
        else:
            df['rank_change'] = 0
            logger.info("两次排名无交集，所有模型视为新上榜")
    else:
        df['rank_change'] = 0
        logger.info("未找到历史排名数据，所有模型的rank_change设为0")
    
    base_columns = ['rank', 'model_name', 'model_id', 'provider', 'overall_score']
    other_columns = [col for col in df.columns if col not in base_columns]
    df = df[base_columns + other_columns]
    
    day_tag = time.strftime('%Y_%m_%d', time.localtime())
    month_tag = time.strftime('%Y-%m', time.localtime())
    results_dir = Path("results") / day_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if task_manager:
        models_file = results_dir / f"{month_tag}-models_{task_manager.task_id}.csv"
        stats_file = results_dir / f"{month_tag}-stats_{task_manager.task_id}.csv"
    else:
        models_file = results_dir / f"{month_tag}-models.csv"
        stats_file = results_dir / f"{month_tag}-stats.csv"
    
    df.to_csv(models_file, index=False)
    
    stats_df = pd.DataFrame([{
        "totalModels": len(df),
        "averageScore": df['overall_score'].mean().round(2),
        "dimensions": len(score_columns),
        "lastUpdate": today,
        "monthlyIncrease": 0,
        "scoreImprovement": 0,
        "hasHistory": False,
    }])
    stats_df.to_csv(stats_file, index=False)
    
    if task_manager:
        task_manager.add_result_file(str(models_file), 'models_ranking')
        task_manager.add_result_file(str(stats_file), 'stats')
    
    logger.info(f"排名已生成: {len(df)} 个模型")
    for _, row in df.head(5).iterrows():
        logger.info(f"  {int(row['rank'])}. {row['model_name']} - {row['overall_score']} ({row['provider']})")
    
    return models_file, stats_file
