"""
收敛性检测器
判断评分系统是否已收敛，可提前结束评测
"""

from typing import List, Dict
import numpy as np


class ConvergenceDetector:
    """收敛性检测器: 通过评分变化率、排名稳定性等指标判断是否收敛"""
    
    def __init__(
        self,
        threshold: float = 0.01,
        min_stable_rounds: int = 3,
        min_rounds: int = 5,
        ranking_stability_weight: float = 0.5,
        rating_change_weight: float = 0.5
    ):
        self.threshold = threshold
        self.min_stable_rounds = min_stable_rounds
        self.min_rounds = min_rounds
        self.ranking_stability_weight = ranking_stability_weight
        self.rating_change_weight = rating_change_weight
        self.rating_history: List[Dict[str, float]] = []
        self.ranking_history: List[List[str]] = []
        self.stable_rounds_count = 0
        
    def check_convergence(
        self, 
        current_ratings: Dict[str, float], 
        current_round: int
    ) -> bool:
        """检查是否收敛"""
        self.rating_history.append(dict(current_ratings))
        current_ranking = self._get_ranking(current_ratings)
        self.ranking_history.append(current_ranking)
        
        if current_round < self.min_rounds:
            return False
        
        if len(self.rating_history) < 2:
            return False
        
        rating_change_rate = self._calculate_rating_change_rate()
        ranking_stability = self._calculate_ranking_stability()
        
        convergence_score = (
            ranking_stability * self.ranking_stability_weight +
            (1 - rating_change_rate) * self.rating_change_weight
        )
        
        is_stable = convergence_score > (1 - self.threshold)
        
        if is_stable:
            self.stable_rounds_count += 1
        else:
            self.stable_rounds_count = 0
        
        return self.stable_rounds_count >= self.min_stable_rounds
    
    def _calculate_rating_change_rate(self) -> float:
        """计算评分变化率（0-1之间，越小越稳定）"""
        if len(self.rating_history) < 2:
            return 1.0
        
        prev_ratings = self.rating_history[-2]
        curr_ratings = self.rating_history[-1]
        
        changes = []
        for model in curr_ratings:
            if model in prev_ratings:
                prev_rating = prev_ratings[model]
                curr_rating = curr_ratings[model]
                
                if prev_rating == 0:
                    change_rate = 0 if curr_rating == 0 else 1.0
                else:
                    change_rate = abs(curr_rating - prev_rating) / abs(prev_rating)
                
                changes.append(change_rate)
        
        if not changes:
            return 0.0
        
        return np.mean(changes)
    
    def _calculate_ranking_stability(self) -> float:
        """计算排名稳定性（0-1之间，越大越稳定）"""
        if len(self.ranking_history) < 2:
            return 0.0
        
        prev_ranking = self.ranking_history[-2]
        curr_ranking = self.ranking_history[-1]
        
        n = len(curr_ranking)
        if n == 0:
            return 1.0
        
        unchanged_count = 0
        for i, model in enumerate(curr_ranking):
            if i < len(prev_ranking) and prev_ranking[i] == model:
                unchanged_count += 1
        
        stability = unchanged_count / n
        top_k = min(3, n)
        top_unchanged = sum(
            1 for i in range(top_k) 
            if i < len(prev_ranking) and curr_ranking[i] == prev_ranking[i]
        )
        top_stability = top_unchanged / top_k if top_k > 0 else 1.0
        
        return 0.6 * stability + 0.4 * top_stability
    
    def _get_ranking(self, ratings: Dict[str, float]) -> List[str]:
        """根据评分获取排名列表（按评分降序排列）"""
        return sorted(ratings.keys(), key=lambda m: ratings[m], reverse=True)
    
    def get_convergence_info(self) -> Dict:
        """获取收敛信息（包含收敛统计信息的字典）"""
        if len(self.rating_history) < 2:
            return {
                'total_rounds': len(self.rating_history),
                'stable_rounds': self.stable_rounds_count,
                'rating_change_rate': 1.0,
                'ranking_stability': 0.0,
                'is_converged': False
            }
        
        return {
            'total_rounds': len(self.rating_history),
            'stable_rounds': self.stable_rounds_count,
            'rating_change_rate': self._calculate_rating_change_rate(),
            'ranking_stability': self._calculate_ranking_stability(),
            'is_converged': self.stable_rounds_count >= self.min_stable_rounds
        }
    
    def reset(self):
        """重置检测器状态"""
        self.rating_history = []
        self.ranking_history = []
        self.stable_rounds_count = 0


class AdaptiveConvergenceDetector(ConvergenceDetector):
    """自适应收敛性检测器: 根据模型数量调整稳定轮数和收敛阈值"""
    
    def __init__(
        self,
        threshold: float = 0.01,
        min_stable_rounds: int = 3,
        min_rounds: int = 5,
        **kwargs
    ):
        super().__init__(threshold, min_stable_rounds, min_rounds, **kwargs)
        self.adaptive_threshold = threshold
    
    def check_convergence(
        self, 
        current_ratings: Dict[str, float], 
        current_round: int
    ) -> bool:
        """检查是否收敛（自适应版本）"""
        num_models = len(current_ratings)
        if num_models <= 5:
            adaptive_min_stable_rounds = 2
        elif num_models <= 10:
            adaptive_min_stable_rounds = 3
        else:
            adaptive_min_stable_rounds = 4
        
        original_min_stable = self.min_stable_rounds
        self.min_stable_rounds = adaptive_min_stable_rounds
        result = super().check_convergence(current_ratings, current_round)
        self.min_stable_rounds = original_min_stable
        
        return result

