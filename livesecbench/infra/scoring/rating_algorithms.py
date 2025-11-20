"""
评分算法模块
提供ELO、Glicko等多种评分系统
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import math


class RatingAlgorithm(ABC):
    """评分算法基类: 定义评分算法接口"""
    
    @abstractmethod
    def update_ratings(
        self,
        model_a: str,
        model_b: str,
        winner: str,
        current_ratings: Dict[str, float],
        **kwargs
    ) -> Tuple[float, float]:
        """更新两个模型的评分"""
        pass
    
    @abstractmethod
    def get_initial_rating(self) -> float:
        """获取初始评分"""
        pass
    
    @abstractmethod
    def get_expected_score(
        self, 
        rating_a: float, 
        rating_b: float
    ) -> float:
        """计算期望得分"""
        pass


class ELORatingAlgorithm(RatingAlgorithm):
    """ELO评分算法: 广泛使用的评分系统，简单高效"""
    
    def __init__(
        self, 
        init_rating: float = 1500, 
        k_factor: float = 32, 
        logistic_constant: float = 400
    ):
        self.init_rating = init_rating
        self.k_factor = k_factor
        self.logistic_constant = logistic_constant
    
    def get_initial_rating(self) -> float:
        """获取初始评分"""
        return self.init_rating
    
    def get_expected_score(
        self, 
        rating_a: float, 
        rating_b: float
    ) -> float:
        """
        计算期望得分
        
        使用逻辑函数计算模型A对模型B的期望胜率
        
        公式: E_a = 1 / (1 + 10^((R_b - R_a) / logistic_constant))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / self.logistic_constant))
    
    def update_ratings(
        self,
        model_a: str,
        model_b: str,
        winner: str,
        current_ratings: Dict[str, float],
        **kwargs
    ) -> Tuple[float, float]:
        """ELO评分更新：new_rating = old_rating + K * (actual - expected)"""
        rating_a = current_ratings.get(model_a, self.init_rating)
        rating_b = current_ratings.get(model_b, self.init_rating)
        
        expected_a = self.get_expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        if winner == model_a:
            actual_a, actual_b = 1.0, 0.0
        elif winner == model_b:
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a, actual_b = 0.5, 0.5
        
        new_rating_a = rating_a + self.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_b - expected_b)
        
        return new_rating_a, new_rating_b
    
    def get_win_probability(
        self, 
        rating_a: float, 
        rating_b: float
    ) -> float:
        """计算模型A战胜模型B的概率（与get_expected_score相同）"""
        return self.get_expected_score(rating_a, rating_b)


class GlickoRatingAlgorithm(RatingAlgorithm):
    """Glicko评分算法（预留接口）- ELO的改进版本，增加评分可靠度（RD）概念"""
    
    def __init__(
        self, 
        init_rating: float = 1500,
        init_rd: float = 350,
        c: float = 63.2
    ):
        self.init_rating = init_rating
        self.init_rd = init_rd
        self.c = c
        self._ratings_deviation = {}
    
    def get_initial_rating(self) -> float:
        return self.init_rating
    
    def get_expected_score(
        self, 
        rating_a: float, 
        rating_b: float
    ) -> float:
        """
        计算期望得分（简化版本，未考虑RD）
        """
        q = math.log(10) / 400
        g_rd = 1 / math.sqrt(1 + 3 * q**2 * self.init_rd**2 / math.pi**2)
        return 1 / (1 + 10 ** (-g_rd * (rating_a - rating_b) / 400))
    
    def update_ratings(
        self,
        model_a: str,
        model_b: str,
        winner: str,
        current_ratings: Dict[str, float],
        **kwargs
    ) -> Tuple[float, float]:
        """
        Glicko评分更新（简化实现）
        """
        rating_a = current_ratings.get(model_a, self.init_rating)
        rating_b = current_ratings.get(model_b, self.init_rating)
        
        expected_a = self.get_expected_score(rating_a, rating_b)
        
        if winner == model_a:
            actual_a = 1.0
        elif winner == model_b:
            actual_a = 0.0
        else:
            actual_a = 0.5
        
        k = 32
        new_rating_a = rating_a + k * (actual_a - expected_a)
        new_rating_b = rating_b + k * ((1 - actual_a) - (1 - expected_a))
        
        return new_rating_a, new_rating_b

