"""
配对策略模块
提供瑞士制、单循环赛、随机配对等多种策略
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set
import random


class PairingStrategy(ABC):
    """配对策略基类: 定义配对策略接口"""

    @abstractmethod
    def generate_pairs(
        self,
        models: List[str],
        current_ratings: Dict[str, float],
        match_history: Dict[str, Set[str]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """生成模型对战配对"""
        pass


class SwissPairingStrategy(PairingStrategy):
    """瑞士制配对策略: 优先配对评分接近的模型，避免重复对战"""

    def generate_pairs(
        self,
        models: List[str],
        current_ratings: Dict[str, float],
        match_history: Dict[str, Set[str]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """瑞士制配对：优先配对评分接近且未对战过的模型"""
        sorted_models = sorted(
            models,
            key=lambda m: current_ratings.get(m, 0),
            reverse=True
        )

        paired = set()
        pairs = []

        for i, model_a in enumerate(sorted_models):
            if model_a in paired:
                continue

            best_opponent = None
            min_rating_diff = float('inf')

            for j in range(i + 1, len(sorted_models)):
                model_b = sorted_models[j]

                if model_b in paired:
                    continue

                rating_diff = abs(
                    current_ratings.get(model_a, 0) -
                    current_ratings.get(model_b, 0)
                )

                if model_b not in match_history.get(model_a, set()):
                    if rating_diff < min_rating_diff:
                        best_opponent = model_b
                        min_rating_diff = rating_diff

            if best_opponent is None:
                for j in range(i + 1, len(sorted_models)):
                    model_b = sorted_models[j]
                    if model_b not in paired:
                        best_opponent = model_b
                        break

            if best_opponent:
                pairs.append((model_a, best_opponent))
                paired.add(model_a)
                paired.add(best_opponent)

                if model_a not in match_history:
                    match_history[model_a] = set()
                if best_opponent not in match_history:
                    match_history[best_opponent] = set()

                match_history[model_a].add(best_opponent)
                match_history[best_opponent].add(model_a)

        return pairs


class RoundRobinPairingStrategy(PairingStrategy):
    """单循环赛配对策略: 每个模型与其他所有模型对战一次"""

    def __init__(self):
        self._all_pairs_generated = False
        self._all_pairs = []
        self._current_index = 0

    def generate_pairs(
        self,
        models: List[str],
        current_ratings: Dict[str, float],
        match_history: Dict[str, Set[str]],
        batch_size: int = None,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """单循环配对：生成所有可能的配对"""
        if not self._all_pairs_generated:
            self._all_pairs = []
            for i, model_a in enumerate(models):
                for model_b in models[i+1:]:
                    self._all_pairs.append((model_a, model_b))

            random.shuffle(self._all_pairs)
            self._all_pairs_generated = True
            self._current_index = 0

        if batch_size is None:
            batch = self._all_pairs[self._current_index:]
            self._current_index = len(self._all_pairs)
        else:
            start = self._current_index
            end = min(start + batch_size, len(self._all_pairs))
            batch = self._all_pairs[start:end]
            self._current_index = end

        for model_a, model_b in batch:
            if model_a not in match_history:
                match_history[model_a] = set()
            if model_b not in match_history:
                match_history[model_b] = set()

            match_history[model_a].add(model_b)
            match_history[model_b].add(model_a)

        return batch

    def has_more_pairs(self) -> bool:
        """是否还有更多配对"""
        return self._current_index < len(self._all_pairs)

    def reset(self):
        """重置策略状态"""
        self._all_pairs_generated = False
        self._all_pairs = []
        self._current_index = 0


class RandomPairingStrategy(PairingStrategy):
    """随机配对策略: 完全随机配对，不考虑评分和历史"""

    def generate_pairs(
        self,
        models: List[str],
        current_ratings: Dict[str, float],
        match_history: Dict[str, Set[str]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        随机配对：完全随机打乱后配对
        """
        shuffled = models.copy()
        random.shuffle(shuffled)

        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            model_a = shuffled[i]
            model_b = shuffled[i + 1]
            pairs.append((model_a, model_b))
            
            if model_a not in match_history:
                match_history[model_a] = set()
            if model_b not in match_history:
                match_history[model_b] = set()
            
            match_history[model_a].add(model_b)
            match_history[model_b].add(model_a)
        
        return pairs
