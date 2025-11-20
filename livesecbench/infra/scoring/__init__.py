"""
评分系统基础设施
提供评分算法、配对策略、评分编排器和收敛性检测
"""

from .pairing_strategies import (
    PairingStrategy,
    SwissPairingStrategy,
    RoundRobinPairingStrategy,
    RandomPairingStrategy,
)
from .rating_algorithms import (
    RatingAlgorithm,
    ELORatingAlgorithm,
)
from .scoring_orchestrator import ScoringOrchestrator
from .convergence_detector import (
    ConvergenceDetector,
    AdaptiveConvergenceDetector,
)

__all__ = [
    # 配对策略
    'PairingStrategy',
    'SwissPairingStrategy',
    'RoundRobinPairingStrategy',
    'RandomPairingStrategy',
    # 评分算法
    'RatingAlgorithm',
    'ELORatingAlgorithm',
    # 评分编排器
    'ScoringOrchestrator',
    # 收敛性检测
    'ConvergenceDetector',
    'AdaptiveConvergenceDetector',
]

