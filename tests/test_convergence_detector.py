"""
ConvergenceDetector单元测试
"""

from livesecbench.infra.scoring.convergence_detector import (
    ConvergenceDetector,
    AdaptiveConvergenceDetector
)


def test_convergence_detector_initialization():
    """测试收敛检测器初始化"""
    detector = ConvergenceDetector(
        threshold=0.01,
        min_stable_rounds=3,
        min_rounds=5
    )
    
    assert detector.threshold == 0.01
    assert detector.min_stable_rounds == 3
    assert detector.min_rounds == 5
    assert len(detector.rating_history) == 0


def test_no_convergence_early_rounds():
    """测试早期轮次不会收敛"""
    detector = ConvergenceDetector(min_rounds=5)
    
    ratings = {'model1': 1500, 'model2': 1500, 'model3': 1500}
    
    # 前5轮不应该收敛
    for round_idx in range(5):
        result = detector.check_convergence(ratings, round_idx)
        assert result is False


def test_convergence_with_stable_ratings():
    """测试评分稳定时的收敛"""
    detector = ConvergenceDetector(
        threshold=0.01,
        min_stable_rounds=2,
        min_rounds=3
    )
    
    # 评分基本不变
    ratings = {'model1': 1500, 'model2': 1600, 'model3': 1400}
    
    # 前3轮建立历史
    for round_idx in range(3):
        detector.check_convergence(ratings, round_idx)
    
    # 第4轮和第5轮应该检测到收敛
    ratings_stable = {'model1': 1500.1, 'model2': 1600.1, 'model3': 1400.1}
    detector.check_convergence(ratings_stable, 3)
    
    ratings_stable = {'model1': 1500.05, 'model2': 1600.05, 'model3': 1400.05}
    result = detector.check_convergence(ratings_stable, 4)
    
    # 应该收敛（连续2轮稳定）
    assert result is True


def test_convergence_with_unstable_ratings():
    """测试评分不稳定时不收敛"""
    detector = ConvergenceDetector(
        threshold=0.01,
        min_stable_rounds=2,
        min_rounds=3
    )
    
    # 评分持续变化
    for round_idx in range(10):
        ratings = {
            'model1': 1500 + round_idx * 50,
            'model2': 1600 - round_idx * 30,
            'model3': 1400 + round_idx * 40
        }
        result = detector.check_convergence(ratings, round_idx)
        
        # 不应该收敛
        assert result is False


def test_ranking_stability():
    """测试排名稳定性计算"""
    detector = ConvergenceDetector(
        min_stable_rounds=2,
        min_rounds=3
    )
    
    # 排名保持不变
    ratings1 = {'model1': 1600, 'model2': 1500, 'model3': 1400}
    detector.check_convergence(ratings1, 0)
    
    ratings2 = {'model1': 1605, 'model2': 1505, 'model3': 1405}
    detector.check_convergence(ratings2, 1)
    
    # 计算排名稳定性
    stability = detector._calculate_ranking_stability()
    
    # 排名完全相同，稳定性应该为1.0
    assert stability == 1.0


def test_rating_change_rate():
    """测试评分变化率计算"""
    detector = ConvergenceDetector()
    
    # 第一轮
    ratings1 = {'model1': 1500, 'model2': 1500}
    detector.check_convergence(ratings1, 0)
    
    # 第二轮，评分变化10%
    ratings2 = {'model1': 1650, 'model2': 1650}
    detector.check_convergence(ratings2, 1)
    
    change_rate = detector._calculate_rating_change_rate()
    
    # 变化率应该是0.1（10%）
    assert abs(change_rate - 0.1) < 0.01


def test_convergence_info():
    """测试获取收敛信息"""
    detector = ConvergenceDetector(
        threshold=0.01,
        min_stable_rounds=2,
        min_rounds=2
    )
    
    ratings = {'model1': 1500, 'model2': 1500}
    
    for round_idx in range(4):
        detector.check_convergence(ratings, round_idx)
    
    info = detector.get_convergence_info()
    
    assert 'total_rounds' in info
    assert 'stable_rounds' in info
    assert 'rating_change_rate' in info
    assert 'ranking_stability' in info
    assert 'is_converged' in info


def test_convergence_reset():
    """测试重置检测器"""
    detector = ConvergenceDetector()
    
    ratings = {'model1': 1500, 'model2': 1500}
    
    for round_idx in range(3):
        detector.check_convergence(ratings, round_idx)
    
    assert len(detector.rating_history) == 3
    
    detector.reset()
    
    assert len(detector.rating_history) == 0
    assert len(detector.ranking_history) == 0
    assert detector.stable_rounds_count == 0


def test_adaptive_convergence_detector():
    """测试自适应收敛检测器"""
    detector = AdaptiveConvergenceDetector(
        threshold=0.01,
        min_stable_rounds=3,
        min_rounds=3
    )
    
    # 少量模型（5个）
    ratings = {f'model{i}': 1500 for i in range(5)}
    
    for round_idx in range(5):
        detector.check_convergence(ratings, round_idx)
    
    # 自适应检测器应该根据模型数量调整参数
    info = detector.get_convergence_info()
    assert info is not None


def test_convergence_with_new_models():
    """测试新增模型时的收敛"""
    detector = ConvergenceDetector(
        min_stable_rounds=2,
        min_rounds=2
    )
    
    # 初始3个模型
    ratings1 = {'model1': 1500, 'model2': 1500, 'model3': 1500}
    detector.check_convergence(ratings1, 0)
    
    # 新增1个模型
    ratings2 = {'model1': 1500, 'model2': 1500, 'model3': 1500, 'model4': 1500}
    detector.check_convergence(ratings2, 1)
    
    # 排名应该仍然可以计算
    stability = detector._calculate_ranking_stability()
    assert 0 <= stability <= 1

