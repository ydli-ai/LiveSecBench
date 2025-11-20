"""
配对策略单元测试
"""

from collections import defaultdict
from livesecbench.infra.scoring.pairing_strategies import (
    SwissPairingStrategy,
    RoundRobinPairingStrategy,
    RandomPairingStrategy
)


def test_swiss_pairing_initial_round():
    """测试瑞士制首轮配对"""
    strategy = SwissPairingStrategy()
    models = ['model1', 'model2', 'model3', 'model4']
    ratings = {m: 1500 for m in models}
    history = defaultdict(set)
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 应该生成2对
    assert len(pairs) == 2
    
    # 所有模型都应该被配对
    paired_models = set()
    for m1, m2 in pairs:
        paired_models.add(m1)
        paired_models.add(m2)
    assert len(paired_models) == 4


def test_swiss_pairing_avoids_duplicates():
    """测试瑞士制避免重复配对"""
    strategy = SwissPairingStrategy()
    models = ['model1', 'model2', 'model3', 'model4']
    ratings = {m: 1500 for m in models}
    
    history = defaultdict(set)
    history['model1'].add('model2')
    history['model2'].add('model1')
    
    # 保存原始历史用于检查
    original_history = {m: history[m].copy() for m in history}
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 不应该包含原始历史中已配对的组合
    for m1, m2 in pairs:
        assert m2 not in original_history.get(m1, set())
        assert m1 not in original_history.get(m2, set())


def test_swiss_pairing_with_ratings():
    """测试瑞士制根据评分配对"""
    strategy = SwissPairingStrategy()
    models = ['model1', 'model2', 'model3', 'model4']
    
    # 设置不同评分
    ratings = {
        'model1': 1700,  # 高分
        'model2': 1600,
        'model3': 1400,
        'model4': 1300   # 低分
    }
    history = defaultdict(set)  # 修改
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 应该配对相近评分的模型
    # model1 vs model2, model3 vs model4
    assert len(pairs) == 2


def test_round_robin_pairing():
    """测试单循环配对"""
    strategy = RoundRobinPairingStrategy()
    models = ['model1', 'model2', 'model3', 'model4']
    ratings = {m: 1500 for m in models}
    history = defaultdict(set)  # 修改
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 单循环应该生成所有可能的配对
    # 4个模型 -> C(4,2) = 6对
    assert len(pairs) == 6


def test_round_robin_with_history():
    """测试单循环配对（即使有历史也生成所有配对）"""
    strategy = RoundRobinPairingStrategy()
    models = ['model1', 'model2', 'model3']
    ratings = {m: 1500 for m in models}
    
    # 已配对 - 单循环不会排除历史
    history = defaultdict(set)
    history['model1'].add('model2')
    history['model2'].add('model1')
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 单循环会生成所有可能的配对：C(3,2) = 3对
    # 包括历史中已有的配对
    assert len(pairs) == 3
    
    # 验证包含所有可能的配对
    pair_set = {tuple(sorted([m1, m2])) for m1, m2 in pairs}
    expected = {('model1', 'model2'), ('model1', 'model3'), ('model2', 'model3')}
    assert pair_set == expected


def test_random_pairing():
    """测试随机配对"""
    strategy = RandomPairingStrategy()
    models = ['model1', 'model2', 'model3', 'model4']
    ratings = {m: 1500 for m in models}
    history = defaultdict(set)  # 修改
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 应该生成2对（4个模型）
    assert len(pairs) == 2
    
    # 所有模型都应该被配对
    paired_models = set()
    for m1, m2 in pairs:
        paired_models.add(m1)
        paired_models.add(m2)
    assert len(paired_models) == 4


def test_random_pairing_with_history():
    """测试随机配对（不考虑历史，可能生成重复）"""
    strategy = RandomPairingStrategy()
    models = ['model1', 'model2', 'model3', 'model4']
    ratings = {m: 1500 for m in models}
    
    # 随机配对不会避免历史中的配对
    history = defaultdict(set)
    history['model1'].add('model2')
    history['model2'].add('model1')
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 随机配对应该生成2对（4个模型）
    assert len(pairs) == 2
    
    # 验证所有模型都被配对（可能包括历史中的）
    paired_models = set()
    for m1, m2 in pairs:
        paired_models.add(m1)
        paired_models.add(m2)
    assert len(paired_models) == 4


def test_pairing_with_odd_number_of_models():
    """测试奇数模型配对"""
    strategy = SwissPairingStrategy()
    models = ['model1', 'model2', 'model3']
    ratings = {m: 1500 for m in models}
    history = defaultdict(set)  # 修改
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 3个模型只能配对1对，有1个模型会轮空
    assert len(pairs) == 1


def test_empty_models_list():
    """测试空模型列表"""
    strategy = SwissPairingStrategy()
    models = []
    ratings = {}
    history = set()
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 应该返回空列表
    assert len(pairs) == 0


def test_single_model():
    """测试单个模型"""
    strategy = SwissPairingStrategy()
    models = ['model1']
    ratings = {'model1': 1500}
    history = set()
    
    pairs = strategy.generate_pairs(models, ratings, history)
    
    # 单个模型无法配对
    assert len(pairs) == 0

