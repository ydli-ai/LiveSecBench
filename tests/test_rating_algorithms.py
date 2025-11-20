"""
评分算法单元测试
"""

from livesecbench.infra.scoring.rating_algorithms import ELORatingAlgorithm


def test_elo_algorithm_initialization():
    """测试ELO算法初始化"""
    algo = ELORatingAlgorithm(
        init_rating=1500,
        k_factor=32,
        logistic_constant=400
    )
    
    assert algo.init_rating == 1500
    assert algo.k_factor == 32
    assert algo.logistic_constant == 400


def test_elo_initialize_ratings():
    """测试初始化评分"""
    algo = ELORatingAlgorithm(init_rating=1500)
    
    # 测试获取初始评分
    init_rating = algo.get_initial_rating()
    
    assert init_rating == 1500


def test_elo_update_after_win():
    """测试胜利后评分更新"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32)
    ratings = {'winner': 1500, 'loser': 1500}
    
    # 正确的调用方式：(model_a, model_b, winner, current_ratings)
    new_winner, new_loser = algo.update_ratings('winner', 'loser', 'winner', ratings)
    
    # 胜者评分应该上升
    assert new_winner > 1500
    # 败者评分应该下降
    assert new_loser < 1500
    
    # 总评分应该保持不变（零和游戏）
    assert abs((new_winner + new_loser) - 3000) < 0.01


def test_elo_update_upset():
    """测试冷门（低分者战胜高分者）"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32)
    
    # 低分者（underdog）战胜高分者（favorite）
    ratings = {'underdog': 1300, 'favorite': 1700}
    
    initial_underdog = ratings['underdog']
    initial_favorite = ratings['favorite']
    
    new_underdog, new_favorite = algo.update_ratings('underdog', 'favorite', 'underdog', ratings)
    
    # 冷门胜利，低分者应该获得更多分数
    underdog_gain = new_underdog - initial_underdog
    favorite_loss = initial_favorite - new_favorite
    
    # 获得的分数应该相等（零和）
    assert abs(underdog_gain - favorite_loss) < 0.01
    
    # 冷门胜利的分数变化应该更大
    assert underdog_gain > 16  # 大于平均的K/2


def test_elo_update_expected_result():
    """测试预期结果（高分者战胜低分者）"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32)
    
    ratings = {'favorite': 1700, 'underdog': 1300}
    
    initial_favorite = ratings['favorite']
    initial_underdog = ratings['underdog']
    
    new_favorite, new_underdog = algo.update_ratings('favorite', 'underdog', 'favorite', ratings)
    
    # 预期结果，分数变化应该较小
    favorite_gain = new_favorite - initial_favorite
    underdog_loss = initial_underdog - new_underdog
    
    # 分数变化应该小于冷门情况
    assert favorite_gain < 16  # 小于平均的K/2
    assert underdog_loss < 16


def test_elo_expected_score():
    """测试期望得分计算"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32, logistic_constant=400)
    
    # 评分相同时，期望得分应该是0.5
    expected = algo.get_expected_score(1500, 1500)
    assert abs(expected - 0.5) < 0.01
    
    # 高分者对低分者，期望得分应该高于0.5
    expected_high = algo.get_expected_score(1700, 1300)
    assert expected_high > 0.5
    
    # 低分者对高分者，期望得分应该低于0.5
    expected_low = algo.get_expected_score(1300, 1700)
    assert expected_low < 0.5
    
    # 两个期望得分应该相加为1
    assert abs(expected_high + expected_low - 1.0) < 0.01


def test_elo_rating_difference_impact():
    """测试评分差距对期望得分的影响"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32, logistic_constant=400)
    
    # 评分差距越大，期望得分差异越大
    expected_100 = algo.get_expected_score(1500, 1400)  # 差距100
    expected_200 = algo.get_expected_score(1500, 1300)  # 差距200
    expected_400 = algo.get_expected_score(1500, 1100)  # 差距400
    
    # 差距增大，期望得分应该增加
    assert expected_100 < expected_200 < expected_400
    
    # 差距400时，期望得分应该约为0.9
    assert abs(expected_400 - 0.9) < 0.1


def test_elo_symmetry():
    """测试ELO算法的对称性"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32)
    
    # 场景1: A战胜B
    ratings1 = {'A': 1500, 'B': 1500}
    new_A1, new_B1 = algo.update_ratings('A', 'B', 'A', ratings1)
    
    # 场景2: B战胜A
    ratings2 = {'A': 1500, 'B': 1500}
    new_B2, new_A2 = algo.update_ratings('B', 'A', 'B', ratings2)
    
    # 变化应该对称
    assert abs(new_A1 - new_B2) < 0.01
    assert abs(new_B1 - new_A2) < 0.01


def test_elo_multiple_updates():
    """测试多次更新"""
    algo = ELORatingAlgorithm(init_rating=1500, k_factor=32)
    ratings = {'model1': 1500, 'model2': 1500, 'model3': 1500}
    
    # model1连胜
    new_m1_1, new_m2 = algo.update_ratings('model1', 'model2', 'model1', ratings)
    ratings['model1'] = new_m1_1
    ratings['model2'] = new_m2
    
    new_m1_2, new_m3 = algo.update_ratings('model1', 'model3', 'model1', ratings)
    ratings['model1'] = new_m1_2
    ratings['model3'] = new_m3
    
    # model1应该是最高分
    assert ratings['model1'] > ratings['model2']
    assert ratings['model1'] > ratings['model3']
    
    # model2和model3应该下降
    assert ratings['model2'] < 1500
    assert ratings['model3'] < 1500

