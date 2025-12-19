"""
ConfigManager单元测试
"""

import os
import tempfile
import pytest
from pathlib import Path
import yaml

from livesecbench.infra.config.config_manager import ConfigManager


@pytest.fixture
def sample_config():
    """创建示例配置"""
    return {
        'eval_run_name': 'test_eval',
        'models_to_test': [
            {
                'model_name': 'Test Model 1',
                'organization': 'TestOrg',
                'is_reasoning': False,
                'image_text_input': False,
                'api_config': {
                    'base_url': 'https://api.test.com',
                    'end_point': '/chat/completions',
                    'api_key': 'env_var:TEST_API_KEY',
                    'model_id': 'provider/model1'
                }
            },
            {
                'model_name': 'Test Model 2',
                'organization': 'TestOrg',
                'is_reasoning': True,
                'image_text_input': True,
                'api_config': {
                    'base_url': 'https://api.test.com',
                    'end_point': '/chat/completions',
                    'api_key': 'env_var:TEST_API_KEY',
                    'model_id': 'provider/model2'
                }
            }
        ],
        'question_selection': [
            {'dimension': 'ethics', 'version': 'v20251030', 'sample_size': 10},
            {'dimension': 'legality', 'version': 'v20251030', 'sample_size': 10}
        ],
        'scoring_config': [
            {'dimension': 'ethics', 'scorer': 'model_based', 'params': {}},
            {'dimension': 'legality', 'scorer': 'model_based', 'params': {}}
        ],
        'scoring_settings': {
            'model_based': {
                'elo': {
                    'init_rating': 1500,
                    'init_k': 32,
                    'result_output_dir': 'elo_results'
                }
            }
        },
        'dimension_name_map': {
            '合法性': 'lawfulness',
            '伦理性': 'ethics'
        },
        'storage': {
            'sqlite': {
                'db_path': 'test.db',
                'model_outputs_table': 'model_outputs',
                'pk_results_table': 'pk_results'
            }
        },
        'judge_model_api': {
            'base_url': 'https://api.judge.com',
            'api_key': 'env_var:JUDGE_API_KEY',
            'model': 'judge-model'
        }
    }


@pytest.fixture
def config_file(sample_config):
    """创建临时配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name
    
    yield config_path
    
    # 清理
    os.unlink(config_path)


@pytest.fixture
def env_keys(monkeypatch):
    """为依赖环境变量的测试提供默认值"""
    monkeypatch.setenv('TEST_API_KEY', 'test_key_123')
    monkeypatch.setenv('JUDGE_API_KEY', 'judge_key_456')
    yield
    monkeypatch.delenv('TEST_API_KEY', raising=False)
    monkeypatch.delenv('JUDGE_API_KEY', raising=False)


def test_config_manager_initialization(config_file):
    """测试ConfigManager初始化"""
    manager = ConfigManager(config_file)
    assert manager.config_path.exists()
    assert manager._config is not None


def test_get_eval_run_name(config_file):
    """测试获取评测运行名称"""
    manager = ConfigManager(config_file)
    assert manager.get_eval_run_name() == 'test_eval'


def test_get_models_to_test(config_file, env_keys):
    """测试获取待测模型列表"""
    manager = ConfigManager(config_file)
    models = manager.get_models_to_test()
    
    assert len(models) == 2
    assert 'model_id' not in models[0]
    assert models[0]['api_config']['model_id'] == 'provider/model1'
    assert models[1]['api_config']['model_id'] == 'provider/model2'
    assert models[1]['image_text_input'] is True


def test_get_reasoning_model_ids(config_file, env_keys):
    """测试获取推理模型ID列表"""
    manager = ConfigManager(config_file)
    reasoning_models = manager.get_reasoning_model_ids()
    
    assert len(reasoning_models) == 1
    assert 'provider/model2' in reasoning_models


def test_get_all_model_ids(config_file, env_keys):
    """测试获取所有模型ID"""
    manager = ConfigManager(config_file)
    all_models = manager.get_all_model_ids()
    
    assert len(all_models) == 2
    assert 'provider/model1' in all_models
    assert 'provider/model2' in all_models


def test_get_dimensions(config_file):
    """测试获取评测维度"""
    manager = ConfigManager(config_file)
    dimensions = manager.get_dimensions()
    
    assert len(dimensions) == 2
    assert 'ethics' in dimensions
    assert 'legality' in dimensions


def test_get_dimension_name_map(config_file):
    """测试获取维度名称映射"""
    manager = ConfigManager(config_file)
    mapping = manager.get_dimension_name_map()
    
    assert mapping['合法性'] == 'lawfulness'
    assert mapping['伦理性'] == 'ethics'


def test_get_model_name_map(config_file):
    """测试获取模型名称映射"""
    manager = ConfigManager(config_file)
    name_map = manager.get_model_name_map()
    
    assert name_map['provider/model1'] == 'Test Model 1'
    assert name_map['provider/model2'] == 'Test Model 2'


def test_get_question_selection(config_file):
    """测试获取问题选择配置"""
    manager = ConfigManager(config_file)
    selection = manager.get_question_selection()
    
    assert len(selection) == 2
    assert selection[0]['dimension'] == 'ethics'
    assert selection[0]['sample_size'] == 10


def test_get_scoring_config(config_file):
    """测试获取评分配置"""
    manager = ConfigManager(config_file)
    scoring_config = manager.get_scoring_config()
    
    assert len(scoring_config) == 2
    assert scoring_config[0]['scorer'] == 'model_based'


def test_get_dimension_scorer_config(config_file):
    """测试获取特定维度的评分配置"""
    manager = ConfigManager(config_file)
    ethics_config = manager.get_dimension_scorer_config('ethics')
    
    assert ethics_config['scorer'] == 'model_based'
    assert ethics_config['dimension'] == 'ethics'


def test_get_elo_settings(config_file):
    """测试获取ELO设置"""
    manager = ConfigManager(config_file)
    elo_settings = manager.get_elo_settings()
    
    assert elo_settings['init_rating'] == 1500
    assert elo_settings['init_k'] == 32


def test_get_elo_results_dir(config_file, env_keys, monkeypatch):
    """测试获取ELO结果目录"""
    fixed_day = "2025_11_18"

    def fake_strftime(_fmt, _ts):
        return fixed_day

    monkeypatch.setattr(
        "livesecbench.infra.config.config_manager.time.strftime",
        fake_strftime,
        raising=False,
    )

    manager = ConfigManager(config_file)
    results_dir = manager.get_elo_results_dir()
    
    assert results_dir == Path('results') / fixed_day / 'elo_results'


def test_get_storage_db_path(config_file):
    """测试获取存储数据库路径"""
    manager = ConfigManager(config_file)
    db_path = manager.get_storage_db_path()
    
    assert db_path == 'test.db'


def test_get_storage_tables(config_file):
    """测试获取存储表名"""
    manager = ConfigManager(config_file)
    tables = manager.get_storage_tables()
    
    assert tables['model_outputs_table'] == 'model_outputs'
    assert tables['pk_results_table'] == 'pk_results'


def test_resolve_env_var(config_file, env_keys):
    """测试环境变量解析（从 api_config 中解析）"""
    manager = ConfigManager(config_file)
    models = manager.get_models_to_test()
    
    # 验证环境变量已正确解析
    assert models[0]['api_config']['api_key'] == 'test_key_123'
    assert models[1]['api_config']['api_key'] == 'test_key_123'


def test_resolve_env_var_missing(config_file, monkeypatch):
    """测试缺失的环境变量"""
    monkeypatch.delenv('TEST_API_KEY', raising=False)
    monkeypatch.delenv('JUDGE_API_KEY', raising=False)
    manager = ConfigManager(config_file)
    
    # 测试从 api_config 解析环境变量时，如果环境变量缺失会抛出异常
    with pytest.raises(ValueError, match="环境变量.*未设置"):
        manager.get_models_to_test()


def test_get_model_error_handlers(config_file):
    """测试获取模型错误处理配置"""
    manager = ConfigManager(config_file)
    handlers = manager.get_model_error_handlers()
    
    # 默认应该返回空字典
    assert isinstance(handlers, dict)


def test_validate_config_valid(config_file, env_keys):
    """测试配置验证 - 有效配置"""
    manager = ConfigManager(config_file)
    errors = manager.validate_config()
    
    assert len(errors) == 0


def test_validate_config_missing_models():
    """测试配置验证 - 缺少模型配置"""
    config = {
        'eval_run_name': 'test',
        'question_selection': [{'dimension': 'ethics'}],
        'scoring_config': [{'dimension': 'ethics', 'scorer': 'model_based'}],
        'inference_platforms': {'test': {'base_url': 'https://test.com', 'api_key': 'key'}},
        'judge_model_api': {'base_url': 'https://judge.com', 'api_key': 'key', 'model': 'judge'}
    }
    
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        manager = ConfigManager(config_path)
        errors = manager.validate_config()
        
        assert len(errors) > 0
        assert any('未配置任何待测模型' in error for error in errors)
    finally:
        import os
        os.unlink(config_path)

