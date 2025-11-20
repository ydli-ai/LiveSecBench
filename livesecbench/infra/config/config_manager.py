"""
统一配置管理器
加载和解析YAML配置文件，支持环境变量解析、配置验证
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
import os
import time

DEFAULT_DIMENSION_NAME_MAP: Dict[str, str] = {
    '合法性': 'legality',
    '伦理性': 'ethics',
    '事实性': 'factuality',
    '隐私性': 'privacy',
    '对抗鲁棒性': 'robustness',
    '推理安全': 'reasoning',
}

class ConfigManager:
    """统一配置管理器: 加载YAML配置、解析环境变量、提供配置访问接口"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        self._config = self._load_config()
        self._model_name_map = self._build_model_name_map()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config:
                    raise ValueError("配置文件为空")
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def _resolve_env_var(self, value: str) -> str:
        """解析环境变量格式的配置值，支持格式: env_var:VARIABLE_NAME"""
        if isinstance(value, str) and value.startswith("env_var:"):
            env_key = value[8:]  # 移除 "env_var:" 前缀
            env_value = os.getenv(env_key)
            if env_value is None:
                raise ValueError(f"环境变量 {env_key} 未设置")
            return env_value
        return value
    
    def _build_model_name_map(self) -> Dict[str, str]:
        """遍历模型配置，生成 {api_model_id -> model_name} 映射"""
        name_map: Dict[str, str] = {}
        for model in self._config.get('models_to_test', []) or []:
            api_config = model.get('api_config', {}) or {}
            model_id = api_config.get('model_id')
            if not model_id:
                continue
            model_name = model.get('model_name', model_id)
            name_map[model_id] = model_name
            normalized_id = model.get('model_id')
            if normalized_id:
                name_map.setdefault(normalized_id, model_name)
        return name_map
    
    def get_raw_config(self) -> dict:
        """获取原始配置字典"""
        return self._config
    
    def get_models_to_test(self) -> List[Dict]:
        """获取待测模型列表"""
        models = self._config.get('models_to_test', [])
        resolved_models = []
        for model in models:
            resolved_model = dict(model)
            api_config = resolved_model.get('api_config', {})
            if api_config:
                resolved_api_config = dict(api_config)
                if 'api_key' in resolved_api_config:
                    resolved_api_config['api_key'] = self._resolve_env_var(resolved_api_config['api_key'])
                if 'base_url' in resolved_api_config:
                    resolved_api_config['base_url'] = self._resolve_env_var(resolved_api_config['base_url'])
                resolved_model['api_config'] = resolved_api_config
            resolved_models.append(resolved_model)
        return resolved_models
    
    def get_all_model_ids(self) -> List[str]:
        """获取所有模型ID列表"""
        model_ids = []
        for model_config in self.get_models_to_test():
            api_config = model_config.get('api_config', {})
            model_id = api_config.get('model_id')
            if model_id:
                model_ids.append(model_id)
        return model_ids
    
    def get_reasoning_model_ids(self) -> List[str]:
        """获取推理模型ID列表"""
        reasoning_models = []
        for model_config in self.get_models_to_test():
            if model_config.get('is_reasoning', False):
                api_config = model_config.get('api_config', {})
                model_id = api_config.get('model_id')
                if model_id:
                    reasoning_models.append(model_id)
        return reasoning_models
    
    def get_dimensions(self) -> List[str]:
        """获取评测维度列表"""
        return [
            item['dimension'] 
            for item in self._config.get('question_selection', [])
            if 'dimension' in item
        ]

    def get_dimension_name_map(self) -> Dict[str, str]:
        """获取维度名称映射（中文 -> 英文）"""
        configured_map = self._config.get('dimension_name_map', {}) or {}
        # 配置文件中的映射优先生效，未覆盖部分使用默认值
        return {**DEFAULT_DIMENSION_NAME_MAP, **configured_map}

    def get_model_name_map(self) -> Dict[str, str]:
        """获取模型ID到模型名称的映射"""
        return dict(self._model_name_map)
    
    def get_question_selection(self) -> List[Dict]:
        """获取题库选择配置"""
        return self._config.get('question_selection', [])
    
    def get_scoring_config(self) -> List[Dict]:
        """获取评分配置"""
        return self._config.get('scoring_config', [])
    
    def get_dimension_scorer_config(self, dimension: str) -> Optional[Dict]:
        """获取指定维度的评分器配置"""
        for item in self.get_scoring_config():
            if item.get('dimension') == dimension:
                return item
        return None
    
    def get_scoring_settings(self) -> Dict:
        """获取评分全局设置"""
        return self._config.get('scoring_settings', {})
    
    def get_elo_settings(self) -> Dict:
        """获取ELO评分设置"""
        return self.get_scoring_settings().get('model_based', {}).get('elo', {})
    
    # ==================== 路径相关配置 ====================
    
    def get_elo_results_dir(self) -> Path:
        """获取ELO结果目录"""
        elo_settings = self.get_elo_settings()
        result_dir = elo_settings.get('result_output_dir', None)
        
        if result_dir is None or result_dir == 'elo_results':
            day_tag = time.strftime('%Y_%m_%d', time.localtime())
            return Path("results") / day_tag / "elo_results"
        
        result_path = Path(result_dir)
        if not result_path.is_absolute() and not str(result_path).startswith('results/'):
            day_tag = time.strftime('%Y_%m_%d', time.localtime())
            return Path("results") / day_tag / result_dir
        
        return result_path
    
    def get_elo_history_dir(self) -> Path:
        """获取ELO历史目录"""
        elo_settings = self.get_elo_settings()
        history_dir = elo_settings.get('history_output_dir', 'elo_histories')
        return Path(history_dir)
    
    def get_elo_record_dir(self) -> Path:
        """获取ELO记录目录"""
        elo_settings = self.get_elo_settings()
        record_dir = elo_settings.get('record_output_dir', '测评记录')
        return Path(record_dir)
    
    def get_storage_config(self) -> Dict:
        """获取存储配置"""
        return self._config.get('storage', {})
    
    def get_storage_db_path(self) -> str:
        """获取SQLite数据库路径"""
        sqlite_config = self.get_storage_config().get('sqlite', {})
        return sqlite_config.get('db_path', 'data/livesecbench.db')
    
    def get_storage_tables(self) -> Dict[str, str]:
        """获取存储表名配置"""
        sqlite_config = self.get_storage_config().get('sqlite', {})
        return {
            'model_outputs_table': sqlite_config.get('model_outputs_table', 'model_outputs'),
            'pk_results_table': sqlite_config.get('pk_results_table', 'pk_results'),
        }
    
    def get_api_call_settings(self) -> Dict:
        """获取API调用配置"""
        return self._config.get('api_call_settings', {})
    
    def get_judge_model_api(self) -> Dict:
        """获取裁判模型API配置"""
        judge_api = dict(self._config.get('judge_model_api', {}))
        if 'api_key' in judge_api:
            judge_api['api_key'] = self._resolve_env_var(judge_api['api_key'])
        return judge_api
    
    def get_eval_run_name(self) -> str:
        """获取评测任务名称"""
        return self._config.get('eval_run_name', 'LiveSecBench_Eval')
    
    def get_description(self) -> str:
        """获取评测任务描述"""
        return self._config.get('description', '')
    
    def get_model_error_handlers(self) -> Dict:
        """获取模型错误处理配置"""
        return self._config.get('model_error_handlers', {})
    
    def get_report_settings(self) -> Dict:
        """获取报告生成配置"""
        return self._config.get('report_settings', {})
    
    def get_report_prompt_template(self) -> str:
        """获取报告提示词模板"""
        report_settings = self.get_report_settings()
        return report_settings.get('prompt_template', '')
    
    def validate_config(self) -> List[str]:
        """验证配置文件的完整性和有效性"""
        errors = []
        
        if not self._config.get('eval_run_name'):
            errors.append("缺少必要配置: eval_run_name")
        
        models = self.get_models_to_test()
        if not models:
            errors.append("未配置任何待测模型")
        else:
            for idx, model in enumerate(models):
                if not model.get('model_name'):
                    errors.append(f"模型配置 #{idx+1} 缺少 model_name 字段")
                
                api_config = model.get('api_config', {})
                if not api_config:
                    errors.append(f"模型配置 #{idx+1} 缺少 api_config 字段")
                    continue
                
                if not api_config.get('base_url'):
                    errors.append(f"模型配置 #{idx+1} 的 api_config 缺少 base_url")
                if not api_config.get('api_key'):
                    errors.append(f"模型配置 #{idx+1} 的 api_config 缺少 api_key")
                if not api_config.get('model_id'):
                    errors.append(f"模型配置 #{idx+1} 的 api_config 缺少 model_id")
        
        dimensions = self.get_dimensions()
        if not dimensions:
            errors.append("未配置任何评测维度")
        
        scoring_config = self.get_scoring_config()
        if not scoring_config:
            errors.append("未配置评分器")
        else:
            for item in scoring_config:
                if not item.get('dimension'):
                    errors.append("评分配置缺少 dimension 字段")
                if not item.get('scorer'):
                    errors.append(f"维度 {item.get('dimension', 'unknown')} 缺少 scorer 字段")
        
        
        judge_api = self.get_judge_model_api()
        if not judge_api.get('base_url'):
            errors.append("judge_model_api 缺少 base_url")
        if not judge_api.get('api_key'):
            errors.append("judge_model_api 缺少 api_key")
        if not judge_api.get('model'):
            errors.append("judge_model_api 缺少 model")
        
        return errors
