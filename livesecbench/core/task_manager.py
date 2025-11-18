"""评测任务管理模块"""

from datetime import datetime
from typing import Dict, Any, Optional, List


class TaskManager:
    """评测任务管理器: 生成任务ID，记录任务信息和结果文件"""
    
    def __init__(self, task_id: Optional[str] = None):
        if task_id:
            self.task_id = task_id
        else:
            self.task_id = self.generate_task_id()
        
        self.created_at = datetime.now()
        self.task_info: Dict[str, Any] = {
            'task_id': self.task_id,
            'created_at': self.created_at.isoformat(),
            'config_path': None,
            'eval_run_name': None,
            'models': [],
            'dimensions': [],
            'question_counts': {},
            'result_files': [],
        }
    
    @staticmethod
    def generate_task_id() -> str:
        """生成唯一的任务ID，格式: YYYYMMDD_HHMMSS"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def set_config_info(self, config_path: str, eval_run_name: str):
        """设置配置信息"""
        self.task_info['config_path'] = config_path
        self.task_info['eval_run_name'] = eval_run_name
    
    def set_models(self, models: List[str]):
        """设置参与评测的模型列表"""
        self.task_info['models'] = models
    
    def set_dimensions(self, dimensions: List[str]):
        """设置评测维度"""
        self.task_info['dimensions'] = dimensions
    
    def set_question_counts(self, question_counts: Dict[str, int]):
        """设置各维度的题目数量"""
        self.task_info['question_counts'] = question_counts
    
    def add_result_file(self, file_path: str, file_type: str = 'unknown'):
        """添加结果文件路径"""
        self.task_info['result_files'].append({
            'path': file_path,
            'type': file_type,
            'added_at': datetime.now().isoformat()
        })
    
    def get_task_info(self) -> Dict[str, Any]:
        """获取完整的任务信息"""
        return self.task_info.copy()
    
    def format_result_filename(self, base_name: str, extension: str = '') -> str:
        """格式化结果文件名，包含任务ID"""
        if extension and not extension.startswith('.'):
            extension = '.' + extension
        return f"{base_name}_{self.task_id}{extension}"
    
    def __str__(self) -> str:
        return f"TaskManager(task_id={self.task_id})"
    
    def __repr__(self) -> str:
        return self.__str__()
