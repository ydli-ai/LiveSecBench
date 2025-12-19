"""
存储层抽象基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import json


class BaseStorage(ABC):
    
    def __init__(
        self,
        model_outputs_table: str = "model_outputs",
        pk_results_table: str = "pk_results",
        tasks_table: str = "evaluation_tasks",
        task_id: Optional[str] = None,
    ) -> None:
        self.model_outputs_table = self._sanitize_identifier(model_outputs_table)
        self.pk_results_table = self._sanitize_identifier(pk_results_table)
        self.tasks_table = self._sanitize_identifier(tasks_table)
        self.task_id = task_id
    
    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        """
        清理并验证数据库标识符（表名、列名等）
        """
        if not value or not value.replace("_", "").isalnum():
            raise ValueError(f"非法的数据库标识符: {value}")
        return value
    
    @staticmethod
    def _normalize_value(value: Any) -> Optional[str]:
        """
        标准化存储值（将复杂类型转为字符串）
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return str(value[0])
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    
    @abstractmethod
    def _connect(self):
        """
        创建数据库连接
        """
        pass
    
    @abstractmethod
    def _ensure_tables(self) -> None:
        """
        确保所有必需的表存在，如不存在则创建
        """
        pass
    
    # ==================== 模型输出相关接口 ====================
    
    @abstractmethod
    def get_model_output(
        self, 
        model: str, 
        category: Optional[str], 
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取模型输出（同步）
        """
        pass
    
    @abstractmethod
    async def aget_model_output(
        self, 
        model: str, 
        category: Optional[str], 
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取模型输出（异步）
        """
        pass
    
    @abstractmethod
    def save_model_output(self, payload: Dict[str, Any]) -> None:
        """
        保存模型输出（同步）
        """
        pass
    
    @abstractmethod
    async def asave_model_output(self, payload: Dict[str, Any]) -> None:
        """
        保存模型输出（异步）
        """
        pass
    
    # ==================== PK结果相关接口 ====================
    
    @abstractmethod
    def get_pk_result(
        self,
        category: Optional[str],
        question: str,
        model_a: str,
        model_b: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取PK结果
        """
        pass
    
    @abstractmethod
    def save_pk_result(
        self,
        evaluation_dimension: str,
        category: Optional[str],
        question: str,
        model_a: str,
        model_b: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        保存PK结果
        """
        pass
    
    def _save_pk_result_from_queue(self, data: Dict[str, Any]) -> None:
        """
        从队列数据恢复并保存 PK 结果
        """
        self.save_pk_result(
            evaluation_dimension=data['evaluation_dimension'],
            category=data['category'],
            question=data['question'],
            model_a=data['model_a'],
            model_b=data['model_b'],
            payload=data['payload'],
        )
    
    # ==================== 任务信息相关接口 ====================
    
    @abstractmethod
    def save_task_info(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """
        保存任务信息
        """
        pass
    
    @abstractmethod
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息
        """
        pass
    
    @abstractmethod
    def list_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出最近的任务
        """
        pass

