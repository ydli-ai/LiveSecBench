"""
MySQL 存储单元测试

测试 MySQLStorage 的基本功能和并发性能

"""

import pytest
import os
import time
import asyncio
from typing import Dict, Any

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

from livesecbench.storage.mysql_storage import MySQLStorage


TEST_MYSQL_CONFIG = {
    'host': os.getenv('TEST_MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('TEST_MYSQL_PORT', '3306')),
    'user': os.getenv('TEST_MYSQL_USER', 'root'),
    'password': os.getenv('TEST_MYSQL_PASSWORD', ''),
    'database': os.getenv('TEST_MYSQL_DATABASE', 'livesecbench_test'),
}


@pytest.fixture
def mysql_storage():
    """创建测试用的 MySQL 存储实例"""
    if not MYSQL_AVAILABLE:
        pytest.skip("PyMySQL 未安装")
    
    try:
        storage = MySQLStorage(
            **TEST_MYSQL_CONFIG,
            model_outputs_table='test_model_outputs',
            pk_results_table='test_pk_results',
            tasks_table='test_evaluation_tasks',
        )
        
        # 清空测试表
        with storage._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"TRUNCATE TABLE `{storage.model_outputs_table}`")
            cursor.execute(f"TRUNCATE TABLE `{storage.pk_results_table}`")
            cursor.execute(f"TRUNCATE TABLE `{storage.tasks_table}`")
            conn.commit()
            cursor.close()
        
        yield storage
        
        # 清理
        with storage._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS `{storage.model_outputs_table}`")
            cursor.execute(f"DROP TABLE IF EXISTS `{storage.pk_results_table}`")
            cursor.execute(f"DROP TABLE IF EXISTS `{storage.tasks_table}`")
            conn.commit()
            cursor.close()
    
    except Exception as e:
        pytest.skip(f"无法连接到 MySQL 测试数据库: {e}")


class TestMySQLStorage:
    """MySQL 存储测试类"""
    
    def test_connection(self, mysql_storage):
        """测试数据库连接"""
        with mysql_storage._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            assert result is not None
    
    def test_save_and_get_model_output(self, mysql_storage):
        """测试保存和获取模型输出"""
        payload = {
            'model': 'test-model',
            'model_name': 'Test Model',
            'category': 'factuality',
            'prompt': 'Test prompt',
            'response': 'Test response',
            'status': 'success',
            'created_at': int(time.time()),
        }
        
        # 保存
        mysql_storage.save_model_output(payload)
        
        # 获取
        result = mysql_storage.get_model_output(
            model='test-model',
            category='factuality',
            prompt='Test prompt',
        )
        
        assert result is not None
        assert result['model'] == 'test-model'
        assert result['response'] == 'Test response'
    
    def test_upsert_model_output(self, mysql_storage):
        """测试模型输出的更新插入（upsert）"""
        payload = {
            'model': 'test-model',
            'category': 'factuality',
            'prompt': 'Test prompt',
            'response': 'Original response',
            'status': 'success',
        }
        
        # 第一次插入
        mysql_storage.save_model_output(payload)
        
        # 更新
        payload['response'] = 'Updated response'
        mysql_storage.save_model_output(payload)
        
        # 验证
        result = mysql_storage.get_model_output(
            model='test-model',
            category='factuality',
            prompt='Test prompt',
        )
        
        assert result['response'] == 'Updated response'
    
    @pytest.mark.asyncio
    async def test_async_save_and_get_model_output(self, mysql_storage):
        """测试异步保存和获取模型输出"""
        payload = {
            'model': 'async-test-model',
            'category': 'factuality',
            'prompt': 'Async test prompt',
            'response': 'Async test response',
            'status': 'success',
        }
        
        # 异步保存
        await mysql_storage.asave_model_output(payload)
        
        # 异步获取
        result = await mysql_storage.aget_model_output(
            model='async-test-model',
            category='factuality',
            prompt='Async test prompt',
        )
        
        assert result is not None
        assert result['response'] == 'Async test response'
    
    def test_save_and_get_pk_result(self, mysql_storage):
        """测试保存和获取 PK 结果"""
        payload = {
            'model_a': 'model-a',
            'model_b': 'model-b',
            'winner': 'model-a',
            'judge_reasoning': 'Model A is better',
        }
        
        # 保存
        mysql_storage.save_pk_result(
            evaluation_dimension='factuality',
            category='test',
            question='Test question',
            model_a='model-a',
            model_b='model-b',
            payload=payload,
        )
        
        # 获取
        result = mysql_storage.get_pk_result(
            category='test',
            question='Test question',
            model_a='model-a',
            model_b='model-b',
        )
        
        assert result is not None
        assert result['winner'] == 'model-a'
    
    def test_save_and_get_task_info(self, mysql_storage):
        """测试保存和获取任务信息"""
        task_id = f"test_task_{int(time.time())}"
        task_info = {
            'eval_run_name': 'Test Run',
            'config_path': 'test/config.yaml',
            'dimensions': ['factuality', 'ethics'],
            'models': ['model-a', 'model-b'],
        }
        
        # 保存
        mysql_storage.save_task_info(task_id, task_info)
        
        # 获取
        result = mysql_storage.get_task_info(task_id)
        
        assert result is not None
        assert result['eval_run_name'] == 'Test Run'
        assert 'factuality' in result['dimensions']
    
    def test_list_tasks(self, mysql_storage):
        """测试列出任务"""
        # 创建多个任务
        for i in range(5):
            task_id = f"test_task_{i}_{int(time.time())}"
            task_info = {
                'eval_run_name': f'Test Run {i}',
                'config_path': f'test/config_{i}.yaml',
            }
            mysql_storage.save_task_info(task_id, task_info)
            time.sleep(0.01)  # 确保时间戳不同
        
        # 列出任务
        tasks = mysql_storage.list_tasks(limit=3)
        
        assert len(tasks) <= 3
        assert all('task_id' in task for task in tasks)
    
    def test_concurrent_writes(self, mysql_storage):
        """测试并发写入"""
        import threading
        
        def write_output(model_id: int):
            for i in range(10):
                payload = {
                    'model': f'model-{model_id}',
                    'category': 'factuality',
                    'prompt': f'Prompt {i}',
                    'response': f'Response {i}',
                    'status': 'success',
                }
                mysql_storage.save_model_output(payload)
        
        # 启动多个线程并发写入
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_output, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证数据完整性
        with mysql_storage._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) as count FROM `{mysql_storage.model_outputs_table}`"
            )
            count = cursor.fetchone()['count']
            cursor.close()
        
        # 应该有 5 个模型 × 10 个提示 = 50 条记录
        assert count == 50
    
    def test_normalize_value(self, mysql_storage):
        """测试值标准化"""
        # None 值
        assert mysql_storage._normalize_value(None) is None
        
        # 单元素列表
        assert mysql_storage._normalize_value(['value']) == 'value'
        
        # 多元素列表
        result = mysql_storage._normalize_value(['a', 'b', 'c'])
        assert 'a' in result and 'b' in result and 'c' in result
        
        # 字符串
        assert mysql_storage._normalize_value('test') == 'test'
    
    def test_large_text_storage(self, mysql_storage):
        """测试大文本存储"""
        large_text = 'A' * 10000  # 10KB 文本
        
        payload = {
            'model': 'test-model',
            'category': 'factuality',
            'prompt': large_text,
            'response': large_text,
            'status': 'success',
        }
        
        # 保存
        mysql_storage.save_model_output(payload)
        
        # 获取
        result = mysql_storage.get_model_output(
            model='test-model',
            category='factuality',
            prompt=large_text,
        )
        
        assert result is not None
        assert len(result['response']) == 10000


@pytest.mark.skipif(not MYSQL_AVAILABLE, reason="PyMySQL 未安装")
def test_storage_factory_mysql():
    """测试存储工厂创建 MySQL 存储"""
    from livesecbench.storage import create_storage_simple
    
    try:
        storage = create_storage_simple(
            storage_type='mysql',
            mysql_config=TEST_MYSQL_CONFIG,
            model_outputs_table='test_model_outputs',
            pk_results_table='test_pk_results',
            tasks_table='test_evaluation_tasks',
        )
        
        assert storage is not None
        assert isinstance(storage, MySQLStorage)
    except Exception as e:
        pytest.skip(f"无法创建 MySQL 存储: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

