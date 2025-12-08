"""
MySQL 存储实现

提供基于 MySQL 的数据存储，支持高并发写入
"""
import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional, List
from contextlib import contextmanager

try:
    import pymysql
    from pymysql.cursors import DictCursor
    from dbutils.pooled_db import PooledDB
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False

try:
    import aiomysql
    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False

from livesecbench.utils.logger import get_logger
from livesecbench.storage.base_storage import BaseStorage

logger = get_logger(__name__)


class MySQLStorage(BaseStorage):
    """MySQL 存储实现"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "livesecbench",
        charset: str = "utf8mb4",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        model_outputs_table: str = "model_outputs",
        pk_results_table: str = "pk_results",
        tasks_table: str = "evaluation_tasks",
        task_id: Optional[str] = None,
    ) -> None:
        """
        初始化 MySQL 存储
        """
        if not PYMYSQL_AVAILABLE:
            raise ImportError(
                "PyMySQL 未安装。请运行: pip install pymysql dbutils"
            )
        
        super().__init__(model_outputs_table, pk_results_table, tasks_table, task_id)
        
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=pool_size + max_overflow,
            mincached=pool_size,
            maxcached=pool_size,
            blocking=True,
            maxusage=None,
            setsession=[],
            ping=1,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            cursorclass=DictCursor,
        )
        
        logger.info(
            f"MySQL 连接池已创建: {host}:{port}/{database}, "
            f"pool_size={pool_size}, max_overflow={max_overflow}"
        )
        
        self._ensure_database()
        self._ensure_tables()
    
    def _ensure_database(self) -> None:
        """确保数据库存在"""
        temp_pool = PooledDB(
            creator=pymysql,
            maxconnections=1,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            charset=self.charset,
            cursorclass=DictCursor,
        )
        
        try:
            conn = temp_pool.connection()
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                f"CHARACTER SET {self.charset} COLLATE {self.charset}_unicode_ci"
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"数据库 {self.database} 已就绪")
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            raise
    
    @contextmanager
    def _connect(self):
        """
        获取数据库连接（使用连接池）
        """
        conn = self.pool.connection()
        try:
            yield conn
        finally:
            conn.close()
    
    def _ensure_tables(self) -> None:
        """创建所有必需的表"""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{self.tasks_table}` (
                    task_id VARCHAR(255) PRIMARY KEY,
                    created_at VARCHAR(50) NOT NULL,
                    config_path TEXT,
                    eval_run_name VARCHAR(255),
                    task_info_json LONGTEXT,
                    updated_at VARCHAR(50)
                ) ENGINE=InnoDB DEFAULT CHARSET={self.charset}
            """)
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{self.model_outputs_table}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    task_id VARCHAR(255),
                    model_name VARCHAR(255),
                    model VARCHAR(255) NOT NULL,
                    category TEXT,
                    prompt LONGTEXT,
                    prompt_hash VARCHAR(32) NOT NULL,
                    status VARCHAR(50),
                    payload_json LONGTEXT NOT NULL,
                    created_at BIGINT,
                    updated_at BIGINT,
                    UNIQUE KEY unique_output (model, category(191), prompt_hash)
                ) ENGINE=InnoDB DEFAULT CHARSET={self.charset}
            """)
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{self.pk_results_table}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    task_id VARCHAR(255),
                    evaluation_dimension VARCHAR(255),
                    category TEXT,
                    question LONGTEXT,
                    question_hash VARCHAR(32) NOT NULL,
                    model_a VARCHAR(255),
                    model_b VARCHAR(255),
                    winner VARCHAR(255),
                    result_json LONGTEXT NOT NULL,
                    created_at VARCHAR(50),
                    UNIQUE KEY unique_pk (category(191), question_hash, model_a, model_b)
                ) ENGINE=InnoDB DEFAULT CHARSET={self.charset}
            """)
            
            try:
                cursor.execute(f"""
                    CREATE INDEX idx_{self.model_outputs_table}_task
                    ON `{self.model_outputs_table}` (task_id)
                """)
            except pymysql.err.OperationalError as e:
                if e.args[0] != 1061:
                    logger.warning(f"创建索引 idx_{self.model_outputs_table}_task 时出错: {e}")
            
            try:
                cursor.execute(f"""
                    CREATE INDEX idx_{self.pk_results_table}_task
                    ON `{self.pk_results_table}` (task_id)
                """)
            except pymysql.err.OperationalError as e:
                if e.args[0] != 1061:
                    logger.warning(f"创建索引 idx_{self.pk_results_table}_task 时出错: {e}")
            
            conn.commit()
            cursor.close()
            
        logger.info("MySQL 表结构已就绪")
    
    @staticmethod
    def _compute_hash(text: str) -> str:
        """计算文本的MD5哈希值"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_model_output(
        self, 
        model: str, 
        category: Optional[str], 
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """获取模型输出（同步）"""
        category_val = self._normalize_value(category)
        prompt_val = self._normalize_value(prompt)
        prompt_hash = self._compute_hash(prompt)
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT payload_json FROM `{self.model_outputs_table}`
                WHERE model = %s AND category <=> %s AND prompt_hash = %s
                LIMIT 1
                """,
                (model, category_val, prompt_hash),
            )
            row = cursor.fetchone()
            cursor.close()
        
        if not row:
            return None
        return json.loads(row["payload_json"])
    
    async def aget_model_output(
        self, 
        model: str, 
        category: Optional[str], 
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """获取模型输出（异步）"""
        return await asyncio.to_thread(
            self.get_model_output, model, category, prompt
        )
    
    def save_model_output(self, payload: Dict[str, Any]) -> None:
        """保存模型输出（同步）"""
        model = payload.get("model")
        category = self._normalize_value(payload.get("category"))
        prompt = self._normalize_value(payload.get("prompt"))
        prompt_hash = self._compute_hash(prompt or "")
        status = payload.get("status")
        now = int(time.time())
        created_at = payload.get("created_at")
        created_ts = created_at if isinstance(created_at, int) else now
        data_json = json.dumps(payload, ensure_ascii=False)
        task_id = self.task_id or payload.get("task_id")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"""
                        INSERT INTO `{self.model_outputs_table}`
                            (task_id, model_name, model, category, prompt, prompt_hash, 
                             status, payload_json, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            task_id=VALUES(task_id),
                            model_name=VALUES(model_name),
                            status=VALUES(status),
                            payload_json=VALUES(payload_json),
                            updated_at=VALUES(updated_at)
                        """,
                        (
                            task_id,
                            payload.get("model_name"),
                            model,
                            category,
                            prompt,
                            prompt_hash,
                            status,
                            data_json,
                            created_ts,
                            now,
                        ),
                    )
                    conn.commit()
                    cursor.close()
                    return
            except pymysql.err.OperationalError as e:
                if "Deadlock" in str(e) and attempt < max_retries - 1:
                    # 死锁重试
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"保存模型输出失败: {e}")
                    raise
            except Exception as e:
                logger.error(f"保存模型输出异常: {e}")
                raise
    
    async def asave_model_output(self, payload: Dict[str, Any]) -> None:
        """保存模型输出（异步）"""
        await asyncio.to_thread(self.save_model_output, payload)
    
    def batch_save_model_outputs(self, payloads: List[Dict[str, Any]]) -> int:
        """批量保存模型输出
        """
        if not payloads:
            return 0
        
        batch_data = []
        for payload in payloads:
            model = payload.get("model")
            category = self._normalize_value(payload.get("category"))
            prompt = self._normalize_value(payload.get("prompt"))
            prompt_hash = self._compute_hash(prompt or "")
            status = payload.get("status")
            now = int(time.time())
            created_at = payload.get("created_at")
            created_ts = created_at if isinstance(created_at, int) else now
            data_json = json.dumps(payload, ensure_ascii=False)
            task_id = self.task_id or payload.get("task_id")
            
            batch_data.append((
                task_id,
                payload.get("model_name"),
                model,
                category,
                prompt,
                prompt_hash,
                status,
                data_json,
                created_ts,
                now,
            ))
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        f"""
                        INSERT INTO `{self.model_outputs_table}`
                            (task_id, model_name, model, category, prompt, prompt_hash, 
                             status, payload_json, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            task_id=VALUES(task_id),
                            model_name=VALUES(model_name),
                            status=VALUES(status),
                            payload_json=VALUES(payload_json),
                            updated_at=VALUES(updated_at)
                        """,
                        batch_data
                    )
                    conn.commit()
                    cursor.close()
                    return len(batch_data)
            except pymysql.err.OperationalError as e:
                if "Deadlock" in str(e) and attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"批量保存模型输出失败: {e}")
                    raise
            except Exception as e:
                logger.error(f"批量保存模型输出异常: {e}")
                raise
        
        return 0
    
    def get_pk_result(
        self,
        category: Optional[str],
        question: str,
        model_a: str,
        model_b: str,
    ) -> Optional[Dict[str, Any]]:
        """获取PK结果"""
        category_val = self._normalize_value(category)
        question_val = self._normalize_value(question)
        question_hash = self._compute_hash(question)
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT result_json FROM `{self.pk_results_table}`
                WHERE category <=> %s
                  AND question_hash = %s
                  AND model_a = %s
                  AND model_b = %s
                LIMIT 1
                """,
                (category_val, question_hash, model_a, model_b),
            )
            row = cursor.fetchone()
            cursor.close()
        
        if not row:
            return None
        return json.loads(row["result_json"])
    
    def save_pk_result(
        self,
        evaluation_dimension: str,
        category: Optional[str],
        question: str,
        model_a: str,
        model_b: str,
        payload: Dict[str, Any],
    ) -> None:
        """保存PK结果"""
        category_val = self._normalize_value(category)
        question_val = self._normalize_value(question)
        question_hash = self._compute_hash(question)
        result_json = json.dumps(payload, ensure_ascii=False)
        created_at = payload.get("current_time") or time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        task_id = self.task_id or payload.get("task_id")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        f"""
                        INSERT INTO `{self.pk_results_table}`
                            (task_id, evaluation_dimension, category, question, 
                             question_hash, model_a, model_b, winner, result_json, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            task_id=VALUES(task_id),
                            evaluation_dimension=VALUES(evaluation_dimension),
                            winner=VALUES(winner),
                            result_json=VALUES(result_json),
                            created_at=VALUES(created_at)
                        """,
                        (
                            task_id,
                            evaluation_dimension,
                            category_val,
                            question_val,
                            question_hash,
                            model_a,
                            model_b,
                            payload.get("winner"),
                            result_json,
                            created_at,
                        ),
                    )
                    conn.commit()
                    cursor.close()
                    return
            except pymysql.err.OperationalError as e:
                if "Deadlock" in str(e) and attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"保存PK结果失败: {e}")
                    raise
            except Exception as e:
                logger.error(f"保存PK结果异常: {e}")
                raise
    
    def batch_save_pk_results(
        self,
        records: List[Dict[str, Any]]
    ) -> int:
        """批量保存PK结果
        """
        if not records:
            return 0
        
        batch_data = []
        for record in records:
            evaluation_dimension = record['evaluation_dimension']
            category = self._normalize_value(record.get('category'))
            question = self._normalize_value(record['question'])
            question_hash = self._compute_hash(question)
            model_a = record['model_a']
            model_b = record['model_b']
            payload = record['payload']
            
            result_json = json.dumps(payload, ensure_ascii=False)
            created_at = payload.get("current_time") or time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            )
            task_id = self.task_id or payload.get("task_id")
            
            batch_data.append((
                task_id,
                evaluation_dimension,
                category,
                question,
                question_hash,
                model_a,
                model_b,
                payload.get("winner"),
                result_json,
                created_at,
            ))
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._connect() as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        f"""
                        INSERT INTO `{self.pk_results_table}`
                            (task_id, evaluation_dimension, category, question, 
                             question_hash, model_a, model_b, winner, result_json, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            task_id=VALUES(task_id),
                            evaluation_dimension=VALUES(evaluation_dimension),
                            winner=VALUES(winner),
                            result_json=VALUES(result_json),
                            created_at=VALUES(created_at)
                        """,
                        batch_data
                    )
                    conn.commit()
                    cursor.close()
                    return len(batch_data)
            except pymysql.err.OperationalError as e:
                if "Deadlock" in str(e) and attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"批量保存PK结果失败: {e}")
                    raise
            except Exception as e:
                logger.error(f"批量保存PK结果异常: {e}")
                raise
        
        return 0
    
    def save_task_info(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """保存任务信息"""
        task_info_json = json.dumps(task_info, ensure_ascii=False)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO `{self.tasks_table}`
                    (task_id, created_at, config_path, eval_run_name, 
                     task_info_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    config_path=VALUES(config_path),
                    eval_run_name=VALUES(eval_run_name),
                    task_info_json=VALUES(task_info_json),
                    updated_at=VALUES(updated_at)
                """,
                (
                    task_id,
                    task_info.get('created_at', now),
                    task_info.get('config_path'),
                    task_info.get('eval_run_name'),
                    task_info_json,
                    now,
                ),
            )
            conn.commit()
            cursor.close()
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT task_info_json FROM `{self.tasks_table}`
                WHERE task_id = %s
                LIMIT 1
                """,
                (task_id,),
            )
            row = cursor.fetchone()
            cursor.close()
        
        if not row:
            return None
        
        return json.loads(row["task_info_json"])
    
    def list_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """列出最近的任务"""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT task_id, created_at, eval_run_name, config_path
                FROM `{self.tasks_table}`
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            cursor.close()
        
        return [
            {
                'task_id': row['task_id'],
                'created_at': row['created_at'],
                'eval_run_name': row['eval_run_name'],
                'config_path': row['config_path'],
            }
            for row in rows
        ]

