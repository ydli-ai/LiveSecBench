import asyncio
import json
import sqlite3
import time
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List

from livesecbench.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteStorage:
    """SQLite评分数据读写操作封装"""

    def __init__(
        self,
        db_path: str,
        model_outputs_table: str = "model_outputs",
        pk_results_table: str = "pk_results",
        tasks_table: str = "evaluation_tasks",
        task_id: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.model_outputs_table = self._sanitize_identifier(model_outputs_table)
        self.pk_results_table = self._sanitize_identifier(pk_results_table)
        self.tasks_table = self._sanitize_identifier(tasks_table)
        self.task_id = task_id
        
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        if not value or not value.replace("_", "").isalnum():
            raise ValueError(f"非法的SQLite标识符: {value}")
        return value

    @staticmethod
    def _normalize_value(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return str(value[0])
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        # 启用 WAL 模式以支持更好的并发读写
        conn.execute("PRAGMA journal_mode=WAL")
        # 设置繁忙超时（毫秒）
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.tasks_table} (
                    task_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config_path TEXT,
                    eval_run_name TEXT,
                    task_info_json TEXT,
                    updated_at TEXT
                );
                """
            )
            
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.model_outputs_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    model_name TEXT,
                    model TEXT NOT NULL,
                    category TEXT,
                    prompt TEXT,
                    status TEXT,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER,
                    updated_at INTEGER,
                    UNIQUE(model, category, prompt)
                );
                """
            )
            
            try:
                conn.execute(f"ALTER TABLE {self.model_outputs_table} ADD COLUMN task_id TEXT;")
            except sqlite3.OperationalError:
                pass
            
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.pk_results_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    evaluation_dimension TEXT,
                    category TEXT,
                    question TEXT,
                    model_a TEXT,
                    model_b TEXT,
                    winner TEXT,
                    result_json TEXT NOT NULL,
                    created_at TEXT,
                    UNIQUE(category, question, model_a, model_b)
                );
                """
            )
            
            try:
                conn.execute(f"ALTER TABLE {self.pk_results_table} ADD COLUMN task_id TEXT;")
            except sqlite3.OperationalError:
                pass
            
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.model_outputs_table}_lookup
                ON {self.model_outputs_table} (model, category, prompt);
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.model_outputs_table}_task
                ON {self.model_outputs_table} (task_id);
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pk_results_table}_lookup
                ON {self.pk_results_table} (category, question, model_a, model_b);
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pk_results_table}_task
                ON {self.pk_results_table} (task_id);
                """
            )

    def get_model_output(self, model: str, category: Optional[str], prompt: str) -> Optional[Dict[str, Any]]:
        return self._get_model_output_sync(model, category, prompt)

    def _get_model_output_sync(self, model: str, category: Optional[str], prompt: str) -> Optional[Dict[str, Any]]:
        category_val = self._normalize_value(category)
        prompt_val = self._normalize_value(prompt)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT payload_json FROM {self.model_outputs_table}
                WHERE model = ? AND category IS ?
                      AND prompt = ?
                LIMIT 1;
                """,
                (model, category_val, prompt_val),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])

    async def aget_model_output(self, model: str, category: Optional[str], prompt: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_model_output_sync, model, category, prompt)

    def save_model_output(self, payload: Dict[str, Any]) -> None:
        self._save_model_output_sync(payload)

    def _save_model_output_sync(self, payload: Dict[str, Any]) -> None:
        model = payload.get("model")
        category = self._normalize_value(payload.get("category"))
        prompt = self._normalize_value(payload.get("prompt"))
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
                    conn.execute(
                        f"""
                        INSERT INTO {self.model_outputs_table}
                            (task_id, model_name, model, category, prompt, status, payload_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(model, category, prompt) DO UPDATE SET
                            task_id=excluded.task_id,
                            model_name=excluded.model_name,
                            status=excluded.status,
                            payload_json=excluded.payload_json,
                            updated_at=excluded.updated_at;
                        """,
                        (
                            task_id,
                            payload.get("model_name"),
                            model,
                            category,
                            prompt,
                            status,
                            data_json,
                            created_ts,
                            now,
                        ),
                    )
                    return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # 指数退避重试
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    raise

    async def asave_model_output(self, payload: Dict[str, Any]) -> None:
        """异步保存模型输出"""
        await asyncio.to_thread(self._save_model_output_sync, payload)

    def get_pk_result(
        self,
        category: Optional[str],
        question: str,
        model_a: str,
        model_b: str,
    ) -> Optional[Dict[str, Any]]:
        category_val = self._normalize_value(category)
        question_val = self._normalize_value(question)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT result_json FROM {self.pk_results_table}
                WHERE category IS ?
                  AND question = ?
                  AND model_a = ?
                  AND model_b = ?
                LIMIT 1;
                """,
                (category_val, question_val, model_a, model_b),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["result_json"])

    def _save_pk_result_from_queue(self, data: Dict[str, Any]) -> None:
        """从队列数据恢复并保存 PK 结果"""
        self.save_pk_result(
            evaluation_dimension=data['evaluation_dimension'],
            category=data['category'],
            question=data['question'],
            model_a=data['model_a'],
            model_b=data['model_b'],
            payload=data['payload'],
        )

    def save_pk_result(
        self,
        evaluation_dimension: str,
        category: Optional[str],
        question: str,
        model_a: str,
        model_b: str,
        payload: Dict[str, Any],
    ) -> None:
        category_val = self._normalize_value(category)
        question_val = self._normalize_value(question)
        result_json = json.dumps(payload, ensure_ascii=False)
        created_at = payload.get("current_time") or time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        task_id = self.task_id or payload.get("task_id")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self._connect() as conn:
                    try:
                        conn.execute(
                            f"""
                            INSERT INTO {self.pk_results_table}
                                (task_id, evaluation_dimension, category, question, model_a, model_b, winner, result_json, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(category, question, model_a, model_b) DO UPDATE SET
                                task_id=excluded.task_id,
                                evaluation_dimension=excluded.evaluation_dimension,
                                winner=excluded.winner,
                                result_json=excluded.result_json,
                                created_at=excluded.created_at;
                            """,
                            (
                                task_id,
                                evaluation_dimension,
                                category_val,
                                question_val,
                                model_a,
                                model_b,
                                payload.get("winner"),
                                result_json,
                                created_at,
                            ),
                        )
                        conn.commit()
                        return
                    except Exception as e:
                        if isinstance(e, sqlite3.OperationalError) and "ON CONFLICT" in str(e):
                            conn.execute(
                                f"""
                                DELETE FROM {self.pk_results_table}
                                WHERE category = ? AND question = ? AND model_a = ? AND model_b = ?
                                """,
                                (category_val, question_val, model_a, model_b),
                            )
                            # 再插入新记录
                            conn.execute(
                                f"""
                                INSERT INTO {self.pk_results_table}
                                    (task_id, evaluation_dimension, category, question, model_a, model_b, winner, result_json, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    task_id,
                                    evaluation_dimension,
                                    category_val,
                                    question_val,
                                    model_a,
                                    model_b,
                                    payload.get("winner"),
                                    result_json,
                                    created_at,
                                ),
                            )
                            conn.commit()
                            return
                        else:
                            raise
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # 指数退避重试
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    raise
    
    def save_task_info(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """
        保存任务信息到数据库
        
        Args:
            task_id: 任务ID
            task_info: 任务信息字典
        """
        import sqlite3
        task_info_json = json.dumps(task_info, ensure_ascii=False)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.tasks_table}
                    (task_id, created_at, config_path, eval_run_name, task_info_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    config_path=excluded.config_path,
                    eval_run_name=excluded.eval_run_name,
                    task_info_json=excluded.task_info_json,
                    updated_at=excluded.updated_at;
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
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT task_info_json FROM {self.tasks_table}
                WHERE task_id = ?
                LIMIT 1;
                """,
                (task_id,),
            ).fetchone()
        
        if not row:
            return None
        
        return json.loads(row["task_info_json"])
    
    def list_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT task_id, created_at, eval_run_name, config_path
                FROM {self.tasks_table}
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        
        return [
            {
                'task_id': row['task_id'],
                'created_at': row['created_at'],
                'eval_run_name': row['eval_run_name'],
                'config_path': row['config_path'],
            }
            for row in rows
        ]

