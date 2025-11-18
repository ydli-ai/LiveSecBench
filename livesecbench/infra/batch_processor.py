"""
批量处理器
提供批量API调用、数据库操作、进度跟踪
"""

import asyncio
from typing import List, Callable, Any, Optional, TypeVar, Coroutine
from dataclasses import dataclass
import time


T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult:
    """批处理结果"""
    successes: List[Any]
    failures: List[tuple]  # (item, exception)
    total: int
    success_count: int
    failure_count: int
    elapsed_time: float
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total if self.total > 0 else 0.0


class BatchProcessor:
    """批量处理器: 支持自动分批、并发控制、错误处理、进度跟踪"""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 5,
        show_progress: bool = True,
        fail_fast: bool = False
    ):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.show_progress = show_progress
        self.fail_fast = fail_fast
    
    async def process_batch(
        self,
        items: List[T],
        processor_func: Callable[[T], Coroutine[Any, Any, R]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """批量处理项目"""
        start_time = time.time()
        
        successes = []
        failures = []
        total = len(items)
        processed = 0
        
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            tasks = [processor_func(item) for item in batch]
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_with_semaphore(task, item):
                async with semaphore:
                    try:
                        result = await task
                        return ('success', result)
                    except Exception as e:
                        if self.fail_fast:
                            raise
                        return ('failure', (item, e))
            
            batch_results = await asyncio.gather(
                *[process_with_semaphore(task, item) for task, item in zip(tasks, batch)],
                return_exceptions=not self.fail_fast
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    failures.append((None, result))
                elif result[0] == 'success':
                    successes.append(result[1])
                else:
                    failures.append(result[1])
            
            processed += len(batch)
            
            if progress_callback:
                progress_callback(processed, total)
            
            if self.show_progress:
                progress_pct = (processed / total) * 100
                print(f"\r处理进度: {processed}/{total} ({progress_pct:.1f}%)", end='', flush=True)
        
        if self.show_progress:
            print()
        
        elapsed_time = time.time() - start_time
        
        return BatchResult(
            successes=successes,
            failures=failures,
            total=total,
            success_count=len(successes),
            failure_count=len(failures),
            elapsed_time=elapsed_time
        )
    
    async def process_with_retry(
        self,
        items: List[T],
        processor_func: Callable[[T], Coroutine[Any, Any, R]],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> BatchResult:
        """带重试机制的批量处理"""
        async def processor_with_retry(item: T) -> R:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await processor_func(item)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    else:
                        raise last_exception
        
        return await self.process_batch(items, processor_with_retry)


class DatabaseBatchWriter:
    """数据库批量写入器"""
    
    def __init__(
        self,
        storage: Any,
        batch_size: int = 100,
        auto_flush: bool = True,
        batch_write_func: Optional[Callable[[List[Any]], None]] = None,
        write_func: Optional[Callable[[Any], None]] = None,
    ):
        self.storage = storage
        self.batch_size = batch_size
        self.auto_flush = auto_flush
        self.buffer = []
        self.total_written = 0
        self.batch_write_func = batch_write_func or getattr(storage, 'batch_insert', None)
        fallback_write = write_func or getattr(storage, 'insert', None)
        if fallback_write is None and hasattr(storage, 'save_model_output'):
            fallback_write = getattr(storage, 'save_model_output')
        self.write_func = fallback_write
        if self.batch_write_func is None and self.write_func is None:
            raise ValueError(
                "storage 未提供 batch_insert/insert/save_model_output，"
                "请通过 write_func 或 batch_write_func 显式传入写入方法"
            )
    
    def add(self, item: Any):
        self.buffer.append(item)
        
        if self.auto_flush and len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        
        if self.batch_write_func:
            self.batch_write_func(self.buffer)
        else:
            for item in self.buffer:
                self.write_func(item)
        
        self.total_written += len(self.buffer)
        self.buffer.clear()
    
    def get_stats(self) -> dict:
        return {
            'total_written': self.total_written,
            'buffer_size': len(self.buffer),
            'batch_size': self.batch_size
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def async_batch_map(
    items: List[T],
    func: Callable[[T], Coroutine[Any, Any, R]],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[R]:
    """异步批量映射"""
    processor = BatchProcessor(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        show_progress=False
    )
    
    result = await processor.process_batch(items, func)
    return result.successes
