"""
缓存管理器
提供内存缓存（LRU）、磁盘缓存、过期策略
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Callable
from collections import OrderedDict
from functools import wraps


class LRUCache:
    """LRU（Least Recently Used）缓存实现: 支持容量限制和自动淘汰"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = {
            'value': value,
            'ttl': ttl,
            'timestamp': time.time()
        }
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def delete(self, key: str):
        """删除缓存项"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class CacheManager:
    """统一缓存管理器: 支持多种缓存策略和持久化选项"""
    
    def __init__(
        self,
        max_size: int = 1000,
        disk_cache_dir: Optional[str] = None,
        enable_disk_cache: bool = False
    ):
        self.memory_cache = LRUCache(max_size=max_size)
        self.enable_disk_cache = enable_disk_cache
        
        if enable_disk_cache and disk_cache_dir:
            self.disk_cache_dir = Path(disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_cache_dir = None
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值（优先从内存缓存获取，失败则从磁盘缓存获取）"""
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        if self.enable_disk_cache and self.disk_cache_dir:
            value = self._load_from_disk(key)
            if value is not None:
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值（同时写入内存缓存和磁盘缓存）"""
        self.memory_cache.set(key, value, ttl)
        
        if self.enable_disk_cache and self.disk_cache_dir:
            self._save_to_disk(key, value, ttl)
    
    def delete(self, key: str):
        """删除缓存项（同时删除内存和磁盘）"""
        self.memory_cache.delete(key)
        
        if self.enable_disk_cache and self.disk_cache_dir:
            cache_file = self._get_disk_cache_path(key)
            if cache_file.exists():
                cache_file.unlink()
    
    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        
        if self.enable_disk_cache and self.disk_cache_dir:
            for cache_file in self.disk_cache_dir.glob('*.cache'):
                cache_file.unlink()
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        stats = self.memory_cache.get_stats()
        
        if self.enable_disk_cache and self.disk_cache_dir:
            disk_cache_files = list(self.disk_cache_dir.glob('*.cache'))
            stats['disk_cache_size'] = len(disk_cache_files)
        
        return stats
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_disk_cache_path(self, key: str) -> Path:
        """获取磁盘缓存文件路径"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.disk_cache_dir / f"{safe_key}.cache"
    
    def _save_to_disk(self, key: str, value: Any, ttl: Optional[int]):
        """保存缓存到磁盘"""
        cache_file = self._get_disk_cache_path(key)
        
        cache_data = {
            'value': value,
            'ttl': ttl,
            'timestamp': time.time()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """从磁盘加载缓存"""
        cache_file = self._get_disk_cache_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data['ttl']:
                elapsed = time.time() - cache_data['timestamp']
                if elapsed > cache_data['ttl']:
                    cache_file.unlink()
                    return None
            
            return cache_data['value']
        
        except Exception:
            # 加载失败，删除损坏的缓存文件
            if cache_file.exists():
                cache_file.unlink()
            return None


_global_cache: Optional[CacheManager] = None


def get_cache_manager(
    max_size: int = 1000,
    disk_cache_dir: Optional[str] = None,
    enable_disk_cache: bool = False
) -> CacheManager:
    """获取全局缓存管理器实例"""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = CacheManager(
            max_size=max_size,
            disk_cache_dir=disk_cache_dir,
            enable_disk_cache=enable_disk_cache
        )
    
    return _global_cache
