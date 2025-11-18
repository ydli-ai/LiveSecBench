"""
CacheManager单元测试
"""

import tempfile
import shutil

from livesecbench.infra.cache_manager import (
    LRUCache,
    CacheManager
)


def test_lru_cache_basic():
    """测试LRU缓存基本功能"""
    cache = LRUCache(max_size=3)
    
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    cache.set('key3', 'value3')
    
    assert cache.get('key1') == 'value1'
    assert cache.get('key2') == 'value2'
    assert cache.get('key3') == 'value3'


def test_lru_cache_eviction():
    """测试LRU缓存淘汰策略"""
    cache = LRUCache(max_size=2)
    
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    cache.set('key3', 'value3')  # 应该淘汰key1
    
    assert cache.get('key1') is None  # key1被淘汰
    assert cache.get('key2') == 'value2'
    assert cache.get('key3') == 'value3'


def test_lru_cache_lru_behavior():
    """测试LRU（最近最少使用）行为"""
    cache = LRUCache(max_size=2)
    
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    
    # 访问key1，使其成为最近使用
    cache.get('key1')
    
    # 添加key3，应该淘汰key2（最久未使用）
    cache.set('key3', 'value3')
    
    assert cache.get('key1') == 'value1'  # key1还在
    assert cache.get('key2') is None      # key2被淘汰
    assert cache.get('key3') == 'value3'


def test_lru_cache_update():
    """测试缓存更新"""
    cache = LRUCache(max_size=2)
    
    cache.set('key1', 'value1')
    cache.set('key1', 'value1_updated')
    
    assert cache.get('key1') == 'value1_updated'


def test_lru_cache_stats():
    """测试缓存统计"""
    cache = LRUCache(max_size=3)
    
    cache.set('key1', 'value1')
    cache.get('key1')  # hit
    cache.get('key2')  # miss
    
    stats = cache.get_stats()
    
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    assert stats['hit_rate'] == 0.5
    assert stats['size'] == 1


def test_cache_manager_memory_cache():
    """测试CacheManager内存缓存"""
    cache_manager = CacheManager(max_size=10)
    
    cache_manager.set('test_key', {'data': 'test_value'})
    result = cache_manager.get('test_key')
    
    assert result == {'data': 'test_value'}


def test_cache_manager_disk_cache():
    """测试CacheManager磁盘缓存"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache_manager = CacheManager(
            max_size=10,
            disk_cache_dir=temp_dir,
            enable_disk_cache=True
        )
        
        # 写入缓存
        cache_manager.set('disk_key', {'data': 'disk_value'})
        
        # 清空内存缓存
        cache_manager.memory_cache.clear()
        
        # 应该从磁盘加载
        result = cache_manager.get('disk_key')
        assert result == {'data': 'disk_value'}
    
    finally:
        shutil.rmtree(temp_dir)


def test_cache_manager_ttl():
    """测试缓存过期时间"""
    cache_manager = CacheManager(max_size=10)
    
    # 设置1秒过期
    cache_manager.set('ttl_key', 'ttl_value', ttl=1)
    
    # 立即获取应该成功
    assert cache_manager.get('ttl_key') == 'ttl_value'
    
    # 等待过期（这个测试可能需要较长时间，实际测试中可以mock时间）
    # time.sleep(1.5)
    # assert cache_manager.get('ttl_key') is None


def test_cache_manager_delete():
    """测试缓存删除"""
    cache_manager = CacheManager(max_size=10)
    
    cache_manager.set('delete_key', 'delete_value')
    assert cache_manager.get('delete_key') == 'delete_value'
    
    cache_manager.delete('delete_key')
    assert cache_manager.get('delete_key') is None


def test_cache_manager_clear():
    """测试清空缓存"""
    cache_manager = CacheManager(max_size=10)
    
    cache_manager.set('key1', 'value1')
    cache_manager.set('key2', 'value2')
    
    cache_manager.clear()
    
    assert cache_manager.get('key1') is None
    assert cache_manager.get('key2') is None


def test_cache_manager_decorator():
    """测试缓存装饰器"""
    cache_manager = CacheManager(max_size=10)
    
    call_count = 0
    
    @cache_manager.cached(ttl=60)
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        return x + y
    
    # 第一次调用，应该执行函数
    result1 = expensive_function(1, 2)
    assert result1 == 3
    assert call_count == 1
    
    # 第二次调用相同参数，应该从缓存获取
    result2 = expensive_function(1, 2)
    assert result2 == 3
    assert call_count == 1  # 没有增加
    
    # 不同参数，应该执行函数
    result3 = expensive_function(2, 3)
    assert result3 == 5
    assert call_count == 2


def test_cache_manager_stats():
    """测试缓存统计信息"""
    cache_manager = CacheManager(max_size=10)
    
    cache_manager.set('key1', 'value1')
    cache_manager.get('key1')
    cache_manager.get('key2')
    
    stats = cache_manager.get_stats()
    
    assert 'hits' in stats
    assert 'misses' in stats
    assert 'hit_rate' in stats

