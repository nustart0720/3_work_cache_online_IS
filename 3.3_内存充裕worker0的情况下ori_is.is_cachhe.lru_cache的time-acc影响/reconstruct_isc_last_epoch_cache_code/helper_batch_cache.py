import new_isc_global_var as glv
from threading import RLock
from collections import namedtuple
from threading import RLock

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses"])


def helper_batch_cache():
    glv.init()
    def decorating_function(user_function):
        wrapper = _helper_cache_wrapper(user_function)
        return wrapper

    return decorating_function

def free_helper_cache():
    glv.helper_batch_cache.clear()

def _helper_cache_wrapper(user_function):
    cache_get = glv.cache.get
    sentinel = object()
    hits = misses = 0
    lock = RLock()

    def wrapper(*args, **kwds):
        nonlocal hits,misses
        key = args[0]
        with lock:
            result = cache_get(key, sentinel)
            if result is not sentinel:
                # cache 有该元素
                hits += 1
                return result
            misses += 1
        result = user_function(*args, **kwds)
        with lock:
            glv.helper_batch_cache[key] = result # 由于每个iteration都会释放helper_batch_cache，所以可以都直接放进去
                                                 # 这里临时缓存后之后train中决定是否要放入glv.cache中真正缓存
        
        return result
    
    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses)
    
    wrapper.cache_info = cache_info
    return wrapper


    
