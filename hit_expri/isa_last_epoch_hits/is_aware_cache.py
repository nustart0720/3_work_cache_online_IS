# 此文件放在和folder.py同目录下，即torchvision.dataset
import heapq
from threading import RLock
from torchvision.datasets import is_aware_global_var as glv
from collections import namedtuple

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

# total_images = 50000
# last_epoch_is = {} # 计算在读取某个图片时，上一轮它的重要性

# 提供可以更新last_epoch_is的函数接口
def update_last_epoch_is(update_dict):
    # global last_epoch_is
    for k,v in update_dict.items():
        # glv.last_epoch_is[k] = (glv.last_epoch_is[k]+v)/2
        glv.last_epoch_is[k] = v
    # is_aware_global_var.update_last_epoch_is(update_dict)

# sample_is_key = [] # 记录样本的重要性和对应的路径，里面元素的放置顺序就是(importance, path)
# heapq.heapify(sample_is_key)
def reconstr_sample_is_heapq():
    glv.new_sample_is_key = []
    for i in range(len(glv.sample_is_key)):
        # glv.sample_is_key[i][0] = glv.last_epoch_is[glv.sample_is_key[i][1]]
        glv.new_sample_is_key.append((glv.last_epoch_is[glv.sample_is_key[i][1]], glv.sample_is_key[i][1]))
    glv.sample_is_key = glv.new_sample_is_key
    heapq.heapify(glv.sample_is_key)
    print(f'reheap glv.sample_is_key and size is:{len(glv.sample_is_key)}')

def is_aware_cache(maxsize=128, typed=False):
    
    if maxsize is not None and not isinstance(maxsize, int):
        raise TypeError('Expected maxsize to be an integer or None')

    def decorating_function(user_function):
        wrapper = _isaware_cache_wrapper(user_function, maxsize)
        return wrapper

    return decorating_function


def _isaware_cache_wrapper(user_function, maxsize):
    sentinel = object() # 判断字典中是否有某个元素

    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get    # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()           # because linkedlist updates aren't threadsafe

    if maxsize == 0:
        # 不缓存，直接每次返回userfunction即可
        def wrapper(*args, **kwds):
            nonlocal misses
            result = user_function(*args, **kwds)
            misses += 1
            return result
    elif maxsize == None:
        # 缓存无限大，每次都缓存，也不用使用到堆
        def wrapper(*args, **kwds):
            nonlocal hits, misses
            key = args[0] # 因为这个场景中,key就是文件的路径，所以本身就是唯一的
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            result = user_function(*args, **kwds)
            cache[key] = result
            misses += 1
            return result
    else:
        # 使用maxsize大小的堆判断是否需要缓存和逐出，堆中的元素是元组，（imp和key）
        def wrapper(*args, **kwds):
            nonlocal hits, misses, full
            # global sample_is_key,last_epoch_is
            key = args[0]
            with lock:
                result = cache_get(key, sentinel)
                if result is not sentinel:
                    # print('in cache and just return')
                    hits += 1
                    return result
                misses += 1
            result = user_function(*args, **kwds)
            # 使用上一轮的重要性判断一下这轮的读取和convert操作是否要进行缓存
            with lock:
                if not full:
                    # print('not in cache and not full')
                    cache[key] = result # cache没满，进行缓存
                    heapq.heappush(glv.sample_is_key, (glv.last_epoch_is[key],key)) # cache没有满，就把（importance, path）放堆中
                    # print(glv.sample_is_key[0][0])
                    full = (cache_len() >= maxsize)
                else:
                    # print('not in cache but full')
                    # print(key, glv.last_epoch_is[key])
                    # cache已经满了，那么看一下目前堆中的最小元素和当前访问的元素的上一轮的importance相比，
                    # 如果现在的更大，那么替换掉，否则保持不动
                    if glv.sample_is_key[0][0] < glv.last_epoch_is[key]:
                        # print('replace')
                        _, delpath = heapq.heapreplace(glv.sample_is_key, (glv.last_epoch_is[key],key)) # 先pop出来is最小的，然后push进去新的元组;返回的是pop出来的元素
                        cache[key] = result
                        del cache[delpath]
            return result
        
    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())
    
    wrapper.cache_info = cache_info
    return wrapper
