import heapq

def init(cache_size=12500):
    print('##########')
    global sample_is_key,cache,helper_batch_cache
    cache = {}
    helper_batch_cache = {}
    # sample_is_key = [(0,'')]*cache_size # store tuple (imp.path)
    sample_is_key = []
    
    heapq.heapify(sample_is_key)

def sample_is_key_init(path_target,cache_size=12500):
    cnt = 0
    for path,_ in path_target:
        sample_is_key.append((0,path))
        cnt += 1
        if cnt>=cache_size:
            break
    print(f'sample_is_key initialized, items:{len(sample_is_key)}')