from torchvision.datasets import new_isc_global_var as glv
import heapq

def isc_update_cache_is(update_dict):
    # 如果在cache中，那么直接更新heap；不在cache中，和最小堆顶上的比，如果大于就替换掉，否则不动
    # 然后重新建堆

    
    in_sample = 0
    for i in range(len(glv.sample_is_key)):
        if glv.sample_is_key[i][1] in update_dict:
            # if glv.sample_is_key[i][1] == glv.sample_is_key[-1][1]:
            #     print('正要更新目前最大的sample_is：',glv.sample_is_key[i][1],'from',glv.sample_is_key[-1][0],'to',update_dict[glv.sample_is_key[i][1]])
            glv.sample_is_key[i]=(update_dict[glv.sample_is_key[i][1]], glv.sample_is_key[i][1]) # 更新sample_is_key中的元组
            in_sample += 1
            # 因为folder.py中sample_is_key_init的时候没有放入cache
            if glv.sample_is_key[i][1] not in glv.cache:
                # print('&&&&&&&&&&&&&&&&&')
                k = glv.sample_is_key[i][1]
                glv.cache[k] = glv.helper_batch_cache[k]

        
    # print(f'update sample_is_key:{in_sample}')
    heapq.heapify(glv.sample_is_key) # 重新堆化


    sample_is_key_path = [v for _,v in glv.sample_is_key]
    replace = 0
    for k,v in update_dict.items():
        if k not in sample_is_key_path:
            if v > glv.sample_is_key[0][0]:
                # print('replace')
                replace += 1
                _,path = heapq.heapreplace(glv.sample_is_key, (v,k))
                if path in glv.cache:
                    del glv.cache[path]
                glv.cache[k] = glv.helper_batch_cache[k]
            else:
                # print('not replace')
                pass
    # print(f'sample_is_key[0][0]:{glv.sample_is_key[0][0]}, sample_is_key[-1][0]:{glv.sample_is_key[-1][0]}')
    # print(f'replace num: {replace}')
    # print('sample_is_key size:',len(glv.sample_is_key),'cache size:',len(glv.cache))
                



