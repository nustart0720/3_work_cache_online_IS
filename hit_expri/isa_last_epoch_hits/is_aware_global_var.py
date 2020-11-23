# 全局变量管理器，也放在和folder.py一样的目录下
import heapq

def _init(update_dict, len_samples): #初始化，只在folder.py中执行一次_init函数
    print('**********')
    global total_images, last_epoch_is, sample_is_key
    total_images = len_samples
    last_epoch_is = {} # 计算在读取某个图片时，上一轮它的重要性 store k:v->path:imp
    sample_is_key = [] # store tuple: (imp,path)
    new_sample_is_key = []
    heapq.heapify(sample_is_key)

    for k,v in update_dict.items():
        last_epoch_is[k] = v
 
    
# def update_last_epoch_is(update_dict):
#     for k,v in update_dict.items():
#         last_epoch_is[k] = v