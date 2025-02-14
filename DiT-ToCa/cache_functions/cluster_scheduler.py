import torch
import math
def cluster_scheduler(cache_dic, current):
    '''
    1 - 50: 3 - 1
    1 - 50: 5 - 1
    1 - 50: 7 - 1
    '''
    return int(cache_dic['current_cluster_nums'][current['step']]), round(5 - current['step'] / 12)