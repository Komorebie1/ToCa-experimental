import torch
import math
def cluster_scheduler(cache_dic, current):

    return int(cache_dic['current_cluster_nums'][current['step']]), round(3 - current['step'] / 25)