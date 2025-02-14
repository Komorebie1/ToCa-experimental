from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
from .token_merge import token_merge
from .cluster_scheduler import cluster_scheduler
import torch
import torch.nn.functional as F
def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio), 0.0, 1.0)
    # Generate the index tensor for fresh tokens
    score = score_evaluate(cache_dic, tokens, current)
    # score = local_selection_with_bonus(score, 0.6, 2) # Uniform Spatial Distribution s4 mentioned in the paper
    # score = consider_neighbor_score(score, 1)

    #######################################
    # # 0.6, 2
    # indices = score.argsort(dim=-1, descending=True)
    # topk = int(fresh_ratio * score.shape[1])
    # fresh_indices = indices[:, :topk]
    # #stale_indices = indices[:, topk:]
    # # (B, fresh_ratio *N)

    # cluster_step = cache_dic['cluster_steps']
    # cluster_nums = cache_dic['cluster_nums']
    # # cluster_nums, k = cluster_scheduler(cache_dic, current) 
    # if layer == 0 and module == 'mlp':
    #     if cache_dic['group_info'] is None or step % cluster_step == 0:
    #         cluster_indices = get_group_indices(cache_dic['key_matrix'], cluster_nums, dims=2)
    #         cache_dic['group_info'] = cluster_indices
    # # visualize_cluster(cache_dic['group_info'], step)
    # # fresh_indices = get_cluster_max_indices(score, cache_dic['group_info'], cluster_nums)
    # # fresh_indices = get_cluster_topk_indices(score, cache_dic['group_info'], cluster_nums, k)
    # fresh_indices = get_max_cluster_indices(score, cache_dic['group_info'], cluster_nums)

    # # print(fresh_indices.shape)
    # # Updating the Cache Frequency Score s3 mentioned in the paper
    # # stale tokens index + 1, fresh tokens index = 0
    # cache_dic['cache_index'][-1][layer][module] += 1
    # cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, 
    #                                                                 src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    # # cache_dic['cache_index'][-1][layer][module] *= (1 - fresh_indices)

    # # select the fresh tokens out
    # fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    # if module in ['mlp', 'attn']:
    #     # cut out the fresh tokens
    #     fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)

    #     return fresh_indices, fresh_tokens
    
    # else:
    #     # no need for this branch hhh.
    #     raise ValueError("Unrecognized module?", module)
    ###############################################

    ### 测试聚类效果 ################################
    if layer == 0:
        cluster_indices = get_group_indices(cache_dic['key_matrix'], cache_dic['cluster_nums'], dims=2)
        cache_dic['group_info'] = cluster_indices

    fresh_indices, fresh_tokens = zero_cluster_tokens(tokens, cache_dic['group_info'], 102)
    # fresh_indices = torch.arange(tokens.shape[1], device=tokens.device).expand(tokens.shape[0], tokens.shape[1])
    # tokens[:, 110, :] = 0
    # fresh_tokens = tokens

    cache_dic['cache_index'][-1][layer][module] += 1
    cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    return fresh_indices, fresh_tokens
    ###############################################
    
def local_selection_with_bonus(score, bonus_ratio, grid_size=2):
    '''
    Uniform Spatial Distribution s4 mentioned in the paper
    '''
    batch_size, num_tokens = score.shape
    image_size = int(num_tokens ** 0.5)
    block_size = grid_size * grid_size
    
    assert num_tokens % block_size == 0, "The number of tokens must be divisible by the block size."
    
    # Step 1: Reshape score to group it by blocks
    score_reshaped = score.view(batch_size, image_size // grid_size, grid_size, image_size // grid_size, grid_size)
    score_reshaped = score_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    score_reshaped = score_reshaped.view(batch_size, -1, block_size)  # [batch_size, num_blocks, block_size]
    
    # Step 2: Find the max token in each block
    max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)  # [batch_size, num_blocks, 1]
    
    # Step 3: Create a mask to identify max score tokens
    mask = torch.zeros_like(score_reshaped)
    mask.scatter_(-1, max_indices, 1)  # Set mask to 1 at the max indices
    
    # Step 4: Apply the bonus only to the max score tokens
    score_reshaped = score_reshaped + (mask * max_scores * bonus_ratio)  # Apply bonus only to max tokens
    
    # Step 5: Reshape the score back to its original shape
    score_modified = score_reshaped.view(batch_size, image_size // grid_size, image_size // grid_size, grid_size, grid_size)
    score_modified = score_modified.permute(0, 1, 3, 2, 4).contiguous()
    score_modified = score_modified.view(batch_size, num_tokens)
    
    return score_modified

def zero_cluster_tokens(tokens, cluster_indices, x):
    B, N, dim = tokens.shape
    target_clusters = cluster_indices[:, x].unsqueeze(-1)
    mask = (cluster_indices == target_clusters)
    mask_expand = mask.unsqueeze(-1).expand(B, N, dim)
    tokens[mask_expand] = 0
    return torch.arange(N, device=tokens.device).expand(B, N), tokens

def consider_neighbor_score(score: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    '''
    param: score: (B, N)
    output: new_score: (B, N)
    '''
    # 使用roll操作来获取相邻的分数，避免cat操作
    next_score = torch.roll(score, shifts=-1, dims=-1)
    prev_score = torch.roll(score, shifts=1, dims=-1)

    # 计算是否大于相邻分数
    over_neighbor = (score > prev_score).int() + (score > next_score).int()

    # 使用mask来替代not_over_neighbor的计算
    mask = (over_neighbor == 2).float()

    # 计算neighbor_cover和score_cover
    neighbor_cover = ((next_score + prev_score) / 2 * (1 - gamma)) * mask
    score_cover = gamma * (1 - mask) + mask
    return score_cover * score + neighbor_cover

# def get_group_indices(key_matrix, cluster_nums, dims=2):
#     similarity_matrix = key_matrix @ key_matrix.transpose(-2, -1)
#     # print(key_matrix.shape, similarity_matrix.shape)
#     distance_matrix = batch_multi_dim_scale(similarity_matrix, num_dims=dims)
#     cluster_indices = kmeans_token_cluster(distance_matrix, cluster_nums)
#     return cluster_indices

# def get_group_score(score, cluster_indices, cluster_nums):
#     new_score = torch.zeros_like(score)
#     for i in range(cluster_nums):
#         mask = (cluster_indices == i)
#         new_score[mask] = score[mask].mean()
    
#     max_indices = new_score.argmax(dim=-1, keepdim=True)

#     return max_indices


# def kmeans_token_cluster(tokens, cluster_num):
#     '''
#     param: similarity matrix: [batch_size, num_tokens, num_tokens]
#     param: cluster_num: int, the number of clusters
#     output: cluster_indices: [batch_size, num_tokens]
#     '''
#     from sklearn.cluster import KMeans
#     import numpy as np
#     batch_size, num_tokens, _ = tokens.shape
#     cluster_indices = []
#     for i in range(batch_size):
#         kmeans = KMeans(n_clusters=cluster_num).fit(tokens[i].cpu().numpy())
#         cluster_indices.append(kmeans.labels_)
#     cluster_indices = np.array(cluster_indices)
#     return torch.tensor(cluster_indices, device=tokens.device)

# def batch_multi_dim_scale(similarity_matrix, num_dims=2, eps=1e-12):
#     # 1. 将相似度矩阵转换为距离矩阵
#     batch_size, n, _ = similarity_matrix.shape
#     similarity_matrix = torch.sigmoid(similarity_matrix)  # 使用 sigmoid 函数将相似度矩阵转换为 0-1 之间
#     distance_matrix = torch.sqrt(2 * (1 - similarity_matrix)) + eps  # 使用欧氏距离公式，并添加一个小数防止数值问题

#     # 2. 中心化矩阵
#     ones = torch.ones((n, n), device=similarity_matrix.device)
#     H = torch.eye(n, device=similarity_matrix.device) - ones / n  # [n, n]
#     H_batch = H.expand(batch_size, n, n)

#     # 3. 计算内积矩阵
#     D_squared = distance_matrix ** 2  # [batch_size, n, n]
#     B = -0.5 * torch.bmm(torch.bmm(H_batch, D_squared), H_batch)  # [batch_size, n, n]

#     # 4. 使用 SVD 分解
#     U, S_eigen, V = torch.svd(B)  # [batch_size, n, n], [batch_size, n], [batch_size, n, n]

#     # 5. 取前 num_dims 个特征值和对应的特征向量
#     S_selected = S_eigen[:, :num_dims].unsqueeze(-2)  # [batch_size, num_dims, 1]
#     U_selected = U[:, :, :num_dims]  # [batch_size, n, num_dims]

#     # print(S_selected.shape, U_selected.shape)

#     # 6. 计算降维后的坐标
#     coords = U_selected * torch.sqrt(S_selected)  # [batch_size, n, num_dims]
#     return coords

def get_group_indices(key_matrix, cluster_nums, dims=2):
    # 使用爱因斯坦求和优化矩阵乘法
    similarity_matrix = torch.einsum('bik,bjk->bij', key_matrix, key_matrix)
    
    # 优化后的多维度缩放
    distance_matrix = batch_multi_dim_scale(similarity_matrix, num_dims=dims)
    
    # 完全GPU化的K-Means实现
    cluster_indices = kmeans_token_cluster_gpu(distance_matrix, cluster_nums)
    return cluster_indices

def get_group_score(score, cluster_indices, cluster_nums):
    # 使用张量广播和索引替代循环
    one_hot = F.one_hot(cluster_indices, cluster_nums).float()  # [B, N, K]
    cluster_sums = torch.einsum('bnk,bn->bk', one_hot, score)
    cluster_counts = one_hot.sum(dim=1)
    new_score = cluster_sums / (cluster_counts + 1e-8)
    
    return torch.eq(cluster_indices, new_score.argmax(dim=-1, keepdim=True)).int()

def get_group_max_score(score, cluster_indices, cluster_nums):
    cluster_mask = F.one_hot(cluster_indices, cluster_nums).bool()
    score_expand = score.unsqueeze(-1).expand(-1, -1, cluster_nums)
    cluster_score = score_expand.masked_fill(~cluster_mask, float('-inf'))
    max_values, max_indices = cluster_score.max(dim=1, keepdim=True)

    valid_maxk = (max_values != float('inf'))
    max_indices = torch.where(valid_maxk, max_indices, torch.tensor(-1, device=score.device))
    return max_indices

def get_max_cluster_indices(score, cluster_indices, cluster_nums):
    '''
    找出每个样本中属于最大聚类的索引
    '''
    B, N = score.shape
    device = score.device

    # 1. 生成聚类掩码 [B, cluster_nums, N]
    cluster_mask = (cluster_indices.unsqueeze(1) == torch.arange(cluster_nums, device=device).view(1, -1, 1))

    # 2. 计算每个聚类的平均分数
    sum_scores = (score.unsqueeze(1) * cluster_mask).sum(dim=2)  # [B, cluster_nums]
    counts = cluster_mask.sum(dim=2).float()  # [B, cluster_nums]
    avg_scores = torch.where(counts > 0, sum_scores / counts, torch.tensor(-float('inf'), device=device))

    # 3. 找到每个样本中平均分最大的聚类索引 [B]
    max_cluster = avg_scores.argmax(dim=1)

    # 4. 生成每个样本的最大聚类掩码 [B, N]
    max_cluster_mask = (cluster_indices == max_cluster.unsqueeze(1))

    # 5. 生成原始索引矩阵 [B, N]
    indices = torch.arange(N, device=device).expand(B, N)

    # 6. 计算每个元素在结果中的位置（通过累加掩码）
    positions = torch.cumsum(max_cluster_mask.int(), dim=1)  # [B, N]
    positions = positions * max_cluster_mask  # 只保留选中位置

    # 7. 初始化结果并填充索引
    result = torch.zeros((B, N), dtype=torch.long, device=device)
    result.scatter_(dim=1, index=(positions - 1).clamp(min=0), src=indices)

    # 8. 计算每个样本的有效元素数量，并找到全局最大值
    valid_counts = max_cluster_mask.sum(dim=1)  # [B]
    max_elements = valid_counts.max().item()

    # 9. 截取结果到最大长度，不足部分自动填充0
    selected_indices = result[:, :max_elements]

    return selected_indices

def get_cluster_max_indices(score, cluster_indices, cluster_nums):
    B, N = score.shape
    device = score.device
    
    # 生成聚类索引的掩码 [B, K, N]
    k_indices = torch.arange(cluster_nums, device=device).view(1, cluster_nums, 1)
    mask = (cluster_indices.unsqueeze(1) == k_indices)
    
    # 将非当前聚类的分数设为负无穷
    score_masked = score.unsqueeze(1).masked_fill(~mask, -float('inf'))
    
    # 找到每个聚类的最大索引 [B, K]
    max_indices = score_masked.argmax(dim=-1)
    
    # 计算每个聚类的元素数量并检查空聚类
    cluster_count = mask.sum(dim=-1)
    empty_cluster_mask = (cluster_count == 0)
    
    # 将空聚类的索引填充为0（或任意有效索引）
    max_indices = torch.where(empty_cluster_mask, torch.tensor(0, device=device), max_indices)
    
    return max_indices

def get_cluster_topk_indices(score, cluster_indices, cluster_nums, K):
    B, N = score.shape
    device = score.device
    
    # 生成聚类索引的掩码 [B, K, N]
    k_indices = torch.arange(cluster_nums, device=device).view(1, cluster_nums, 1)
    mask = (cluster_indices.unsqueeze(1) == k_indices)
    
    # 将非当前聚类的分数设为负无穷
    score_masked = score.unsqueeze(1).masked_fill(~mask, -float('inf'))
    
    # 找到每个聚类的前K个最大索引 [B, K, topK]
    _, topk_indices = torch.topk(score_masked, K, dim=-1)
    
    # 计算每个聚类的元素数量并检查是否小于K
    cluster_count = mask.sum(dim=-1)
    # insufficient_mask = (cluster_count < K)
    
    # # 对于元素数量小于K的聚类，用0（或任意有效索引）填充
    # if insufficient_mask.any():
    #     # 创建一个填充索引的张量 [B, K, K]
    #     fill_indices = torch.zeros((B, cluster_nums, K), dtype=torch.long, device=device)
        
    #     # 将填充索引应用到不足的聚类
    #     topk_indices = torch.where(insufficient_mask.unsqueeze(-1), fill_indices, topk_indices)
    # 处理空聚类和不足K的情况
    valid_mask = (torch.arange(K, device=device).view(1, 1, -1) < cluster_count.unsqueeze(-1))
    topk_indices = torch.where(
        valid_mask | (cluster_count == 0).unsqueeze(-1),
        topk_indices,
        torch.zeros_like(topk_indices)
        # torch.randint(0, N, (B, cluster_nums, K), device=device)
    )
    
    return topk_indices.view(B, -1)

def kmeans_token_cluster_gpu(tokens, cluster_num, max_iters=100):
    '''
    GPU加速的K-Means实现
    '''
    B, N, _ = tokens.shape
    device = tokens.device
    centroids = tokens[torch.arange(B, device=device)[:, None], torch.randint(0, N, (B, cluster_num), device=device)]
    
    for _ in range(max_iters):
        dists = torch.cdist(tokens, centroids, p=2) # [B, N, K]
        labels = dists.argmin(dim=-1)

        new_centroids = torch.zeros_like(centroids)
        for k in range(cluster_num):
            mask = (labels == k).unsqueeze(-1)
            cluster_points = tokens * mask
            new_centroids[:, k] = cluster_points.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        if torch.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids
    
    return labels

def batch_multi_dim_scale(similarity_matrix, num_dims=2, eps=1e-12):
    # 优化后的矩阵运算流程
    B, N, _ = similarity_matrix.shape
    device = similarity_matrix.device
    
    # 使用更高效的距离计算
    distance_matrix = 2 * (1 - similarity_matrix.sigmoid()).sqrt_().add_(eps)
    
    # 中心化矩阵（利用广播机制）
    H = torch.eye(N, device=device) - 1/N
    B_matrix = -0.5 * H @ (distance_matrix**2) @ H
    
    # 使用截断SVD加速计算
    U, S, _ = torch.svd_lowrank(B_matrix, q=num_dims+2, niter=3)
    coords = U[:, :, :num_dims] * torch.sqrt(S[:, :num_dims]).unsqueeze(1)
    
    return coords

def visualize_cluster(cluster_indices, step):
    import matplotlib.pyplot as plt
    import os
    h = w = int(cluster_indices.shape[1] ** 0.5)
    labels = cluster_indices[0].reshape(h, w).cpu().numpy()
    plt.imshow(labels, cmap='tab20')
    dir = 'samples_cluster'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, f'cluster_{step}.png'))
