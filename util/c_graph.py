import torch
import numpy as np
import torch.nn.functional as F



def build_topk_normalized_adj(context, k=10, symmetric=False):
        N = context.size(0)

        # 1. 计算相似度
        context_norm = context / torch.norm(context, p=2, dim=-1, keepdim=True)
        sim = torch.mm(context_norm, context_norm.T)  # N×N

        # 2. 去掉对角元素
        sim.fill_diagonal_(-float('inf'))

        # 3. 获取 top-k 索引
        top_val, topk_indices = torch.topk(sim, k=k, dim=-1)  # (N, k)

        # 4. 构造邻接矩阵，值为1
        row_idx = torch.arange(N, device=context.device).unsqueeze(1).expand(-1, k).reshape(-1)
        col_idx = topk_indices.reshape(-1)
        values = torch.ones_like(row_idx, dtype=torch.float32)
        #values = top_val.flatten()

        # 如果对称化则加上反向边
        if symmetric:
            row_idx = torch.cat([row_idx, col_idx], dim=0)
            col_idx = torch.cat([col_idx, row_idx[:len(col_idx)]], dim=0)
            values = torch.ones_like(row_idx, dtype=torch.float32)

        indices = torch.stack([row_idx, col_idx], dim=0)
        adj = torch.sparse.FloatTensor(indices, values, torch.Size([N, N]))

        # 5. 邻接归一化：D^(-1/2) A D^(-1/2)
        deg = torch.sparse.sum(adj, dim=1).to_dense()  # (N,)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        d_row = deg_inv_sqrt[row_idx]
        d_col = deg_inv_sqrt[col_idx]
        norm_values = d_row * values * d_col

        adj_norm = torch.sparse.FloatTensor(indices, norm_values, torch.Size([N, N]))

        return  adj_norm

def norm_sparse_adj(sparse_adj: torch.sparse.FloatTensor):
    """
    对稀疏邻接矩阵 A 做归一化： D^{-1/2} A D^{-1/2}
    Args:
        sparse_adj: torch.sparse.FloatTensor, (N, N) 稀疏邻接矩阵
    Returns:
        adj_norm: 归一化后的稀疏邻接矩阵 (torch.sparse.FloatTensor)
    """
    # 保证 coalesce（合并重复索引）
    sparse_adj = sparse_adj.coalesce()
    indices = sparse_adj.indices()   # (2, E) → 行和列索引
    values = sparse_adj.values()     # (E,)   → 边的权重
    N = sparse_adj.size(0)

    # 计算度数 D
    deg = torch.sparse.sum(sparse_adj, dim=1).to_dense()  # (N,)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0         # 避免除零

    # 对每条边 (i, j)，权重变为 D^{-1/2}[i] * A[i,j] * D^{-1/2}[j]
    row, col = indices
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]

    # 构造新的稀疏张量
    adj_norm = torch.sparse_coo_tensor(indices, norm_values, (N, N))
    return adj_norm.coalesce()

def sparse_topk_by_row(sparse_tensor, k=10):
    """
    对稀疏张量的每一行，仅保留前 k 个值（按值大小排序），保持稀疏性。
    Args:
        sparse_tensor: torch.sparse.FloatTensor (COO 格式)
        k: 每行保留的最大非零个数
    Return:
        稀疏张量（已过滤）
    """
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    rows, cols = indices

    keep_rows, keep_cols, keep_vals = [], [], []
    for r in rows.unique():
        mask = (rows == r)
        r_cols = cols[mask]
        r_vals = values[mask]

        # 按值排序，取前 k 个
        topk = torch.topk(r_vals, min(k, r_vals.numel()))
        keep_rows.append(torch.full_like(topk.indices, r))
        keep_cols.append(r_cols[topk.indices])
        keep_vals.append(topk.values)

    new_rows = torch.cat(keep_rows)
    new_cols = torch.cat(keep_cols)
    new_vals = torch.cat(keep_vals)

    new_indices = torch.stack([new_rows, new_cols], dim=0)
    return torch.sparse_coo_tensor(new_indices, new_vals, sparse_tensor.size())
