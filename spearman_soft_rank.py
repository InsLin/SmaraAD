import torch

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def soft_rank(x, beta):
    soft_ranks = []
    for xi in x:
        rank = torch.sum(sigmoid(beta * (x - xi)))
        soft_ranks.append(rank)
    return torch.tensor(soft_ranks)

def spearman_soft_rank(x, y, beta):
    soft_ranks_x = soft_rank(x, beta)
    soft_ranks_y = soft_rank(y, beta)
    rank_diff = soft_ranks_x - soft_ranks_y
    n = len(x)
    spearman_rho = 1 - (6 * torch.sum(rank_diff ** 2)) / (n * (n**2 - 1))
    return spearman_rho
