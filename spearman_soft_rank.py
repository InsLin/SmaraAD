import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def soft_rank(x, beta):
    soft_ranks = []
    for xi in x:
        rank = np.sum(sigmoid(beta * (x - xi)))
        soft_ranks.append(rank)
    return soft_ranks

def spearman_soft_rank(x, y, beta):
    soft_ranks_x = np.array(soft_rank(x, beta))
    soft_ranks_y = np.array(soft_rank(y, beta))
    rank_diff = soft_ranks_x - soft_ranks_y
    n = len(x)
    spearman_rho = 1 - (6 * np.sum(rank_diff ** 2)) / (n * (n**2 - 1))
    return spearman_rho
