import numpy as np
import random


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def grid_random_sampling(k, data):
    """
    Conduct uniform sampling
    """
    if data == []:
        return []
    # build a sqaure table
    m = isqrt(k)

    if len(data) < m**2:
        return list(range(len(data)))

    sampled_data = []
    min_acc, max_acc = min([item[0] for item in data]), max([item[0] for item in data])
    min_flops, max_flops = min([item[1] for item in data]), max([item[1] for item in data])
    acc_range = np.linspace(min_acc, max_acc, m+1)
    flops_range = np.linspace(min_flops, max_flops, m+1)
    for i in range(1, len(acc_range)):
        for j in range(1, len(flops_range)):
            restricted_data = [e for e in range(len(data))
                               if acc_range[i-1] < data[e][0] < acc_range[i] and
                               flops_range[j-1] < data[e][1] < flops_range[j]]
            if len(restricted_data) > 0:
                sampled_data.append(random.choices(restricted_data, k=1)[0])

    sampled_data += random.choices(range(len(data)), k=min(k - len(sampled_data), len(data)))

    return sampled_data


def rank_sort(k, data, l=1):
    if data == []:
        return []
    accs = [-entry[0] for entry in data]
    flops = [entry[1] for entry in data]

    acc_idx = np.argsort(accs)
    flops_idx = np.argsort(flops)

    def idx_to_rank(idx_list):
        ranks = [0] * len(idx_list)
        for i, v in enumerate(idx_list):
            ranks[v] = i
        return ranks
    acc_ranks, flops_ranks = idx_to_rank(acc_idx), idx_to_rank(flops_idx)
    flops_ranks = [f * l for f in flops_ranks]
    rank_sum = [acc_ranks[i] + flops_ranks[i] for i in range(len(acc_ranks))]
    rank_sum_argsort = np.argsort(rank_sum)

    return rank_sum_argsort[:k]
