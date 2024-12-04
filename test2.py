import numpy as np


def random_sum_se(L, K, M, N, noise_power, P_max):
    """
    随机生成信道参数并计算未优化的 Sum-SE。

    参数:
    - L: 接入点 (AP) 数量
    - K: 用户数量
    - M: 每个 AP 的天线数量
    - N: RIS 元件数量
    - noise_power: 噪声功率
    - P_max: 最大总功率

    返回:
    - sum_se: 随机信道下的总谱效率
    """
    # 随机生成信道
    H_direct = np.random.randn(L, K, M) + 1j * np.random.randn(L, K, M)  # AP 到用户直接信道
    H_ris = np.random.randn(K, N) + 1j * np.random.randn(K, N)  # RIS 到用户信道
    G_ris = np.random.randn(L, N, M) + 1j * np.random.randn(L, N, M)  # AP 到 RIS 信道

    # 随机生成 RIS 相移矩阵
    Theta = np.diag(np.exp(1j * 2 * np.pi * np.random.rand(N)))

    # 计算有效信道
    H_eff = np.zeros((L, K, M))
    for l in range(L):
        H_eff[l, :, :] = H_direct[l, :, :] + H_ris @ Theta @ G_ris[l, :, :]

    W = np.zeros((L, K, M))  # W 的维度为 (M x K)

    for l in range(L):
        for k in range(K):  # 对每个用户
            W[l, k, :] = H_eff[l, k, :].conj() / np.linalg.norm(H_eff[l, k, :])  # MRT 向量

    W /= np.linalg.norm(W, axis=0)
    W *= np.sqrt(P_max / K)

    # 计算每个用户的 SINR 和 SE
    sum_se = 0
    for k in range(K):
        signal_power = 0
        interference_power = 0
        for l in range(L):
            h_k = H_eff[l, k, :]
            signal_power = np.abs(np.dot(h_k, W[l, k, :])) ** 2
            interference_power = np.sum([np.abs(np.dot(h_k, W[l, j, :])) ** 2 for j in range(K) if j != k])
        sinr_k = signal_power / (interference_power + noise_power)
        print(f"{k}用户信噪比:{sinr_k}")
        sum_se += np.log2(1 + sinr_k)

    return sum_se


# 参数设置
L, K, M, N = 8, 4, 8, 16
noise_power = 1e-9  # 噪声功率
P_max = 10.0  # 最大功率

# 计算随机信道下的 Sum-SE
sum_se = random_sum_se(L, K, M, N, noise_power, P_max)
print(f"随机信道产生的 Sum-SE: {sum_se:.2f} bps/Hz")
