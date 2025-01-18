import numpy as np


class Env_Generate(object):
    def __init__(
        self, num_APs, num_AP_antennas, num_STARRIS_elements, num_users, area_size, num_UE_antennas, num_STARRIS
    ):
        self.L = num_APs
        self.M = num_AP_antennas
        self.R = num_STARRIS  # the number of STARRISs
        self.N = num_STARRIS_elements
        self.K = num_users
        self.U = num_UE_antennas

        self.area_size = area_size
        self.AP_position = None
        self.STARRIS_position = None
        self.user_position = None
        self.AP_height = 15
        self.STARRIS_height = 5
        self.user_height = 2
        self.disAP2STARRIS = None
        self.disAP2user = None
        self.disSTARRIS2user = None

        self.H = np.zeros((self.L, self.K, self.M, self.U), dtype=complex)
        self.F = np.zeros((self.R, self.K, self.N, self.U), dtype=complex)
        self.G = np.zeros((self.L, self.R, self.N, self.M), dtype=complex)
        # the dim of W is L,M,K
        self.W = np.zeros((self.L, self.K, self.M), dtype=complex) + 1j * np.zeros(
            (self.L, self.K, self.M), dtype=complex
        )

        # P矩阵用于记录基站,STARRIS,用户的相对位置关系
        self.P = np.zeros((self.L, self.R, self.K), dtype=int)
        # print(self.W.shape)
        # the dim of phi is R*N,R*N
        # self.Phi = np.eye(self.R * self.N, dtype=complex) + 1j * np.eye(self.R * self.N, dtype=complex)
        self.Phi_R = np.diag(np.exp(1j * 2 * np.pi * np.random.rand(self.R * self.N)))
        self.Phi_T = np.diag(np.exp(1j * 2 * np.pi * np.random.rand(self.R * self.N)))

    def _position_generate_(self):
        # RIS is at the center of the area, the area is a square, with the length of area_size
        # self.RIS_position = np.array([self.area_size / 2, self.area_size / 2])
        self.STARRIS_position = np.random.uniform(0, self.area_size, (self.R, 2))
        # AP seperate the area into L same parts, and the num of lines and rows should be m and n, where m is the
        # nearest int of sqrt(L) and n = L/m
        m = int(np.sqrt(self.L))
        n = self.L // m
        # APs are the centers of the L parts
        self.AP_position = np.zeros((self.L, 2))
        for i in range(m):
            for j in range(n):
                self.AP_position[i * n + j, :] = np.array(
                    [(i + 0.5) * self.area_size / m, (j + 0.5) * self.area_size / n]
                )

        # users are randomly distributed in the area
        self.user_position = np.random.uniform(0, self.area_size, (self.K, 2))
        # self.user_position = np.array([[25.5, 59.5], [70.5, 20.5], [40, 30.5], [80.5, 87],[10, 11],[95, 65]])

        self.disAP2user = np.zeros((self.L, self.K), dtype=complex)
        self.disSTARRIS2user = np.zeros((self.R, self.K), dtype=complex)
        self.disAP2STARRIS = np.zeros((self.L, self.R), dtype=complex)

        for i in range(self.L):
            for j in range(self.K):
                self.disAP2user[i, j] = np.sqrt(
                    np.sum((self.AP_position[i, :] - self.user_position[j, :]) ** 2)
                    + (self.AP_height - self.user_height) ** 2
                )

        for r in range(self.R):
            for k in range(self.K):
                self.disSTARRIS2user[r, k] = np.sqrt(
                    np.sum((self.STARRIS_position[r, :] - self.user_position[k, :]) ** 2)
                    + (self.STARRIS_height - self.user_height) ** 2
                )

        for i in range(self.L):
            for r in range(self.R):
                self.disAP2STARRIS[i, r] = np.sqrt(
                    np.sum((self.STARRIS_position[r, :] - self.AP_position[i, :]) ** 2)
                    + (self.STARRIS_height - self.AP_height) ** 2
                )

        self._P_generate()

    def _channel_H_generate(self, dis):
        tmp_H = np.random.rayleigh(1, (self.M, self.U)) + 1j * np.zeros((self.M, self.U), dtype=complex)
        for m in range(self.M):
            for u in range(self.U):
                tmp_H[m, u] = tmp_H[m, u] * np.exp(1j * 2 * np.pi * np.random.rand(), dtype=complex)
        disAP2UE = np.sqrt(1e-3 * (dis ** (-3.5)))
        channel_H = disAP2UE * tmp_H
        return channel_H

    def _channel_F_generate(self, dis):
        tmp_F = np.random.rayleigh(1, (self.N, self.U)) + 1j * np.zeros((self.N, self.U), dtype=complex)
        for n in range(self.N):
            for u in range(self.U):
                tmp_F[n, u] = tmp_F[n, u] * np.exp(1j * 2 * np.pi * np.random.rand(), dtype=complex)

        disSTARRIS2UE = np.sqrt(1e-3 * (dis ** (-2.8)))
        channel_F = disSTARRIS2UE * tmp_F
        return channel_F

    def _channel_G_generate(self, dis):
        tmp_G = np.random.rayleigh(1, (self.N, self.M)) + 1j * np.zeros((self.N, self.M), dtype=complex)
        for n in range(self.N):
            for m in range(self.M):
                tmp_G[n, m] = tmp_G[n, m] * np.exp(1j * 2 * np.pi * np.random.rand(), dtype=complex)
        disAP2STARRIS = np.sqrt(2e-3 * (dis ** (-2.2)))
        channel_G = disAP2STARRIS * tmp_G
        return channel_G

    def _channel_generate_(self):
        # the dimension of H is (L*M, K),obey rayleigh distribution,use np.random.rayleigh
        # beta_H = np.zeros((self.L, self.K))

        # print(beta_H.shape)

        for l in range(self.L):
            for k in range(self.K):
                self.H[l, k, :, :] = self._channel_H_generate(self.disAP2user[l, k])

        for r in range(self.R):
            for k in range(self.K):
                self.F[r, k, :, :] = self._channel_F_generate(self.disSTARRIS2user[r, k])

        # self.F = beta_F * np.random.rayleigh(1, (self.N, self.K)) * np.exp(1j * 2 * np.pi * np.random.rand())
        # print(self.F.shape)
        # the dimension of G is (N,L*M)
        for l in range(self.L):
            for r in range(self.R):
                self.G[l, r, :, :] = self._channel_G_generate(self.disAP2STARRIS[l, r])
        # self.G = beta_G * np.ones((self.N, self.M), dtype=complex)

    # 该函数用于确定基站,STARRIS,用户的相对位置关系
    def _P_generate(self):
        # 如果基站和用户分别在RIS两侧(横坐标),说明这三者应该采用T相移矩阵，将P置为1
        # 如果基站和用户在RIS的同一侧,说明这三者应该采用R相移矩阵，将P置为0(初始化的值,所以不用处理)
        for l in range(self.L):
            for r in range(self.R):
                for k in range(self.K):
                    if (
                        self.AP_position[l][0] <= self.STARRIS_position[r][0]
                        and self.user_position[k][0] > self.STARRIS_position[r][0]
                    ):
                        self.P[l][r][k] = 1
                    elif (
                        self.AP_position[l][0] > self.STARRIS_position[r][0]
                        and self.user_position[k][0] <= self.STARRIS_position[r][0]
                    ):
                        self.P[l][r][k] = 1
