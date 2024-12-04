import numpy as np

from .env import Env_Generate


def dB2power(dB):
    power = 10 ** (np.float64(dB) / 10)
    return power


class CF_MIMO(object):
    def __init__(
        self,
        num_APs,
        num_AP_antennas,
        num_RIS_elements,
        num_users,
        area_size,
        AWGN_var,
        power_limit,
        num_UE_antennas,
        num_RIS,
        channel_est_error=False,
        channel_noise_var=1e-11,
    ):
        self.L = num_APs
        self.M = num_AP_antennas
        self.R = num_RIS  # the number of RISs
        self.N = num_RIS_elements
        self.K = num_users
        self.U = num_UE_antennas

        self.Env = Env_Generate(
            num_APs, num_AP_antennas, num_RIS_elements, num_users, area_size, num_UE_antennas, num_RIS
        )

        self.awgn_var = AWGN_var

        self.power_limit = dB2power(power_limit)

        self.channel_est_error = channel_est_error
        self.channel_noise_var = channel_noise_var

        self.Env._position_generate_()
        self.Env._channel_generate_()

        self.H = self.Env.H
        self.F = self.Env.F
        self.G = self.Env.G
        self.W = self.Env.W
        self.Phi = self.Env.Phi

        self.G_l = np.zeros((self.L, self.R * self.N, self.M), dtype=complex)
        self.F_l = np.zeros((self.K, self.R * self.N, self.U), dtype=complex)
        self.h_hat_l = np.zeros((self.L, self.K, self.M, self.U), dtype=complex)
        self.h_hat = np.zeros((self.K, self.L * self.M, self.U), dtype=complex)

        self.w_k = np.zeros((self.K, self.L * self.M), dtype=complex)

        power_size = 2 * self.K

        channel_size = 2 * (self.N * self.M * self.R + self.N * self.K * self.R * self.U + self.M * self.K * self.U)

        self.action_dim = 2 * self.M * self.K + 2 * self.R * self.N
        self.state_dim = channel_size + self.action_dim

        self.state = np.zeros((self.L, self.state_dim))

        self.done = None

        self.episode_t = None

        self.EE_res = 0.0

        self._compute_h_hat_()
        self._W_generate()
        self._compute_w_k_()

    def _compute_h_hat_(self):
        for k in range(self.K):
            for r in range(self.R):
                self.F_l[k, r * self.N : (r + 1) * self.N, :] = self.F[r, k, :, :]
        for l in range(self.L):
            for r in range(self.R):
                self.G_l[l, r * self.N : (r + 1) * self.N, :] = self.G[l, r, :, :]
        for l in range(self.L):
            for k in range(self.K):
                tmp0 = self.H[l, k, :, :].reshape(self.M, self.U)
                tmp1 = self.G_l[l, :, :].reshape(self.R * self.N, self.M)
                tmp2 = self.F_l[k, :, :].reshape(self.R * self.N, self.U)
                tmp3 = tmp1.conj().T @ self.Phi @ tmp2
                self.h_hat_l[l, k, :, :] = tmp0 + tmp3

        for k in range(self.K):
            for l in range(self.L):
                self.h_hat[k, l * self.M : (l + 1) * self.M, :] = self.h_hat_l[l, k, :, :]

    def _compute_w_k_(self):
        for k in range(self.K):
            for l in range(self.L):
                self.w_k[k, l * self.M : (l + 1) * self.M] = self.W[l, k, :]

    def reset(self):
        self.episode_t = 0
        self.Env._position_generate_()
        self.Env._channel_generate_()
        self._compute_h_hat_()
        self._W_generate()
        self._compute_w_k_()

        for l in range(self.L):
            init_action_W = np.hstack((np.real(self.W[l].reshape(1, -1)), np.imag(self.W[l].reshape(1, -1))))

            init_action_Phi = np.hstack(
                (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1))
            )

            # print(init_action_W.shape)
            # print(init_action_Phi.shape)
            init_action = np.hstack((init_action_W, init_action_Phi))

            W_real = init_action[:, : (self.M) * self.K]
            # print(W_real.shape)
            W_imag = init_action[:, self.M * self.K : 2 * (self.M * self.K)]

            Phi_real = init_action[:, -2 * self.R * self.N : -self.R * self.N]
            Phi_imag = init_action[:, -self.R * self.N :]

            self.Phi = np.eye(self.R * self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)
            self.W[l] = W_real.reshape(self.K, self.M) + 1j * W_imag.reshape(self.K, self.M)
            # h_hat_l = self.h_hat_l[l]

            # power_real = np.real(np.diag(self.W[l] @ self.W[l].conjugate().T)).reshape(1, -1)
            #
            # power_limit = dB2power(self.power_limit) * np.ones((1, self.K)).reshape(1, -1)

            H_real, H_imag = np.real(self.H[l]).reshape(1, -1), np.imag(self.H[l]).reshape(1, -1)
            F_real, F_imag = np.real(self.F_l).reshape(1, -1), np.imag(self.F_l).reshape(1, -1)
            G_real, G_imag = np.real(self.G_l[l]).reshape(1, -1), np.imag(self.G_l[l]).reshape(1, -1)
            # H_hat_real,H_hat_imag = np.real(self.h_hat_l[l]).reshape(1,-1),np.imag(self.h_hat_l[l]).reshape(1,-1)
            # print(H_imag.shape)
            # print(F_imag.shape)
            # print(G_imag.shape)
            # print(power_real.shape)
            # print(power_limit.shape)
            self.state[l] = np.hstack((init_action, H_real, H_imag, F_real, F_imag, G_real, G_imag))
            # self.state[l] = np.hstack(
            #     (init_action, power_real, power_limit, H_hat_real,H_hat_imag))
            # print("1")

        return self.state

    def _compute_reward_(self):
        rewards = []
        a_EE, s_EE, EE_res = self._compute_EE_()

        # 根据EE返回reward
        reward = s_EE

        # 检查功率是否超标
        for i in range(self.L):
            if np.linalg.norm(self.w_k[:, i * self.M : (i + 1) * self.M]) ** 2 >= self.power_limit:
                reward -= 10
                break

        rewards = [reward for _ in range(self.L)]

        return rewards

    def step(self, action):
        Phi_real_tmp = []
        Phi_imag_tmp = []
        for l in range((int)(self.L)):
            H_real, H_imag = np.real(self.H[l]).reshape(1, -1), np.imag(self.H[l]).reshape(1, -1)
            F_real, F_imag = np.real(self.F_l).reshape(1, -1), np.imag(self.F_l).reshape(1, -1)
            G_real, G_imag = np.real(self.G_l[l]).reshape(1, -1), np.imag(self.G_l[l]).reshape(1, -1)

            t_action = action[l].reshape(1, -1)
            # 更新每个agent的状态(将新的动作空间替换原来的动作空间)
            self.state[l] = np.hstack((t_action, H_real, H_imag, F_real, F_imag, G_real, G_imag))

            # 更新W 预编码矩阵和Phi 相移矩阵
            W_real = action[l, : (self.M) * self.K]
            W_imag = action[l, self.M * self.K : 2 * (self.M * self.K)]

            Phi_real_tmp.append(action[l, -2 * self.R * self.N : -self.R * self.N].reshape(self.R * self.N))
            Phi_imag_tmp.append(action[l, -self.R * self.N :].reshape(self.R * self.N))

            self.W[l] = W_real.reshape(self.K, self.M) + 1j * W_imag.reshape(self.K, self.M)

        Phi_real = np.mean(np.array(Phi_real_tmp), axis=0)
        Phi_imag = np.mean(np.array(Phi_imag_tmp), axis=0)

        self.Phi = np.eye(self.R * self.N, dtype=complex) * (Phi_real + 1j * Phi_imag)
        self._compute_h_hat_()
        self._compute_w_k_()
        reward = self._compute_reward_()

        return self.state, reward

    def _compute_EE_(self):
        EE_res = np.zeros(self.K, dtype=float)
        for k in range((int)(self.K)):
            signal_power = 0.0
            interference_power = (float)(self.awgn_var**2)
            for l in range((int)(self.L)):
                tmp1 = self.h_hat_l[l, k, :, :].T
                for j in range((int)(self.K)):
                    tmp2 = self.W[l, j, :]
                    if j != k:
                        interference_power += np.sum(np.abs(tmp1 @ tmp2) ** 2)
                    else:
                        signal_power += np.sum(np.abs(tmp1 @ tmp2) ** 2)

            sinr_k = np.divide(signal_power, interference_power)
            print(sinr_k)
            EE_res[k] = np.log2(1 + sinr_k)

        s_EE = 0.0
        a_EE = 0.0
        for k in range((int)(self.K)):
            s_EE += EE_res[k]
        a_EE = np.divide(s_EE, self.K)
        print(f"随机信道产生的 Sum-SE: {s_EE:.2f} bps/Hz")
        return a_EE, s_EE, EE_res

    def close(self):
        pass

    def _W_generate(self):
        for l in range(self.L):
            for k in range(self.K):
                self.W[l, k, :] = np.sum(self.h_hat_l[l, k, :, :].conj(), axis=1)
                # self.W[l, k, :] /= np.linalg.norm(self.h_hat_l[l, k, :, :])
                gamma = np.sqrt(np.divide(1.0, np.trace(self.h_hat_l[l, k, :, :] @ self.h_hat_l[l, k, :, :].conj().T)))
                self.W[l, k, :] *= gamma
        # self.W /= np.linalg.norm(self.W, axis=0)
        self.W *= np.sqrt(self.power_limit / self.K)
        self.Env.W = self.W

    def _W_generate_ZF(self):

        for l in range(self.L):
            H_list = []
            for k in range(self.K):
                H_list.append(self.h_hat_l[l, k, :, :].T)
            H_total = np.vstack(H_list)

            # W_zf (M*(U*K))
            W_zf = np.linalg.pinv(H_total)

            for k in range(self.K):
                self.W[l, k, :] = np.mean(W_zf[:, k * self.U : (k + 1) * self.U], axis=1)
                H = H_list[k]
                gamma = np.divide(1.0, np.linalg.norm(np.linalg.inv(H @ H.conj().T)))
                self.W[l, k, :] *= gamma

        self.W *= np.sqrt(self.power_limit / self.K)
        self.Env.W = self.W
