import numpy as np

from .env_STARRIS import Env_Generate


def dB2power(dB):
    power = 10 ** (np.float64(dB) / 10)
    return power


class CF_MIMO(object):
    def __init__(
        self,
        num_APs,
        num_AP_antennas,
        num_STARRIS_elements,
        num_users,
        area_size,
        AWGN_var,
        power_limit,
        num_UE_antennas,
        num_STARRIS,
        channel_est_error=False,
        channel_noise_var=1e-11,
    ):
        self.L = num_APs
        self.M = num_AP_antennas
        self.R = num_STARRIS  # the number of STARRISs
        self.N = num_STARRIS_elements
        self.K = num_users
        self.U = num_UE_antennas

        self.Env = Env_Generate(
            num_APs, num_AP_antennas, num_STARRIS_elements, num_users, area_size, num_UE_antennas, num_STARRIS
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
        self.Phi_R = self.Env.Phi_R
        self.Phi_T = self.Env.Phi_T
        self.P = self.Env.P

        self.G_l = np.zeros((self.L, self.R * self.N, self.M), dtype=complex)
        self.F_l = np.zeros((self.K, self.R * self.N, self.U), dtype=complex)
        self.h_hat_l = np.zeros((self.L, self.K, self.M, self.U), dtype=complex)
        self.h_hat = np.zeros((self.K, self.L * self.M, self.U), dtype=complex)

        self.w_k = np.zeros((self.K, self.L * self.M), dtype=complex)

        power_size = 2 * self.K

        channel_size = 2 * (self.N * self.M * self.R + self.N * self.K * self.R * self.U + self.M * self.K * self.U)

        # 由于每个STAR-RIS的相位转移矩阵由传输和反射,并且2个矩阵各有实部和虚部,故乘4
        self.action_dim = 2 * self.M * self.K + 4 * self.R * self.N
        self.state_dim = channel_size + self.action_dim

        self.state = np.zeros((self.L, self.state_dim))

        self.done = None

        self.episode_t = None

        self.EE_res = 0.0

        self.LastRewards = [0 for _ in range(self.L)]

        #  STAR-RIS采用ES策略,并且将传输和反射的gamma因子 T/R 设为该变量
        self.STARRIS_RATE = 1
        self.STARRIS_T = None
        self.STARRIS_R = None
        self._init_STARRIS_arg()

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
                tmp = np.zeros((self.U, self.M), dtype=complex)
                tmp0 = self.H[l, k, :, :]

                # tmp1 = self.G_l[l, :, :].reshape(self.R * self.N, self.M)
                # tmp2 = self.F_l[k, :, :].reshape(self.R * self.N, self.U)
                # tmp3 = tmp1.conj().T @ self.Phi @ tmp2
                # self.h_hat_l[l, k, :, :] = tmp0 + tmp3

                # tmp4 = tmp0 + tmp3

                for r in range(self.R):
                    tmp1 = self.F[r, k, :, :]
                    tmp2 = self.G[l, r, :, :]
                    tmpPhi = None
                    if self.P[l][r][k] == 1:
                        tmpPhi = (
                            np.sqrt(self.STARRIS_T)
                            * self.Phi_T[
                                r * self.N : (r + 1) * self.N,
                                r * self.N : (r + 1) * self.N,
                            ]
                        )
                    else:
                        tmpPhi = (
                            np.sqrt(self.STARRIS_R)
                            * self.Phi_R[
                                r * self.N : (r + 1) * self.N,
                                r * self.N : (r + 1) * self.N,
                            ]
                        )
                    tmp += tmp1.conj().T @ tmpPhi.conj().T @ tmp2
                self.h_hat_l[l, k, :, :] = (tmp0.conj().T + tmp).conj().T
                # print((tmp0.conj().T + tmp).conj().T - tmp4)

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
        self._W_generate_ave()
        self._compute_w_k_()

        for l in range(self.L):
            init_action_W = np.hstack((np.real(self.W[l].reshape(1, -1)), np.imag(self.W[l].reshape(1, -1))))

            init_action_Phi_T = np.hstack(
                (np.real(np.diag(self.Phi_T)).reshape(1, -1), np.imag(np.diag(self.Phi_T)).reshape(1, -1))
            )

            init_action_Phi_R = np.hstack(
                (np.real(np.diag(self.Phi_R)).reshape(1, -1), np.imag(np.diag(self.Phi_R)).reshape(1, -1))
            )

            # print(init_action_W.shape)
            # print(init_action_Phi.shape)
            init_action = np.hstack((init_action_W, init_action_Phi_T, init_action_Phi_R))

            W_real = init_action[:, : (self.M) * self.K]
            # print(W_real.shape)
            W_imag = init_action[:, self.M * self.K : 2 * (self.M * self.K)]

            Phi_real_T = init_action[:, -4 * self.R * self.N : -3 * self.R * self.N]
            Phi_imag_T = init_action[:, -3 * self.R * self.N : -2 * self.R * self.N]

            Phi_real_R = init_action[:, -2 * self.R * self.N : -self.R * self.N]
            Phi_imag_R = init_action[:, -self.R * self.N :]

            self.Phi_T = np.eye(self.R * self.N, dtype=complex) * (Phi_real_T + 1j * Phi_imag_T)
            self.Phi_R = np.eye(self.R * self.N, dtype=complex) * (Phi_real_R + 1j * Phi_imag_R)
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

        # 检查功率是否超标
        # for i in range(self.L):
        #     if np.linalg.norm(self.w_k[:, i * self.M : (i + 1) * self.M]) ** 2 > self.power_limit:
        #         reward -= 10
        #         break
        for l in range(self.L):
            reward = 0
            for k in range((int)(self.K)):
                signal_power = 0.0
                interference_power = (float)(self.awgn_var**2)
                for j in range((int)(self.K)):
                    if j == k:
                        signal_vector = np.zeros((self.U), dtype=complex)
                        tmp1 = self.h_hat_l[l, k, :, :].T
                        tmp2 = self.W[l, j, :]
                        signal_vector += tmp1 @ tmp2
                        signal_power += np.linalg.norm(signal_vector) ** 2
                    else:
                        interference_vector = np.zeros((self.U), dtype=complex)
                        for i in range((int)(self.L)):
                            tmp1 = self.h_hat_l[i, k, :, :].T
                            tmp2 = self.W[i, j, :]
                            interference_vector += tmp1 @ tmp2
                        interference_power += np.linalg.norm(interference_vector) ** 2
                sinr_k = np.divide(signal_power, interference_power)
                reward += np.log2(1+sinr_k)
                # print(sinr_k)
            rewards.append(reward)
        # print(rewards)
        return rewards

    def step(self, action):
        Phi_real_T_tmp = []
        Phi_imag_T_tmp = []
        Phi_real_R_tmp = []
        Phi_imag_R_tmp = []
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

            Phi_real_T_tmp.append(action[l, -4 * self.R * self.N : -3 * self.R * self.N].reshape(self.R * self.N))
            Phi_imag_T_tmp.append(action[l, -3 * self.R * self.N : -2 * self.R * self.N].reshape(self.R * self.N))
            Phi_real_R_tmp.append(action[l, -2 * self.R * self.N : -self.R * self.N].reshape(self.R * self.N))
            Phi_imag_R_tmp.append(action[l, -self.R * self.N :].reshape(self.R * self.N))

            self.W[l] = W_real.reshape(self.K, self.M) + 1j * W_imag.reshape(self.K, self.M)

        Phi_real_T = np.mean(np.array(Phi_real_T_tmp), axis=0)*np.sqrt(self.STARRIS_T)
        Phi_imag_T = np.mean(np.array(Phi_imag_T_tmp), axis=0)*np.sqrt(self.STARRIS_T)
        Phi_real_R = np.mean(np.array(Phi_real_R_tmp), axis=0)*np.sqrt(self.STARRIS_R)
        Phi_imag_R = np.mean(np.array(Phi_imag_R_tmp), axis=0)*np.sqrt(self.STARRIS_R)

        self.Phi_T = np.eye(self.R * self.N, dtype=complex) * (Phi_real_T + 1j * Phi_imag_T)
        self.Phi_R = np.eye(self.R * self.N, dtype=complex) * (Phi_real_R + 1j * Phi_imag_R)
        self._compute_h_hat_()
        self._compute_w_k_()
        reward = self._compute_reward_()

        return self.state, reward

    def _compute_EE_(self):
        EE_res = np.zeros(self.K, dtype=float)
        s_EE = 0

        for k in range((int)(self.K)):
            signal_power = 0.0
            interference_power = (float)(self.awgn_var**2)
            for j in range((int)(self.K)):
                if j == k:
                    signal_vector = np.zeros((self.U), dtype=complex)
                    for l in range((int)(self.L)):
                        tmp1 = self.h_hat_l[l, k, :, :].T
                        tmp2 = self.W[l, j, :]
                        signal_vector += tmp1 @ tmp2
                    signal_power += np.linalg.norm(signal_vector) ** 2
                else:
                    interference_vector = np.zeros((self.U), dtype=complex)
                    for l in range((int)(self.L)):
                        tmp1 = self.h_hat_l[l, k, :, :].T
                        tmp2 = self.W[l, j, :]
                        interference_vector += tmp1 @ tmp2
                    interference_power += np.linalg.norm(interference_vector) ** 2
            sinr_k = np.divide(signal_power, interference_power)
            # print(sinr_k)
            EE_res[k] = sinr_k
            s_EE += np.log2(1 + sinr_k)

        a_EE = 0.0
        a_EE = np.divide(s_EE, self.K)
        # print(f"随机信道产生的 Sum-SE: {s_EE:.2f} bps/Hz")
        return a_EE, s_EE, EE_res

    def close(self):
        pass

    def _W_generate(self):
        for l in range(self.L):
            for k in range(self.K):
                self.W[l, k, :] = np.sum(self.h_hat_l[l, k, :, :].conj().T, axis=0)
                # self.W[l, k, :] /= np.linalg.norm(self.h_hat_l[l, k, :, :])
                gamma = np.sqrt(np.divide(1.0, np.trace(self.h_hat_l[l, k, :, :] @ self.h_hat_l[l, k, :, :].conj().T)))
                self.W[l, k, :] *= gamma
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

    def _W_generate_ave(self):
        gamma = np.sqrt(self.power_limit / (2 * self.K * self.M))
        for l in range(self.L):
            for k in range(self.K):
                for m in range(self.M):
                    self.W[l, k, m] = np.random.rand() * 2 * gamma - gamma + np.random.rand() * gamma * 2j - gamma * 1j
        self.Env.W = self.W

    def _init_STARRIS_arg(self):
        self.STARRIS_R = np.divide(1, self.STARRIS_RATE + 1)
        self.STARRIS_T = self.STARRIS_RATE * self.STARRIS_R


if __name__ == "__main__":
    testEnv = CF_MIMO(2, 4, 2, 4, 50, 1e-9, 0, 1, 4)
    print(testEnv.P)
    print(testEnv.STARRIS_R, testEnv.STARRIS_T)
