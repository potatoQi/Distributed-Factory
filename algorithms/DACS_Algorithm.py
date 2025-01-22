import autograd.numpy as anp  # 使用 autograd 的 numpy 版本来支持自动求导
from autograd import elementwise_grad  # 导入 grad 和 elementwise_grad 函数用于计算梯度
import numpy as np
import random
import sys

class DACS_Algorithm:
    def __init__(self, config):
        self.alpha = config.get('learning_rate', 0.01)
        self.iter_num = config.get('max_iterations', 1000)
        self.bug_op = config.get('bug_op', 0)

    def phi_function(self, X):
        """
        计算 \(\phi_i(x_i)\) 函数，默认情况下 \(\phi_i(x_i) = x_i\)
        """
        return X

    def cost_function(self, X, Sigma_X, p, b):
        """
        计算代价函数 f_i(x_i, sigma(x)) 的总和
        """
        term1 = 0.5 * anp.sum(anp.square(X - p), axis=1)  # 计算 (X_i - p_i)² 的总和
        term2 = 0.5 * anp.sum(anp.square(X - b), axis=1)  # 计算 (X_i - b)² 的总和
        term3 = anp.sum(anp.square(Sigma_X - b), axis=1)  # 计算 (Sigma_X - b)² 的总和并乘以权重
        term4 = anp.sum(anp.square(X - Sigma_X), axis=1)  # 计算 (X_i - Sigma_X)² 的总和
        
        return term1 + term2 + term3 + term4  # 返回代价函数的总和

    def run(self, B_list):
        B_list = [anp.array(B) for B in B_list]
        self.n = len(B_list[0])
        B_standard = anp.full((self.n, self.n), 1 / self.n)

        grad_f_x = elementwise_grad(self.cost_function, 0)  # 对 X 计算元素级别的梯度
        grad_f_sigma = elementwise_grad(self.cost_function, 1)  # 对 Sigma_X 计算元素级别的梯度
        grad_phi_x = elementwise_grad(self.phi_function, 0)  # 对 \(\phi(x)\) 计算元素级别的梯度

        self.X_best = anp.zeros((self.n, 2))
        self.p = anp.floor(anp.random.uniform(1, 10, size=(self.n, 2))).astype(float)
        self.initialize(grad_f_sigma)  # 初始化状态变量

        results = []
        for T in range(2000):   # 因为这里是要得到准确答案，所以固定跑1w次
            self.B = B_standard.copy()
            self.update_all_params(grad_f_x, grad_f_sigma, grad_phi_x, 0)  # 更新所有参数
            sum_res = self.calculate_residual()  # 计算残差矩阵
            results.append(sum_res)
            self.backup()  # 备份当前状态

        print('最终cost为: ', np.sum(self.cost_function(self.X, self.Z, self.p, self.b)))

        # 再来一次得到收敛图
        self.X_best = self.X
        self.initialize(grad_f_sigma)  # 初始化状态变量
        results = []
        len_B_list = len(B_list)
        for T in range(self.iter_num):
            if 450 <= T <= 500 or 950 <= T <= 1000: # 手动造成节点故障 / 通信中断
                if self.bug_op == 1:
                    num_nodes_to_zero = self.n // 2  # 计算50%的节点数
                    indices = np.random.choice(self.n, num_nodes_to_zero, replace=False)
                    self.X[indices] = (0, 0)
                elif self.bug_op == 2:
                    len_B_list = (int)(0.5 * len(B_list)) # 随机扔掉50%的时变图
                    if len_B_list == 0: len_B_list = 1
            else:
                len_B_list = len(B_list)

            self.B = B_list[T % len_B_list].copy()
            self.update_all_params(grad_f_x, grad_f_sigma, grad_phi_x, T)  # 更新所有参数
            sum_res = self.calculate_residual()  # 计算残差矩阵
            results.append(sum_res)
            self.backup()  # 备份当前状态

        return anp.sum(anp.array(results), axis=1).tolist()

    def initialize(self, grad_f_sigma):
        """
        初始化算法的必要参数和变量
        """
        n, d = self.n, 2  # n 是节点数，d 是每个 x_i 的维度
        self.b = anp.array([5.0, 5.0])  # 初始化 b 向量, 大小为 (2,)
        # self.p = anp.array([[1.0, 6.0], [4.0, 8.0], [9.0, 8.0], [4.0, 2.0], [8.0, 3.0]])    # 5个入侵者的坐标
        self.X = anp.floor(anp.random.uniform(1, 10, size=(n, d))).astype(float) # 随机初始化 X 矩阵
        self.S = self.X.copy()
        self.Y = grad_f_sigma(self.X, self.S, self.p, self.b)
        self.V = anp.ones((n, d))
        self.U = anp.zeros((n, d))
        self.P = self.Y.copy()
        self.W = anp.zeros((n, d))
        self.Z = self.X.copy()
        self.XX = self.X.copy()
        # self.X_best = anp.array([[4.0, 5.25], [4.75, 5.75], [6.0, 5.75], [4.75, 4.25], [5.75, 4.5]])    # 5个最终位置

    def calculate_residual(self):
        # 计算残差矩阵
        return anp.sqrt(anp.sum(anp.square(self.X - self.X_best), axis=1))  # 返回每个节点的残差

    def update_all_params(self, grad_f_x, grad_f_sigma, grad_phi_x, iter_num):
        """
        根据公式更新参数
        """
        if 0 <= iter_num <= 200 or 210 <= iter_num <= 400: # 手动阻止辅助变量的更新
            if self.bug_op == 3:
                self.V_nxt = self.V
            else:
                self.V_nxt = self.B @ self.V
        else:
            self.V_nxt = self.B @ self.V

        grad_f_1 = grad_f_x(self.X, self.Z, self.p, self.b)
        self.X_nxt = self.X - self.alpha * (grad_f_1 + grad_phi_x(self.X) * self.P)

        if 0 <= iter_num <= 200 or 210 <= iter_num <= 400: # 手动阻止辅助变量的更新
                if self.bug_op == 3:
                    self.S_nxt = self.S
                else:
                    self.S_nxt = self.B @ self.S + self.phi_function(self.X_nxt) - self.phi_function(self.X)
        else:
            self.S_nxt = self.B @ self.S + self.phi_function(self.X_nxt) - self.phi_function(self.X)

        self.W_nxt = self.B @ self.S

        self.Z_nxt = self.W_nxt / self.V_nxt

        if 0 <= iter_num <= 200 or 210 <= iter_num <= 400: # 手动阻止辅助变量的更新
                if self.bug_op == 3:
                    self.Y_nxt = self.Y
                else:
                    self.Y_nxt = self.B @ self.Y + grad_f_sigma(self.X_nxt, self.Z_nxt, self.p, self.b) - grad_f_sigma(self.X, self.Z, self.p, self.b)
        else:
            self.Y_nxt = self.B @ self.Y + grad_f_sigma(self.X_nxt, self.Z_nxt, self.p, self.b) - grad_f_sigma(self.X, self.Z, self.p, self.b)

        self.U_nxt = self.B @ self.Y

        self.P_nxt = self.U_nxt / self.V_nxt

    def backup(self):
        """
        将临时状态应用为当前状态
        """
        self.V = self.V_nxt.copy()
        self.W = self.W_nxt.copy()
        self.Z = self.Z_nxt.copy()
        self.S = self.S_nxt.copy()
        self.XX = self.X.copy()  # 保存当前的 X 作为下次的 XX
        self.X = self.X_nxt.copy()
        self.Y = self.Y_nxt.copy()
        self.U = self.U_nxt.copy()
        self.P = self.P_nxt.copy()

# 示例使用
if __name__ == "__main__":
    config = {
        "learning_rate": 0.01,
        "max_iterations": 2000
    }
    algorithm = DACS_Algorithm(config)
    B = anp.array([  # 示例邻接矩阵
        [1/2, 0, 0, 1/2, 1/2],
        [1/2, 1/2, 0, 0, 0],
        [0, 1/2, 1/3, 0, 0],
        [0, 0, 1/3, 1/2, 0],
        [0, 0, 1/3, 0, 1/2]
    ])
    B1 = anp.array([
        [1/2, 0, 0, 0, 0],
        [1/2, 1, 0, 0, 0],
        [0, 0, 1/3, 0, 0],
        [0, 0, 1/3, 1/2, 0],
        [0, 0, 1/3, 1/2, 1]
    ])
    B2 = anp.array([
        [1/2, 0, 0, 0, 1/2],
        [1/2, 1/2, 0, 0, 0],
        [0, 1/2, 1, 0, 0],
        [0, 0, 0, 1/2, 0],
        [0, 0, 0, 1/2, 1/2]
    ])
    residuals = algorithm.run([B1,B2])
    print(residuals[-1])
