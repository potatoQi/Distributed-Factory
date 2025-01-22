import autograd.numpy as anp  # 使用 autograd 的 numpy 版本来支持自动求导
from autograd import elementwise_grad  # 导入 grad 和 elementwise_grad 函数用于计算梯度

class MDGT_Algorithm:
    def __init__(self, config):
        self.alpha = config.get('alpha', 0.01)
        self.beta = config.get('beta', 0.01)
        self.iter_num = config.get('max_iterations', 1000)
        self.r1 = anp.array([1.0, 4.0, 9.0, 4.0, 8.0])
        self.r2 = anp.array([6.0, 8.0, 8.0, 2.0, 3.0])
        self.x_best1 = anp.array([4.0, 4.75, 6.0, 4.75, 5.75])
        self.x_best2 = anp.array([5.25, 5.75, 5.75, 4.25, 4.5])

    def f1(self, x, s, r):
        return x - r + x - 5 + 2 * (x - s)

    def f2(self, x, s):
        return 2 * (s - 5) - 2 * (x - s)

    def run(self, A_list):
        A_list = [anp.array(A) for A in A_list]
        A = anp.array(A_list[0])
        self.A = A
        self.n = len(A)
        res1, res2 = [], []

        self.initialize(self.r1, self.x_best1)

        for T in range(self.iter_num):
            A = A_list[T % len(A_list)].copy()
            self.A = A.copy()
            res1.append(self.calculate_residual())
            self.update_all_params()
            self.backup()

        self.initialize(self.r2, self.x_best2)

        for T in range(self.iter_num):
            A = A_list[T % len(A_list)].copy()
            self.A = A.copy()
            res2.append(self.calculate_residual())
            self.update_all_params()
            self.backup()

        return [x + y for x, y in zip(res1, res2)]

    def initialize(self, r, x_best):
        self.r = r
        self.x_best = x_best
        # self.x = anp.random.randint(1, 5, self.n).astype(float)   受初始值影响很大
        self.x = anp.zeros(self.n)
        self.y = self.x.copy()
        self.v = anp.eye(self.n)
        self.s = self.x.copy() / self.n
        self.q = 2 * (self.s.copy() - 5) / self.n
        self.xx = self.x.copy()
        self.xxx = self.x.copy()
        self.yy = self.y.copy()
        self.vv = self.v.copy()
        self.ss = self.s.copy()
        self.qq = self.q.copy()

    def calculate_residual(self):
        return anp.sqrt(anp.sum(anp.square(self.x - self.x_best)))

    def update_all_params(self):
        for i in range(self.n):
            self.vv[i] = anp.dot(self.A[i], self.v)
            self.xx[i] = self.y[i] - self.alpha * (self.f1(self.y[i], self.s[i], self.r[i]) + self.q[i]) \
                                   + self.beta * (self.x[i] - self.xxx[i])
            
            self.yy[i] = self.xx[i] + self.beta * (self.xx[i] - self.x[i])

            self.ss[i] = anp.dot(self.A[i], self.s) + self.yy[i] / (self.n * self.vv[i, i]) \
                                                    - self.y[i] / (self.n * self.v[i, i])
            
            self.qq[i] = anp.dot(self.A[i], self.q) + self.f2(self.yy[i], self.ss[i]) / (self.n * self.vv[i, i]) \
                                                    - self.f2(self.y[i], self.s[i]) / (self.n * self.v[i, i])

    def backup(self):
        self.xxx = self.x.copy()
        self.x = self.xx.copy()
        self.y = self.yy.copy()
        self.v = self.vv.copy()
        self.s = self.ss.copy()
        self.q = self.qq.copy()

# 示例使用
if __name__ == "__main__":
    config = {
        "alpha": 0.01,
        "beta": 0.01,
        "max_iterations": 1000
    }
    algorithm = MDGT_Algorithm(config)
    A = anp.array([
        [1/2, 0, 1/2, 0, 0],
        [0, 1/3, 1/3, 0, 1/3],
        [1/4, 1/4, 1/4, 1/4, 0],
        [0, 1/3, 0, 1/3, 1/3],
        [0, 0, 0, 1/2, 1/2]
    ])
    residuals = algorithm.run([A])
    print("残差:", residuals[-1])
