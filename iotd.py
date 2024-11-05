import numpy as np


class Iotd:
    def __init__(self, dop=0, transmit_power=0.1, capacitance_coef=1e-28, cycle_bit=1000,
                 frequency_local=1, frequency_alloc=1, fkmax=2.5e9):
        self.capacitance_coef = capacitance_coef  # 有效电容系数
        self.cycle_bit = cycle_bit
        self.frequency_local = frequency_local
        self.frequency_alloc = frequency_alloc
        """
            cpu 计算1bit需要 1000 个时钟周期
            cpu 计算频率上限为 2.5e9 Hz 周期/秒
            t = 周期数n / 频率f
            1600 * 1024 * 1000 / 2.5e9
        """
        self.compute_task = []  # 保存每个时隙的任务大小 单位为bit
        self.transmit_power = transmit_power
        self.dop = dop
        self.position = ()

        self.fkmax = fkmax  # 2.5GHz

    def get_t_star(self, t):
        t_star = 0
        # print(f"task size = {self.compute_task[t]}")
        # print(f"bit = {np.cbrt(2 * self.capacitance_coef)}")
        t_best = self.cycle_bit * self.compute_task[t] * 1024 * np.cbrt(2 * self.capacitance_coef)
        # print(f"tbest = {t_best}")
        t_min = self.cycle_bit * self.compute_task[t] * 1024 / self.fkmax
        # print(f"tmin = {t_min}")
        if t_min < t_best < 1:
            t_star = t_best
        if t_best < t_min < 1:
            t_star = t_min
        if t_best > 1:
            t_star = 1

        # 顺便更新最优本地计算频率 频率和时间可以互推
        self.frequency_local = self.cycle_bit * self.compute_task[t] / t_star
        return t_star

    def gen_tasks(self, max_time_steps):
        for i in range(max_time_steps + 1):
            task_size = np.random.randint(1200, 1600)  # 单位 kb
            # print(f"task size: {task_size}")
            self.compute_task.append(task_size)
