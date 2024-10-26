import gym
from gym import spaces
import numpy as np

from multi_UAV.connection_schedule import connection_schedule
from multi_UAV.uav import Uav
from multi_UAV.iotd import Iotd


# TODO 所有参数默认值记得修改
class UAVEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, area_size_x, area_size_y, num_uavs=4, num_iotds=10, max_speed=10,
                 max_angle=np.pi * 2, time_slot_size=1):
        super(UAVEnv, self).__init__()

        self.uav = Uav(max_angle, max_speed)
        self.iotds: list[Iotd] = []

        self.num_uavs = num_uavs
        self.nums_iotds = num_iotds
        self.area_size_x = area_size_x
        self.area_size_y = area_size_y
        self.time_slot_size = time_slot_size

        # 状态空间: 每个UAV的位置 (x, y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] * num_uavs).astype(np.float32),  # 每个UAV的x和y坐标最小为0
            high=np.array([area_size_x, area_size_y] * num_uavs).astype(np.float32),
            # 每个UAV的x坐标最大为area_size_x，y坐标最大为area_size_y
            dtype=np.float32
        )

        # 动作空间: 每个UAV的角度增量 Δθ 和速度增量 Δv
        # 每个UAV的角度Δθ范围为[0, 2π] 速度Δv范围为[0, max_speed]
        self.action_space = spaces.Box(
            low=np.array([0, 0] * self.num_uavs).astype(np.float32),  # 角度和速度的下限
            high=np.array([self.uav.max_angle, self.uav.max_speed] * self.num_uavs).astype(np.float32),  # uav角度和速度的上限
            dtype=np.float32
        )

        # 初始化UAV的位置 (x, y) 和速度 (默认初始速度为0)
        self.state = self._initialize_state()
        self.iotd_positions = []

        # 其他参数
        self.time_step = 0
        self.max_time_steps = 15

    def _initialize_state(self):
        # 初始化每个UAV的(x, y)位置在限定范围中
        # 状态空间为[x1, y1, x2, y2, ..., xn, yn]
        # 顺序为 左上[x1]  右上[x2]  左下[x3]  右下[x4]
        uav_positions = np.array([25, 375, 375, 375, 25, 25, 375, 25], dtype=np.float32)

        # 初始化全部设备 设备位置 每个iotd的dop K = 10
        self.iotd_positions = [(10, 10), (20, 15), (30, 40), (15, 10), (50, 50), (60, 40), (70, 80), (90, 100), (25, 35),
                          (45, 55)]
        iotd_dops = np.random.randint(1, 55, self.nums_iotds).tolist()

        for i in range(self.nums_iotds):
            ins = Iotd()
            # ins.position = iotd_positions[i]
            ins.dop = iotd_dops[i]
            self.iotds.append(ins)

        return uav_positions

    def step(self, action):
        # 根据输入的动作更新每个UAV的飞行角度和速度
        # action: [Δθ1, Δv1, Δθ2, Δv2, ..., Δθn, Δvn]
        self.state = self._update_state(action)

        # 计算奖励函数
        reward = self._calculate_reward(self.time_step)

        # 检查是否完成
        done = self.time_step >= self.max_time_steps

        # 递增时间步
        self.time_step += 1

        return self.state, reward, done, {}

    def _update_state(self, action):
        # 当前状态：每个UAV的位置 [x1, y1, x2, y2, ..., xn, yn]
        positions = self.state.reshape(self.num_uavs, 2)

        # 动作：每个UAV的角度和速度增量 [Δθ1, Δv1, Δθ2, Δv2, ..., Δθn, Δvn]
        actions = action.reshape(self.num_uavs, 2)

        # 计算新的位置
        positions: np.ndarray = self.uav.calculate_new_position(self.num_uavs, actions, positions, self.time_slot_size,
                                                    self.area_size_x, self.area_size_y)
        # 返回更新后的状态[展平后的格式]
        return positions.flatten()

    def _calculate_reward(self, t):
        # 这里计算奖励函数 r_t = -Σ Σ ψ_k,n(t) - p(t)
        # ψ_k,n(t) 表示第k个设备在时隙t内完成任务的的成本
        # p(t) 是违反约束的惩罚项

        # 计算cost
        cost = 0
        cs = connection_schedule(10, 4, self.iotd_positions, self.state, 80, 4)

        for n in range(self.num_uavs):
            for k in range(self.nums_iotds):
                iotd = self.iotds[k]
                Tcomk_t = iotd.cycle_bit * iotd.compute_task[t] / iotd.frequency_local  # 本地计算时间
                Ecomk_t = Tcomk_t * iotd.capacitance_coef * iotd.frequency_local ** 3  # 本地能量消耗
                # local_cost = Pk * (Ecom[k, t] + Tcom[k, t])
                local_cost = iotd.dop * (Ecomk_t + Tcomk_t)

                # 在5G通信中，常用频段包括28 GHz和39 GHz
                c, gt, gr, f, h, ka_f = 3e8, 1, 1, 28, 5, 1
                # TODO 到时候迁移到moc中 可能对当前uav的state会有影响 也就是uav的位置是不是实时在环境中更新 or 何处更新
                dgi_u = np.linalg.norm(np.array(iotd.position) - np.array(self.state.reshape(self.num_uavs, 2)[n]))
                di_u = np.sqrt(dgi_u ** 2 + h ** 2)
                propagation_loss = c * np.sqrt(gt * gr) / (4 * np.pi * f * di_u)
                molecular_absorption = np.exp(-0.5 * ka_f * di_u)
                # hk_n_l_t
                path_loss = propagation_loss * molecular_absorption

                # 还没仔细研究计算公式
                multipath_fade = 1
                misalignment_fade = 1

                hk_n = path_loss * multipath_fade * misalignment_fade  # 信道增益

                gaussian_noise = 1
                rk_n = iotd.transmit_power * hk_n / gaussian_noise  # 信噪比

                B = 1
                Rk_n = B * np.log2(1 + rk_n)  # 卸载率

                Toffk_n = iotd.compute_task[t] / Rk_n  # 卸载时间
                Eoffk_n = Toffk_n * iotd.transmit_power  # 卸载能量
                Tcomk_n = self.uav.cycle_bit * iotd.compute_task[t] / iotd.frequency_alloc  # uav计算时间
                off_cost = Eoffk_n + Toffk_n + Tcomk_n
                # off_cost = Eoff[k, n, t] + Toff[k, n, t] + Tcom[k, n, t]

                # cost =
                cost += cs[k, 0] * local_cost + cs[k, n] * off_cost

        # TODO 不太会设计
        penalty = 0

        # 最终奖励为负的代价和惩罚
        reward = -(cost + penalty)
        return reward

    def reset(self, **kwargs):
        # 重置环境到初始状态
        self.state = self._initialize_state()
        self.time_step = 0
        return self.state, {}

    def render(self, mode='human', close=False):
        # 可视化环境状态，例如显示UAV的位置等
        pass


this = UAVEnv(400, 400)
this._initialize_state()

print(this._update_state(np.random.randint(1, 5, 8)))

# iotds = this.iotds
# for i in range(this.nums_iotds):
#     print(iotds[i].dop)
#     print(iotds[i].position)
