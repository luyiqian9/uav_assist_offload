import gym
from gym import spaces
import numpy as np

from ..connection_schedule import connection_schedule


# TODO 所有参数默认值记得修改
class Uav:
    def __init__(self, max_angle=np.pi * 2, max_speed=10):
        self.max_angle = max_angle
        self.max_speed = max_speed


class Iotd:
    def __init__(self, data_size, tolerable_delay, capacitance_coef=0.2, cycle_bit=1, frequency=1):
        self.capacitance_coef = capacitance_coef  # 有效电容系数
        self.cycle_bit = cycle_bit
        self.frequency = frequency
        self.compute_task = []
        # self.data_size = data_size   # 单位为bit
        # self.tolerate_delay = tolerable_delay   # 单位s


class UAVEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, area_size_x, area_size_y, num_uavs=4, num_iotds=10, max_speed=10, max_angle=np.pi * 2, time_slot=1):
        super(UAVEnv, self).__init__()

        self.uav = Uav(max_angle, max_speed)
        self.iotd = Iotd()

        self.num_uavs = num_uavs
        self.nums_iotds = num_iotds
        self.area_size_x = area_size_x
        self.area_size_y = area_size_y
        self.time_slot = time_slot

        # 状态空间: 每个UAV的位置 (x, y)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] * num_uavs),  # 每个UAV的x和y坐标最小为0
            high=np.array([area_size_x, area_size_y] * num_uavs),
            # 每个UAV的x坐标最大为area_size_x，y坐标最大为area_size_y
            dtype=np.float32
        )

        # 动作空间: 每个UAV的角度增量 Δθ 和速度增量 Δv
        # 每个UAV的角度Δθ范围为[0, 2π] 速度Δv范围为[0, max_speed]
        self.action_space = spaces.Box(
            low=np.array([0, 0] * self.num_uavs),  # 角度和速度的下限
            high=np.array([self.uav.max_angle, self.uav.max_speed] * self.num_uavs),  # uav角度和速度的上限
            dtype=np.float32
        )

        # 初始化UAV的位置 (x, y) 和速度 (默认初始速度为0)
        self.state = self._initialize_state()
        self.iotd_positions = []
        self.iotd_dop = []

        # 其他参数
        self.time_step = 0
        self.max_time_steps = 15

    def _initialize_state(self):
        # 初始化每个UAV的(x, y)位置在限定范围中
        # 状态空间为[x1, y1, x2, y2, ..., xn, yn]
        # 顺序为 左上[x1]  右上[x2]  左下[x3]  右下[x4]
        uav_positions = np.array([25, 375, 375, 375, 25, 25, 375, 25])
        # 初始化设备位置 K = 10
        self.iotd_positions = [(10, 10), (20, 15), (30, 40), (15, 10), (50, 50), (60, 40), (70, 80), (90, 100), (25, 35),
                          (45, 55)]
        # 随机初始化每个iotd的dop
        self.iotd_dop = np.random.randint(1, 55, self.nums_iotds).tolist()

        return uav_positions

    def step(self, action):
        # 根据输入的动作更新每个UAV的飞行角度和速度
        # action: [Δθ1, Δv1, Δθ2, Δv2, ..., Δθn, Δvn]
        self.state = self._update_state(action)

        # 计算奖励函数
        reward = self._calculate_reward()

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
        for i in range(self.num_uavs):
            delta_theta, delta_v = actions[i]

            # 当前UAV的x和y坐标
            x, y = positions[i]

            # 计算速度方向上的增量 (cos(θ), sin(θ))
            # TODO
            dx = delta_v * np.sin(delta_theta)
            dy = delta_v * np.cos(delta_theta)

            # 更新UAV的位置
            # TODO 此处直接裁剪避免了不会uav越界 是否需要给其负奖励限制
            new_x = np.clip(x + dx * self.time_slot, 0, self.area_size_x)
            new_y = np.clip(y + dy * self.time_slot, 0, self.area_size_y)
            positions[i] = [new_x, new_y]

        # 返回更新后的状态[展平后的格式]
        return positions.flatten()

    def _calculate_reward(self):
        # 这里计算奖励函数 r_t = -Σ Σ ψ_k,n(t) - p(t)
        # ψ_k,n(t) 表示第k个设备在时隙t内完成任务的的成本
        # p(t) 是违反约束的惩罚项

        # 计算cost
        cost = 0
        cs = connection_schedule(10, 4, self.iotd_positions, self.state, 80, 4)

        for n in range(self.num_uavs):
            for k in range(self.nums_iotds):

                # local_cost = Pk * (Ecom[k, t] + Tcom[k, t])
                local_cost = 0
                # off_cost = Eoff[k, n, t] + Toff[k, n, t] + Tcom[k, n, t]
                off_cost = 0
                # cost =
                cost += cs[k, 0] * local_cost + cs[k, n] * off_cost

        # TODO 不太会设计
        penalty = 0

        # 最终奖励为负的代价和惩罚
        reward = -(cost + penalty)
        return reward

    def reset(self):
        # 重置环境到初始状态
        self.state = self._initialize_state()
        self.time_step = 0
        return self.state

    def render(self, mode='human', close=False):
        # 可视化环境状态，例如显示UAV的位置等
        pass
