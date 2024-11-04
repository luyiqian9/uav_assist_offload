import numpy as np


class Uav:
    def __init__(self, cycle_bit, max_angle=np.pi * 2, max_speed=15, cmax=2,
                 fnmax=100):
        self.max_angle = max_angle
        self.max_speed = max_speed
        self.cycle_bit = cycle_bit
        self.max_connection = cmax
        self.fnmax = fnmax

    def calculate_new_position(self, num_uavs, actions, positions, time_slot_size, area_size_x, area_size_y):
        for i in range(num_uavs):
            delta_theta, delta_v = actions[i]

            # 当前UAV的x和y坐标
            x, y = positions[i]

            # 计算速度方向上的增量 (cos(θ), sin(θ))
            # TODO
            dx = delta_v * np.sin(delta_theta)
            dy = delta_v * np.cos(delta_theta)

            # 更新UAV的位置
            # TODO 此处直接裁剪保证uav不会越界 是否需要给其负奖励限制
            new_x = np.clip(x + dx * time_slot_size, 0, area_size_x)
            new_y = np.clip(y + dy * time_slot_size, 0, area_size_y)
            positions[i] = [new_x, new_y]

        return np.round(positions, 2)

    def tostring(self):
        pass

        """
        for i in range(self.num_uavs):
            delta_theta, delta_v = actions[i]

            # 当前UAV的x和y坐标
            x, y = positions[i]

            # 计算速度方向上的增量 (cos(θ), sin(θ))
            # TODO
            dx = delta_v * np.sin(delta_theta)
            dy = delta_v * np.cos(delta_theta)

            # 更新UAV的位置
            # TODO 此处直接裁剪保证uav不会越界 是否需要给其负奖励限制
            new_x = np.clip(x + dx * self.time_slot_size, 0, self.area_size_x)
            new_y = np.clip(y + dy * self.time_slot_size, 0, self.area_size_y)
            positions[i] = [new_x, new_y]

        # 返回更新后的状态[展平后的格式]
        return positions.flatten()
        """