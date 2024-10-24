import numpy as np

ci = []
MI, LU = {}, {}


# 初始化参数
def initialize(K, N):
    ci = np.zeros((K, N + 1), dtype=int)  # 连接矩阵，初始为全本地计算
    ci[:, 0] = 1  # 所有设备初始为本地计算
    MI, LU = {}, {j: [] for j in range(1, N + 1)}  # 未决连接设备集和UAV负载列表
    return ci, MI, LU


# 计算IoTD与UAV的距离
def calculate_distance(iotd_pos, uav_pos):
    return np.linalg.norm(np.array(iotd_pos) - np.array(uav_pos))


# pop出优先级最低的
def pop_low_priority(l):
    LU[l] = sorted(LU[l], key=lambda x: (x[1], x[2]))  # 按优先级排序
    removed_task = LU[l].pop(0)
    m = removed_task[0]  # 获取移除任务的编号
    ci[m, l] = 0
    ci[m, 0] = 1


# 连接调度算法
def connection_schedule(K, N, iotd_positions, uav_positions, dlink, Cmax):

    # ci 二维数组 记录每个iotd的连接状态
    # MI 记录未确定连接的iotd
    # LU 记录每个uav当前的连接任务列表

    # 遍历每个IoTD设备
    for i in range(K):
        LI = []  # 当前 iotd 被多少个 uav 覆盖

        # 遍历每个UAV
        for j in range(1, N + 1):
            dGi_j = calculate_distance(iotd_positions[i], uav_positions[j - 1])

            # 如果距离小于通信阈值，将该UAV加入候选列表
            if dGi_j <= dlink:
                LI.append((j, dGi_j))

        # 如果仅有一个UAV覆盖IoTD
        if len(LI) == 1:
            l = LI[0][0]
            LU[l].append((i, 0, 0))  # uav当前的连接任务列表 0，0模拟设备优先级Pi和任务数据量Di(t)
            ci[i, 0] = 0  # 第零列值为0 表示当前 IoTD 不再执行本地计算
            ci[i, l] = 1  # 值为1 表示当前 iotd 已连接上编号为 l 的 uav

            # 如果UAV负载超出最大容量，移除优先级低的任务
            if len(LU[l]) > Cmax:
                pop_low_priority(l)  # 编号为 l 的uav和其连接的iotd的信息

        elif len(LI) > 1:
            MI[i] = LI  # 记录多UAV覆盖的IoTD  LI(j, dGi_j)

    # 处理多UAV覆盖的IoTD
    # TODO 会不会出现做完循环之后 还有 iotd 是在本地计算
    for i in MI:
        # 找到所有任务负载最少的 UAV 编号
        min_load = min(len(LU[x[0]]) for x in MI[i])  # 找到最小负载值
        candidate_uavs = [x for x in MI[i] if len(LU[x[0]]) == min_load]  # 找到负载最少的UAV列表

        # 如果最少负载的 UAV 不止一个，选择距离 IoTD 最近的 UAV
        if len(candidate_uavs) > 1:
            j = min(candidate_uavs, key=lambda x: x[1])[0]  # 按距离选最近的
        else:
            j = candidate_uavs[0][0]  # 如果只有一个UAV，直接选择

        # 将 IoTD 任务分配给 UAV j
        LU[j].append((i, 0, 0))  # 模拟Pi和Di(t)
        ci[i, 0] = 0  # 更新连接矩阵，表示不再进行本地计算
        ci[i, j] = 1  # 将IoTD连接到UAV j

        # 检查UAV负载是否超出最大容量
        if len(LU[j]) > Cmax:
            pop_low_priority(j)  # 移除优先级最低的任务

    return ci


if __name__ == '__main__':
    K = 10  # 10个IoTD
    N = 3  # 3个UAV

    ci, MI, LU = initialize(K, N)

    iotd_positions = [(10, 10), (20, 15), (30, 40), (15, 10), (50, 50), (60, 40), (70, 80), (90, 100), (25, 35),
                      (45, 55)]
    uav_positions = [(20, 20), (60, 60), (90, 90)]
    dlink = 50  # 通信链路最大距离
    Cmax = 3  # 每个UAV的最大连接容量

    # 运行算法
    ci = connection_schedule(K, N, iotd_positions, uav_positions, dlink, Cmax)

    for i in range(len(ci)):
        for j in range(len(ci[i])):
            print(ci[i][j], end=" ")
        print()
