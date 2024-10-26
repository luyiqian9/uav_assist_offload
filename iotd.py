class Iotd:
    def __init__(self, dop=0, transmit_power=1, capacitance_coef=0.2, cycle_bit=1,
                 frequency_local=1, frequency_alloc=1):
        self.capacitance_coef = capacitance_coef  # 有效电容系数
        self.cycle_bit = cycle_bit
        self.frequency_local = frequency_local
        self.frequency_alloc = frequency_alloc
        self.compute_task = []  # 保存每个时隙的任务大小 单位为bit
        self.transmit_power = transmit_power
        self.dop = dop
        self.position = ()