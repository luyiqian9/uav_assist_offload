import numpy as np
import matplotlib.pyplot as plt


def plot_return(return_list):
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Training Return')
    plt.show()


# iotd_pos = np.array([1, 1], dtype=np.float32)
# uav_pos = np.array([2, 2], dtype=np.float32)
# iotd_pos = iotd_pos.reshape(1, 2)
# uav_pos = uav_pos.reshape(1, 2)
# print(np.linalg.norm(np.array(iotd_pos) - np.array(uav_pos)))
# print(np.exp(0))

print(np.cbrt(7))
# test = np.array([10, 20] * 4)
# position = test.reshape(4, 2)
# position[0] = [16, 18]
# print(position)
# x, y = position[0]
# print(test)
# print(x, y)