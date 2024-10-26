import numpy as np
import matplotlib.pyplot as plt


def plot_return(return_list):
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Training Return')
    plt.show()


iotd_pos = (1, 1)
uav_pos = (2, 2)
print(np.linalg.norm(np.array(iotd_pos) - np.array(uav_pos)))
print(np.exp(0))
# test = np.array([10, 20] * 4)
# position = test.reshape(4, 2)
# position[0] = [16, 18]
# print(position)
# x, y = position[0]
# print(test)
# print(x, y)