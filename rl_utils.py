import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """添加新的经验到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states).reshape(batch_size, -1), np.array(actions), np.array(rewards), \
               np.array(next_states).reshape(batch_size, -1), np.array(dones)

    def size(self):
        """ 返回缓冲区中的样本数量 """
        return len(self.buffer)


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """ 训练离线策略的代理 """
    return_list = []
    for i in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        episode_return = 0
        while not done:
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action[0])  # 这里取 action 的第一个值
            replay_buffer.add(state, action, reward, next_state, done)
            episode_return += reward
            state = next_state

            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'rewards': b_r,
                    'next_states': b_ns,
                    'dones': b_d
                }
                agent.update(transition_dict)
        return_list.append(episode_return)
    return return_list


def plot_return(return_list):
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Training Return')
    plt.show()


test = np.array([10, 20] * 4)
position = test.reshape(4, 2)
position[0] = [16, 18]
print(position)
x, y = position[0]
print(test)
print(x, y)