import gym  # 导入OpenAI Gym库，用于构建和管理环境
from agent import SACAgent
from envs.uav_env import UAVEnv

import numpy as np
from tqdm import tqdm  # 导入进度条模块，用于显示进度
import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn.functional as F  # 导入PyTorch中的功能模块
from torch.distributions import Normal  # 导入正态分布类
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图

""" 
    先得make一个环境出来 
"""
# 注册环境
gym.envs.register(
    id='UAVEnv-v0',
    entry_point='envs.uav_env:UAVEnv',
)

""" 
    超参数设置
"""

""" 
    写训练逻辑
"""
# 初始化环境和智能体
env = UAVEnv(400, 400)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim)

# 训练智能体
state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.replay_buffer.add((state, action, reward, next_state))
    state = next_state

    # 每一步都更新智能体
    agent.update(batch_size=64)


""" 
    绘制训练回报
"""

