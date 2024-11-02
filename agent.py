import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Actor 网络定义，用于生成动作
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 800)  # 第一层全连接 800 600 400
        self.fc2 = nn.Linear(800, 600)        # 第二层全连接
        self.fc3 = nn.Linear(600, 400)        # 第三层全连接
        self.fc_mu = nn.Linear(400, action_dim)  # 输出动作的均值
        self.fc_std = nn.Linear(400, action_dim)  # 输出动作的均值
        self.action_bound = action_bound
        # self.fc4 = nn.Linear(400, action_dim)  # 输出层，用于生成动作

    def forward(self, state):
        # 依次通过三层隐藏层并使用 ReLU 激活函数
        state = state.unsqueeze(0)   # 将 (8,) 变成 (1, 8) 即(batch_size, state_dim)

        x = F.relu(self.fc1(state))  # 第一个隐藏层 (输入 -> 800)
        x = F.relu(self.fc2(x))  # 第二个隐藏层 (800 -> 600)
        x = F.relu(self.fc3(x))  # 第三个隐藏层 (600 -> 400)

        # 通过输出层计算均值和标准差
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # 该函数将输入值转换为正数 确保标准差为正值

        # 正态分布采样和计算 log 概率
        dist = Normal(mu, std)  # 创建正态分布的实例 distribution
        normal_sample = dist.rsample()  # 重参数化采样
        log_prob = dist.log_prob(normal_sample)  # 计算sample的对数概率密度
        action = torch.tanh(normal_sample)

        # 计算 tanh 变换后分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound  # 将动作缩放到指定范围内

        return action, log_prob


# Critic 网络定义，用于估计Q值
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 第一层，输入维度为状态和动作的维度之和，输出为800个神经元
        self.fc1 = nn.Linear(state_dim + action_dim, 800)
        # 第二层，输入为800个神经元，输出为600个神经元
        self.fc2 = nn.Linear(800, 600)
        # 第三层，输入为600个神经元，输出为400个神经元
        self.fc3 = nn.Linear(600, 400)
        # 输出层，输入为400个神经元，输出为1个 Q 值
        self.fc_out = nn.Linear(400, 1)

    def forward(self, state, action):
        state = state.unsqueeze(0)  # 将 (8,) 变成 (1, 8) 即(batch_size, state_dim)格式
        action = action.unsqueeze(0)  # 将 (8,) 变成 (1, 8)
        # 将状态和动作在维度1上拼接，形成联合输入
        cat = torch.cat([state, action], dim=1)
        # 通过全连接层并逐层进行ReLU激活
        x = F.relu(self.fc1(cat))  # 第一层激活
        x = F.relu(self.fc2(x))  # 第二层激活
        x = F.relu(self.fc3(x))  # 第三层激活

        # 输出层不使用激活函数，直接生成 Q 值
        return self.fc_out(x)


# 经验回放缓冲区，用于存储状态、动作、奖励和下一个状态
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []                  # 缓冲区初始化为空列表
        self.max_size = max_size          # 最大缓冲区大小
        self.ptr = 0                      # 指针，用于循环替换

    def add(self, experience):
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)    # 如果未达到最大容量，则添加新经验
        else:
            self.buffer[self.ptr] = experience # 达到最大容量时，替换旧的经验
            self.ptr = (self.ptr + 1) % self.max_size # 更新指针

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)         # 随机抽取批次样本
        states, actions, rewards, next_states = zip(*batch)    # 分别提取状态、动作、奖励和下一个状态
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states))


# SAC 智能体定义，包含 Actor 和 Critic 网络
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, device, buffer_size=100000):
        # 初始化 Actor 和 Critic 网络
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)  # 策略网络
        self.critic1 = Critic(state_dim, action_dim).to(device)   # 第一个 Q 网络
        self.critic2 = Critic(state_dim, action_dim).to(device)   # 第二个 Q 网络

        # 目标网络 用于稳定训练
        self.target_critic1 = Critic(state_dim, action_dim).to(device)  # 第一个目标 Q 网络
        self.target_critic2 = Critic(state_dim, action_dim).to(device)  # 第二个目标 Q 网络

        # 将 Q 网络参数复制到目标 Q 网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器，用于更新网络参数
        lr_a, lr_c, alpha_lr = 3e-4, 3e-3, 3e-4

        # Adam是一种种优化器 基于自适应学习率算法 能够快速有效地进行梯度下降优化
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)  # 策略网络的优化器
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_c)  # Q1 网络的优化器
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_c)  # Q2 网络的优化器

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = -8  # 目标熵的大小
        self.gamma = 0.99         # discount rate ???
        self.tau = 0.005          # 软更新参数
        self.device = device

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)  # 初始化经验回放池

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为张量，并添加批次维度
        action = self.actor(state_tensor).detach().numpy()[0]  # 使用策略网络生成动作，并转换为 numpy 数组
        return action

    def _soft_update(self, target, source, tau):
        # 使用软更新公式更新目标网络
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update(self, batch_size, gamma=0.99, tau=0.005):
        if len(self.replay_buffer.buffer) < batch_size:
            return  # 如果经验池中数据不足，跳过更新

        # 从经验池中采样
        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)

        # 将数据转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # 奖励需要增加一个维度
        next_states = torch.FloatTensor(next_states)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions = self.actor(next_states)               # 下一个状态的动作
            target_q1 = self.target_critic1(next_states, next_actions)  # 第一个目标 Q 值
            target_q2 = self.target_critic2(next_states, next_actions)  # 第二个目标 Q 值
            target_q = torch.min(target_q1, target_q2)                 # 选择最小的 Q 值
            yj = rewards + gamma * target_q                            # 计算目标值

        # 更新 Critic 网络
        current_q1 = self.critic1(states, actions)                   # 当前 Q1 值
        current_q2 = self.critic2(states, actions)                   # 当前 Q2 值
        critic_loss1 = nn.MSELoss()(current_q1, yj)                  # Q1 损失
        critic_loss2 = nn.MSELoss()(current_q2, yj)                  # Q2 损失

        # 反向传播并更新 Critic 网络
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        # 更新 Actor 网络
        new_actions = self.actor(states)                       # 新动作
        actor_loss = -self.critic1(states, new_actions).mean() # 计算策略损失

        self.actor_optimizer.zero_grad()
        actor_loss.backward()                                  # 反向传播
        self.actor_optimizer.step()                            # 更新策略网络参数

        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1, tau)
        self._soft_update(self.target_critic2, self.critic2, tau)
