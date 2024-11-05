import collections

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
        # state = state.unsqueeze(0)   # 将 (8,) 变成 (1, 8) 即(batch_size, state_dim)

        x = F.relu(self.fc1(state))  # 第一个隐藏层 (输入 -> 800)
        x = F.relu(self.fc2(x))  # 第二个隐藏层 (800 -> 600)
        x = F.relu(self.fc3(x))  # 第三个隐藏层 (600 -> 400)

        # 通过输出层计算均值和标准差
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # 该函数将输入值转换为正数 确保标准差为正值

        # 正态分布采样和计算 log 概率
        dist = Normal(mu, std)  # 创建正态分布的实例 distribution
        normal_sample = dist.rsample()  # 重参数化采样 确保梯度可以反传
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
        # 在 update 中使用时 已经转换过张量格式了
        # state = state.unsqueeze(0)  # 将 (8,) 变成 (1, 8) 即(batch_size, state_dim)格式
        # action = action.unsqueeze(0)  # 将 (8,) 变成 (1, 8)
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
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 缓冲区初始化为空列表

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))  # 如果未达到最大容量，则添加新经验

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)         # 随机抽取批次个样本元组
        # 将batch列表解压生成多个元组列表 每个元组的第一个元素打包成state 第二个元素打包成action 依此类推
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def size(self):
        return len(self.buffer)


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

        # 使用alpha的log值,可以使训练结果比较稳定    α【熵的温度参数】
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32)  # 定义一个PyTorch张量 进一步用于梯度优化
        self.log_alpha.requires_grad = True  # 使log_alpha在优化步骤中参与参数更新 这对于通过梯度下降方法优化log_alpha很重要
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)  # 定义优化器优化 alpha

        self.target_entropy = -8  # 目标熵的大小
        self.gamma = 0.99         # discount rate
        self.tau = 0.005          # 软更新参数
        self.device = device

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)  # 初始化经验回放池

    def select_action(self, state):
        # 将状态转换为张量，并添加批次维度 即(1, state_dim) 便于适配学习模型
        state = torch.tensor([state], dtype=torch.float32).to(self.device)

        action = self.actor(state)[0]  # 使用策略网络生成动作，并转换为 numpy 数组
        action = action.detach().cpu().numpy()   # 将 action 转成 numpy 数组，方便后续与 NumPy 库进行兼容的操作
        # print(f"selected action = {action}")
        return action

    # def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
    def calc_target(self, rewards, next_states):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)

        # 计算策略熵，用于增强策略的探索性
        entropy = -log_prob

        # 选择两个 Q 值中的较小值，符合双 Q 学习的策略，避免过高估计
        q1_value = self.target_critic1(next_states, next_actions)
        q2_value = self.target_critic2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy

        # 计算目标 TD 值：reward + γ * (未来状态的值) * (1 - done)
        # td_target = rewards + self.gamma * next_value * (1 - dones)
        td_target = rewards + self.gamma * next_value

        return td_target

    # theta_target  <--- (1 - tau) * theta_target + tau * theta_main
    def _soft_update(self, net, target_net):
        # 使用软更新公式更新目标网络
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    # TODO 1.归一化    2.td_target    4.penalty    5. dop数值
    def update(self, transition_dict):
        # 将数据转换为张量  view(-1, 1) 用于将张量调整为 (batch_size, 1) 的形状
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float32).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float32).to(self.device)
        # dones = torch.tensor(transition_dict['dones'],
        #                      dtype=torch.float32).view(-1, 1).to(self.device)

        # 计算目标Q值(td_target) 和 预测Q值 之间的损失  然后根据损失来更新网络参数
        # td_target = self.calc_target(rewards, next_states, dones)
        td_target = self.calc_target(rewards, next_states)

        current_q1 = self.critic1(states, actions)   # 计算当前状态和动作对应的 预测 Q1 值
        current_q2 = self.critic2(states, actions)   # 预测 Q2 值 (batch_size, 1)
        # mse 均方误差   torch.mean 求平均损失值
        critic1_loss = torch.mean(
            F.mse_loss(current_q1, td_target.detach())
        )    # Q1 损失
        critic2_loss = torch.mean(
            F.mse_loss(current_q2, td_target.detach())
        )    # Q2 损失

        # 反向传播并更新 Critic 网络
        """
        标量张量仍然与图中的参数关联 因此可以调用 backward()
        critic2_loss 是torch.mean计算的标量值（一个单一的标量损失）尽管是标量值 它仍然可以用于反向传播
        这是因为在 PyTorch 中，标量张量会默认与计算图中的参数关联，可以触发网络的梯度计算。
        """
        self.critic1_optimizer.zero_grad()   # 在反向传播前清除 网络 的梯度缓存，避免累积梯度影响更新
        critic1_loss.backward()
        self.critic1_optimizer.step()  # 使用计算得到的梯度更新 critic_1 的参数
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新 Actor 网络
        new_actions, log_prob = self.actor(states)    # 新动作
        entropy = -log_prob
        # new_actions = torch.tensor([new_actions], dtype=torch.float32)
        q1_value = self.critic1(states, new_actions)
        q2_value = self.critic2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()                                  # 反向传播
        self.actor_optimizer.step()                            # 更新策略网络参数

        # 更新 alpha 值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
