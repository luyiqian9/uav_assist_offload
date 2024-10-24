import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出动作


# Critic 网络定义（用于值函数估计）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)  # Q值输出

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # 将状态和动作拼接
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 经验回放池
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, experience):
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states))


# SAC 智能体定义
class SACAgent:
    def __init__(self, state_dim, action_dim, buffer_size=100000):
        # 初始化 Actor 和 Critic 网络
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)

        # 目标网络
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        return action

    def update(self, batch_size, gamma=0.99, tau=0.005):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        # 从经验池中采样
        states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)

        # 转为Tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            yj = rewards + gamma * target_q

        # 更新Critic网络
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss1 = nn.MSELoss()(current_q1, yj)
        critic_loss2 = nn.MSELoss()(current_q2, yj)

        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        # 更新Actor网络
        new_actions = self.actor(states)
        actor_loss = -self.critic1(states, new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1, tau)
        self._soft_update(self.target_critic2, self.critic2, tau)

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

