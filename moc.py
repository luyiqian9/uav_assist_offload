from envs.uav_env import UAVEnv
from agent import SACAgent

import torch

# 环境和智能体初始化
env = UAVEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = torch.tensor(env.action_space.high, dtype=torch.float32)
print("action_bound = ", end="")
print(action_bound)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = SACAgent(state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, device=device)

# 主训练循环
episodes = 1000
for episode in range(episodes):
    state, addition = env.reset()
    print("state = ", end="")
    print(state)
    episode_reward = 0

    time_step = 15
    for t in range(time_step):
        # 智能体选择动作
        action = agent.select_action(state)
        print("action = ", end="")
        print(action)

        # 环境执行动作，完成连接调度、资源分配和奖励计算
        next_state, reward, _ = env.step(action)
        print(_)

        # 将经验存储到经验池中
        agent.replay_buffer.add((state, action, reward, next_state))
        state = next_state
        episode_reward += reward

        # TODO 若达到mini_batch 则开始更新策略
        agent.update(batch_size=64)

    print(f"Episode {episode}, Reward: {episode_reward}")


""" 
    1.超参数设置
    2.写训练逻辑
    3.绘制训练回报
"""

