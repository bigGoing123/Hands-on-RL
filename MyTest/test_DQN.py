import gym
import torch
from train_DQN import DQN  # 假设模型类在 train_DQN.py 中
import numpy as np
# 创建 Gym 环境
env = gym.make('CartPole-v0')
lr = 1e-3
num_episodes = 5000
hidden_dim = 128
gamma = 0.98
epsilon = 0.03
target_update = 1000
buffer_size = 3000
minimal_size = 1000
batch_size = 64
# env.seed(42)
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# 创建模型对象
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
agent.q_net.load_state_dict(torch.load('trained_model_DoubleDQN.pth'))
# 加载训练好的模型参数

# 测试模型
state = env.reset()
done = False
for _ in range(10):
    sum_r=0
    env.seed(np.random.randint(1,100))#设置环境随机数
    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()  # 显示环境状态
        sum_r+=reward
    print(sum_r)
    state=env.reset()
    done=False
env.close()