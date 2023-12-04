import gym
import torch
from train import DQN  # 假设模型类在 train.py 中

# 创建 Gym 环境
env = gym.make('CartPole-v0')
hidden_dim = 128
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
env.seed(0)
torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# 创建模型对象
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

# 加载训练好的模型参数
loaded_state_dict = torch.load('trained_model.pth')
agent.q_net.load_state_dict(loaded_state_dict)

# 测试模型
state = env.reset()
done = False
for _ in range(10):
    sum_r=0
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