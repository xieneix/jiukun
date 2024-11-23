import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import time
import matplotlib.pyplot as plt
import ray
import matplotlib
matplotlib.use("TkAgg")  # 强制使用 Tkinter 后端
import optuna
from functools import partial
import os
print(torch.cuda.is_available())
# 初始化 Ray
ray.init(num_gpus=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 将 GPU 0 暴露给所有进程
torch.backends.cudnn.benchmark = True  # 加速 GPU 性能（可选）
torch.backends.cudnn.enabled = True  # 确保启用 cuDNN

# 参数定义
MAP_SIZE = 12
NUM_CLASSES = 21  # 符号类别总数为0-20
GAMMA = 0.99  # 折扣因子
LR = 1e-3  # 学习率
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 5000
MAX_STEPS = 4*MAP_SIZE**2  # 每局游戏的最大步数
TEST_EPISODES = 10  # 测试时的游戏局数
REWARD_THRESHOLD = 50  # 用于筛选高质量片段的奖励阈值

# 环境类定义
class Environment:
    def __init__(self, map_size=MAP_SIZE, num_classes=NUM_CLASSES):
        self.map_size = map_size
        self.num_classes = num_classes
        self.max_steps = 4 * map_size * map_size
        self.reset()

    def reset(self):
        # 随机生成符合规则的地图
        self.map = self._generate_map()
        self.agent_pos = [
            np.random.randint(0, self.map_size),
            np.random.randint(0, self.map_size),
        ]
        self.bag = {i: 0 for i in range(self.num_classes)}  # 背包初始化
        self.steps = 0  # 当前轮次计数
        return self._get_state()

    def _generate_map(self):
        # 计算地图单元格数
        total_cells = self.map_size * self.map_size

        # 每个符号数量为4的倍数
        num_repeats = total_cells // self.num_classes
        remainder = total_cells % self.num_classes

        # 确保符号数量与地图单元格数匹配
        symbols = np.repeat(range(self.num_classes), num_repeats)
        if remainder > 0:
            extra_symbols = np.random.choice(range(self.num_classes), remainder, replace=True)
            symbols = np.concatenate([symbols, extra_symbols])

        np.random.shuffle(symbols)  # 随机打乱
        return symbols.reshape(self.map_size, self.map_size)

    def step(self, action):
        reward = 0
        x, y = self.agent_pos

        # 计算当前背包负载总数
        carry = sum(self.bag.values())

        if action == 0 and x > 0:  # 上
            self.agent_pos[0] -= 1
            reward -= (0.1 + carry / (self.map_size**2))
        elif action == 1 and x < self.map_size - 1:  # 下
            self.agent_pos[0] += 1
            reward -= (0.1 + carry / (self.map_size**2))
        elif action == 2 and y > 0:  # 左
            self.agent_pos[1] -= 1
            reward -= (0.1 + carry / (self.map_size**2))
        elif action == 3 and y < self.map_size - 1:  # 右
            self.agent_pos[1] += 1
            reward -= (0.1 + carry / (self.map_size**2))
        elif action == 4:  # 收集
            symbol = self.map[x, y]
            if symbol == -1:  # 已经收集过
                reward -= 200  # 重复收集的惩罚
            else:
                reward -= (0.1 + carry / (self.map_size ** 2))  # 收集消耗
                self.bag[symbol] += 1  # 更新背包
                self.map[x, y] = -1  # 标记为已收集
                # 检查是否触发消除
                if self.bag[symbol] == 4:  # 背包中有4个相同符号
                    reward += 1  # 消除加分
                    self.bag[symbol] = 0  # 清除背包中的符号
        else:
            reward -= 200  # 无效动作惩罚

        self.steps += 1

        # 检查是否完成游戏
        done = self._is_done()

        if done:
            # 地图完全清除奖励
            reward += 1000
        elif self.steps >= self.max_steps:
            # 超过轮次上限的惩罚
            remaining_symbols = np.sum(self.map != -1)
            remaining_bag = sum(self.bag.values())
            reward -= 3 * (remaining_symbols + remaining_bag)
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        # 返回地图、智能体位置和背包信息
        return (self.map.copy(), tuple(self.agent_pos), dict(self.bag))

    def _is_done(self):
        # 判断是否完成任务（地图清空）
        return np.all(self.map == -1)


# Q网络定义（增加可调网络结构）
class QNetwork(nn.Module):
    def __init__(self, map_size, num_classes, action_dim, cnn_filters=64, fc_units=256):
        super(QNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(cnn_filters, cnn_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_filters * 2 * (map_size // 4) * (map_size // 4) + 2 + num_classes, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, action_dim),
        )

    def forward(self, map_data, agent_pos, bag_info):
        cnn_features = self.cnn(map_data).view(map_data.size(0), -1)
        agent_pos = agent_pos.float()
        bag_info = bag_info.float()
        combined = torch.cat([cnn_features, agent_pos, bag_info], dim=1)
        return self.fc(combined)


# DQN算法
class DQNAgent:
    def __init__(self, map_size, num_classes, action_dim, cnn_filters=64, fc_units=256):
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

        # 确定设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化网络到指定设备
        self.q_network = QNetwork(map_size, num_classes, action_dim, cnn_filters, fc_units).to(self.device)
        self.target_network = QNetwork(map_size, num_classes, action_dim, cnn_filters, fc_units).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.update_target()  # 初始化目标网络的权重
        self.high_quality_episodes = []

    def update_target(self):
        # 同步 q_network 的权重到 target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(self.device)  # 确保 target_network 移动到正确设备

    def store_experience(self, experience, reward_threshold=REWARD_THRESHOLD):
        self.memory.append(experience)
        # 提取高质量经验
        _, _, reward, _, _ = experience
        if reward > reward_threshold:
            self.high_quality_episodes.append(experience)

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            map_data, agent_pos, bag_info = self._process_state(state)
            with torch.no_grad():
                q_values = self.q_network(map_data, agent_pos, bag_info)
            return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 预处理状态
        map_data, agent_pos, bag_info = self._process_batch(states)
        next_map_data, next_agent_pos, next_bag_info = self._process_batch(next_states)

        # 当前 Q 值
        q_values = self.q_network(map_data, agent_pos, bag_info)
        q_values = q_values.gather(1, torch.tensor(actions, dtype=torch.long).unsqueeze(1).cuda()).squeeze(1)

        # 目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_network(next_map_data, next_agent_pos, next_bag_info).max(1)[0]
            target_q_values = torch.tensor(rewards, dtype=torch.float32).cuda() + (
                    GAMMA * next_q_values * (1 - torch.tensor(dones, dtype=torch.float32).cuda())
            )

        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def _process_batch(self, states):
        # 使用 numpy array 更高效，确保 map_data 的形状为 [batch_size, map_size, map_size]
        map_data = np.array([s[0] for s in states])
        map_data = torch.tensor(map_data, dtype=torch.float32).unsqueeze(1).cuda()  # 变成 [batch_size, 1, map_size, map_size] 并转换为 float

        agent_pos = torch.tensor([s[1] for s in states], dtype=torch.float32).cuda()  # [batch_size, 2] 并转换为 float
        bag_info = torch.tensor([list(s[2].values()) for s in states], dtype=torch.float32).cuda()  # [batch_size, num_classes] 并转换为 float

        return map_data, agent_pos, bag_info

    def _process_state(self, state):
        # 使用 numpy array 更高效，确保 map_data 的形状为 [1, map_size, map_size]
        map_data = np.array(state[0])
        map_data = torch.tensor(map_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # 变成 [1, 1, map_size, map_size] 并转换为 float

        agent_pos = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).cuda()  # [1, 2] 并转换为 float
        bag_info = torch.tensor(list(state[2].values()), dtype=torch.float32).unsqueeze(0).cuda()  # [1, num_classes] 并转换为 float

        return map_data, agent_pos, bag_info

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        # 动态选择设备（GPU 或 CPU）
        map_location = self.device if hasattr(self, 'device') else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载权重到 q_network，使用 map_location 确保在正确设备上
        self.q_network.load_state_dict(torch.load(path, map_location=map_location))

        # 切换 q_network 到正确设备并设置为评估模式
        self.q_network.to(map_location)
        self.q_network.eval()

        # 同步 target_network 的权重并确保它也在正确设备上
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(map_location)  # 确保 target_network 也在正确设备上

    def save_high_quality_episodes(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.high_quality_episodes, f)


# PPO Actor-Critic网络
class PPOActorCritic(nn.Module):
    def __init__(self, map_size, num_classes, action_dim, cnn_filters=64, fc_units=256):
        super(PPOActorCritic, self).__init__()
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(cnn_filters, cnn_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # 动作选择头
        self.actor = nn.Sequential(
            nn.Linear(cnn_filters * 2 * (map_size // 4) * (map_size // 4) + 2 + num_classes, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, action_dim),
            nn.Softmax(dim=-1),
        )
        # 状态值预测头
        self.critic = nn.Sequential(
            nn.Linear(cnn_filters * 2 * (map_size // 4) * (map_size // 4) + 2 + num_classes, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 1),
        )

    def forward(self, map_data, agent_pos, bag_info):
        # 通过 CNN 提取特征
        cnn_features = self.cnn(map_data).view(map_data.size(0), -1)  # [batch_size, cnn_output_size]

        # 将 CNN 特征与 agent_pos 和 bag_info 合并
        combined = torch.cat([cnn_features, agent_pos, bag_info], dim=1)  # [batch_size, flattened_feature_size]

        # 通过 critic 和 actor 头
        actor_output = self.actor(combined)
        critic_output = self.critic(combined)

        # 打印输入和输出的形状
        # print(f"Critic input shape: {combined.shape}")  # 应该是 [batch_size, flattened_feature_size]
        # print(f"Critic output shape: {critic_output.shape}")  # 应该是 [batch_size, 1]

        return actor_output, critic_output

# PPO算法
class PPOAgent:
    def __init__(self, map_size, num_classes, action_dim, cnn_filters=64, fc_units=256, lr=3e-4):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 使用 PPOActorCritic 统一网络结构
        self.model = PPOActorCritic(map_size, num_classes, action_dim, cnn_filters, fc_units).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # PPO 超参数
        self.eps_clip = 0.2
        self.gamma = GAMMA
        self.lmbda = 0.95
        self.entropy_beta = 0.01
        self.ppo_epochs = 4  # 设置 PPO 的训练轮数

    def select_action(self, state):
        map_data, agent_pos, bag_info = self._process_state(state)

        # CNN 提取特征
        cnn_features = self.model.cnn(map_data).view(map_data.size(0), -1)
        combined_input = torch.cat([cnn_features, agent_pos, bag_info], dim=1)

        # 动作策略和状态值
        logits, value = self.model.actor(combined_input), self.model.critic(combined_input)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        # 确保 value 是二维张量
        if len(value.shape) == 1:
            value = value.unsqueeze(-1)

        # 返回动作、对应的 log_prob 和状态值
        return action.item(), dist.log_prob(action).unsqueeze(0), value

    def evaluate_action(self, state, action):
        """
        计算给定状态和动作的 log_prob、entropy 和 value。
        """
        map_data, agent_pos, bag_info = self._process_state(state)
        prob, value = self.model(map_data, agent_pos, bag_info)
        action_dist = torch.distributions.Categorical(prob)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return log_prob, entropy, value

    def train(self, trajectory):
        # 解包轨迹
        states, actions, rewards, log_probs_old, values, dones = zip(*[
            (self._process_state(s), a, r, lp, v, d) for (s, a, r, lp, v, d) in trajectory
        ])

        # 打包 states 并检查形状
        states = [torch.cat(s, dim=1) for s in zip(*states)]  # 打包 map_data, agent_pos, bag_info
        # print(f"States content: {states}")
        # print(f"Length of states: {len(states)}")

        actions = torch.tensor(actions[:-1], dtype=torch.long).to(self.device)  # 忽略最后一步动作
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        log_probs_old = torch.stack(log_probs_old[:-1]).to(self.device)  # 忽略最后一步 log_prob
        values = torch.cat([v for v in values], dim=0).squeeze(-1)  # 保留所有 values
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 计算 returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).unsqueeze(-1)

        # 计算优势函数
        advantages = returns - values

        # 训练 PPO
        for _ in range(self.ppo_epochs):
            # 解包 states
            map_data, agent_pos, bag_info = states

            # 如果已经是单个张量，不需要 torch.cat()
            if isinstance(map_data, torch.Tensor):
                print("map_data is already a Tensor, skipping cat()")
            else:
                map_data = torch.cat(map_data, dim=0)  # 如果是列表，才进行拼接

            if isinstance(agent_pos, torch.Tensor):
                print("agent_pos is already a Tensor, skipping cat()")
            else:
                agent_pos = torch.cat(agent_pos, dim=0)

            if isinstance(bag_info, torch.Tensor):
                print("bag_info is already a Tensor, skipping cat()")
            else:
                bag_info = torch.cat(bag_info, dim=0)

            # 调试打印
            print(f"map_data shape: {map_data.shape}")
            print(f"agent_pos shape: {agent_pos.shape}")
            print(f"bag_info shape: {bag_info.shape}")

            # 传入模型
            logits, new_values = self.model(map_data, agent_pos, bag_info)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            # 计算 ratio
            ratios = torch.exp(new_log_probs - log_probs_old)

            # PPO 损失函数
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = nn.MSELoss()(new_values.squeeze(-1), returns[:-1])  # 不包括最后一步
            loss = policy_loss + 0.5 * value_loss

            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _process_state(self, state):
        """
        处理环境状态为 PyTorch 张量。
        """
        map_data = np.array(state[0])
        map_data = torch.tensor(map_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        agent_pos = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
        bag_info = torch.tensor(list(state[2].values()), dtype=torch.float32).unsqueeze(0).to(self.device)

        return map_data, agent_pos, bag_info

    def _process_batch(self, states):
        """
        批量处理环境状态为 PyTorch 张量。
        """
        map_data = np.array([s[0] for s in states])
        map_data = torch.tensor(map_data, dtype=torch.float32).unsqueeze(1).to(self.device)

        agent_pos = torch.tensor([s[1] for s in states], dtype=torch.float32).to(self.device)
        bag_info = torch.tensor([list(s[2].values()) for s in states], dtype=torch.float32).to(self.device)

        return map_data, agent_pos, bag_info


@ray.remote
class EnvironmentWorker:
    def __init__(self, map_size, num_classes, action_dim, cnn_filters=64, fc_units=256):
        # 初始化环境和模型
        self.env = Environment(map_size, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(map_size, num_classes, action_dim, cnn_filters, fc_units).to(self.device)

    def run_episode(self, agent_weights):
        # 加载传入的智能体权重并确保它在正确设备上
        self.q_network.load_state_dict(agent_weights)
        self.q_network.to(self.device)  # 确保模型在正确设备上

        state = self.env.reset()
        total_reward = 0
        done = False
        episode_experience = []

        for step in range(MAX_STEPS):
            # 使用当前权重选择动作
            action = self._select_action(state)
            next_state, reward, done = self.env.step(action)
            episode_experience.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break

        return total_reward, episode_experience

    def _select_action(self, state):
        # 根据当前状态和模型权重选择动作
        self.q_network.eval()  # 设置为评估模式
        map_data, agent_pos, bag_info = self._process_state(state)
        with torch.no_grad():
            q_values = self.q_network(map_data, agent_pos, bag_info)
        return torch.argmax(q_values).item()

    def _process_state(self, state):
        # 处理状态，转换为 PyTorch 张量
        map_data = np.array(state[0])
        map_data = torch.tensor(map_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        agent_pos = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
        bag_info = torch.tensor(list(state[2].values()), dtype=torch.float32).unsqueeze(0).to(self.device)

        return map_data, agent_pos, bag_info


def search_hyperparameters():
    # 目标函数定义
    def objective(trial, num_workers=4, num_episodes=100):
        # 采样超参数
        gamma = trial.suggest_float("gamma", 0.90, 0.99)
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        epsilon_decay = trial.suggest_float("epsilon_decay", 0.990, 0.999)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # 初始化智能体
        agent = DQNAgent(MAP_SIZE, NUM_CLASSES, 5, cnn_filters=128, fc_units=512)
        agent.q_network.load_state_dict(agent.q_network.state_dict())
        agent.optimizer = optim.Adam(agent.q_network.parameters(), lr=lr)
        agent.epsilon_decay = epsilon_decay
        agent.gamma = gamma

        # 初始化环境 Workers
        workers = [EnvironmentWorker.remote(MAP_SIZE, NUM_CLASSES, 5, cnn_filters=128, fc_units=512) for _ in
                   range(num_workers)]

        total_rewards = []

        for episode in range(num_episodes):
            # 广播权重给所有 Worker
            agent_weights = agent.q_network.state_dict()
            tasks = [worker.run_episode.remote(agent_weights) for worker in workers]

            # 收集所有 Worker 的结果
            rewards = ray.get(tasks)
            total_rewards.extend(rewards)

            # 模拟训练更新
            for reward in rewards:
                agent.store_experience((None, None, reward, None, None))  # 简化经验示例
                agent.train()

            if episode % TARGET_UPDATE == 0:
                agent.update_target()

        # 计算平均奖励作为目标值
        average_reward = sum(total_rewards) / len(total_rewards)
        return average_reward

    # 创建 Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, num_workers=1, num_episodes=100), n_trials=20)

    # 输出最佳超参数
    print("Best Hyperparameters:", study.best_params)

# 主训练函数
def train_dqn_with_ray():
    # 初始化 DQN 代理
    agent = DQNAgent(MAP_SIZE, NUM_CLASSES, 5, cnn_filters=128, fc_units=512)

    # 创建 Ray 的 environment workers
    action_dim = 5  # 动作空间大小（例如，上下左右收集）
    num_workers = 4
    # 创建多个 Ray Worker
    workers = [
        EnvironmentWorker.remote(MAP_SIZE, NUM_CLASSES, action_dim, cnn_filters=128, fc_units=512)
        for _ in range(num_workers)
    ]

    reward_history = []
    smoothed_rewards = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label="Total Reward per Episode", color='blue', alpha=0.5)
    line2, = ax.plot([], [], label="Smoothed Reward (100-episode average)", color='red', linewidth=2)
    ax.set_xlim(0, NUM_EPISODES)
    ax.set_ylim(-1000, 200)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward Trend (Real-time)")
    ax.legend()
    ax.grid()

    for episode in range(NUM_EPISODES):
        # 获取当前智能体的权重
        agent_weights = agent.q_network.state_dict()

        # 分发任务到 workers
        tasks = [worker.run_episode.remote(agent_weights) for worker in workers]
        results = ray.get(tasks)

        # 收集经验和奖励
        for total_reward, episode_experience in results:
            reward_history.append(total_reward)
            for experience in episode_experience:
                agent.store_experience(experience)
            agent.train()

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        # 计算滑动平均
        if len(reward_history) >= 100:
            smoothed_rewards.append(np.mean(reward_history[-100:]))
        else:
            smoothed_rewards.append(np.mean(reward_history))

        # 更新实时图像
        line1.set_data(range(len(reward_history)), reward_history)
        line2.set_data(range(len(smoothed_rewards)), smoothed_rewards)
        ax.set_xlim(0, max(100, len(reward_history)))
        ax.set_ylim(min(reward_history) - 50, max(reward_history) + 50)
        plt.pause(0.01)

        print(f"Episode {episode}, Average Reward: {np.mean(reward_history[-100:])}")

    plt.ioff()
    plt.show()

    # 保存模型
    agent.save_model("dqn_agent_ray.pth")
    print("Training completed and model saved.")


def train_dqn_single_thread():
    # 初始化 DQN 代理
    agent = DQNAgent(MAP_SIZE, NUM_CLASSES, 5, cnn_filters=128, fc_units=512)
    env = Environment(MAP_SIZE, NUM_CLASSES)  # 单线程只需一个环境实例

    reward_history = []
    smoothed_rewards = []

    # 设置实时绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label="Total Reward per Episode", color='blue', alpha=0.5)
    line2, = ax.plot([], [], label="Smoothed Reward (100-episode average)", color='red', linewidth=2)
    ax.set_xlim(0, NUM_EPISODES)
    ax.set_ylim(-1000, 200)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward Trend (Real-time)")
    ax.legend()
    ax.grid()

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        # 收集经验和进行训练
        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_experience((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        reward_history.append(total_reward)

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        # 计算滑动平均
        if len(reward_history) >= 100:
            smoothed_rewards.append(np.mean(reward_history[-100:]))
        else:
            smoothed_rewards.append(np.mean(reward_history))

        # 更新实时图像
        line1.set_data(range(len(reward_history)), reward_history)
        line2.set_data(range(len(smoothed_rewards)), smoothed_rewards)
        ax.set_xlim(0, max(100, len(reward_history)))
        ax.set_ylim(min(reward_history) - 50, max(reward_history) + 50)
        plt.pause(0.01)

        print(f"Episode {episode}, Total Reward: {total_reward}, Average Reward (Last 100): {np.mean(reward_history[-100:])}")

    plt.ioff()
    plt.show()

    # 保存模型
    agent.save_model("dqn_agent_single.pth")
    print("Training completed and model saved.")


def train_ppo_single_thread():
    # 环境和智能体初始化
    agent = PPOAgent(MAP_SIZE, NUM_CLASSES, 5, cnn_filters=128, fc_units=512, lr=3e-4)
    env = Environment(MAP_SIZE, NUM_CLASSES)

    reward_history = []
    smoothed_rewards = []

    # 设置实时绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line1, = ax.plot([], [], label="Total Reward per Episode", color='blue', alpha=0.5)
    line2, = ax.plot([], [], label="Smoothed Reward (100-episode average)", color='red', linewidth=2)
    ax.set_xlim(0, NUM_EPISODES)
    ax.set_ylim(-1000, 200)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward Trend (Real-time)")
    ax.legend()
    ax.grid()

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        trajectory = []

        for step in range(MAX_STEPS):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward, log_prob, value, done))
            state = next_state
            total_reward += reward
            if done:
                break

        # 添加最后一步状态值
        map_data, agent_pos, bag_info = agent._process_state(state)
        cnn_features = agent.model.cnn(map_data).view(map_data.size(0), -1)
        combined_input = torch.cat([cnn_features, agent_pos, bag_info], dim=1)

        # 计算 critic 的值
        value = agent.model.critic(combined_input)

        # 确保最后一步的 value 是二维张量，形状为 [1, 1]
        if len(value.shape) == 1:
            value = value.unsqueeze(0)  # 添加 batch 维度

        trajectory.append((state, None, -1, None, value, True))

        # 调试信息
        print(f"Trajectory Length: {len(trajectory)}")
        for idx, (s, a, r, lp, v, d) in enumerate(trajectory):
            print(f"Step {idx + 1}:")
            print(f"  State: {s}")
            print(f"  Action: {a}")
            print(f"  Reward: {r}")
            print(f"  Log Prob: {lp}")
            print(f"  Value: {v}")
            print(f"  Done: {d}")

        # 更新 PPO
        agent.train(trajectory)

        reward_history.append(total_reward)

        # 计算滑动平均
        if len(reward_history) >= 100:
            smoothed_rewards.append(np.mean(reward_history[-100:]))
        else:
            smoothed_rewards.append(np.mean(reward_history))

        # 更新实时图像
        line1.set_data(range(len(reward_history)), reward_history)
        line2.set_data(range(len(smoothed_rewards)), smoothed_rewards)
        ax.set_xlim(0, max(100, len(reward_history)))
        ax.set_ylim(min(reward_history) - 50, max(reward_history) + 50)
        plt.pause(0.01)

        print(f"Episode {episode}, Total Reward: {total_reward}, Average Reward (Last 100): {np.mean(reward_history[-100:])}")

    plt.ioff()
    plt.show()

    # 保存模型
    torch.save(agent.model.state_dict(), "ppo_actor_critic.pth")
    print("Training completed and model saved.")


def test_agent(agent, env, num_episodes=10, max_steps=4 * MAP_SIZE**2, render=False):
    """
    测试智能体在给定环境中的表现。
    :param agent: 训练好的智能体
    :param env: 测试环境
    :param num_episodes: 测试的游戏局数
    :param max_steps: 每局游戏的最大步数
    :param render: 是否渲染环境（如果实现了环境渲染功能，可选）
    :return: 平均奖励和每局的奖励记录
    """
    print("\nTesting the trained PPO agent...\n")

    test_scores = []
    all_steps = []

    for test_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            if render:  # 如果需要渲染环境
                env.render()

            # 使用 PPO 策略选择动作
            action, _, _ = agent.select_action(state)
            state, reward, done = env.step(action)
            total_reward += reward

            if done:
                break

        test_scores.append(total_reward)
        all_steps.append(step + 1)  # 记录实际步数

        print(f"Test Episode {test_episode + 1}, Total Reward: {total_reward}, Steps Taken: {step + 1}")

    # 计算平均奖励和步数
    average_score = sum(test_scores) / len(test_scores)
    average_steps = sum(all_steps) / len(all_steps)

    print(f"\nAverage Score over {num_episodes} Test Episodes: {average_score}")
    print(f"Average Steps per Episode: {average_steps}")

    return average_score, test_scores





if __name__ == "__main__":
    # 训练 PPO
    train_ppo_single_thread()

    # 初始化环境
    env = Environment(map_size=MAP_SIZE, num_classes=NUM_CLASSES)

    # 初始化 PPO 智能体
    agent = PPOAgent(MAP_SIZE, NUM_CLASSES, 5, cnn_filters=128, fc_units=512)

    # 加载训练好的模型
    agent.model.load_state_dict(torch.load("ppo_actor_critic.pth"))
    agent.model.eval()  # 设置为评估模式

    # 测试智能体
    average_score, test_scores = test_agent(
        agent=agent,
        env=env,
        num_episodes=TEST_EPISODES,
        max_steps=MAX_STEPS,
        render=False
    )

    # 绘制测试奖励分布
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_scores) + 1), test_scores, marker="o", label="Reward per Episode")
    plt.axhline(average_score, color="red", linestyle="--", label=f"Average Reward ({average_score:.2f})")
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.title("Test Performance of PPO Agent")
    plt.legend()
    plt.grid()
    plt.show()

