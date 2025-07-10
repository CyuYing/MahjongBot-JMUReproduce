import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from env import MahjongGBEnv_Dense_Reward as MahjongGBEnv
# from env import MahjongGBEnv
from feature import FeatureAgent
from model import ResnetModel
from torch import nn
from collections import defaultdict, deque
import random
import os
import logging
import matplotlib.pyplot as plt
import warnings
from replay_buffer import PrioritizedReplayBuffer
from ppo import compute_returns_and_advantages, compute_gae, ppo_update

warnings.filterwarnings("ignore", category=FutureWarning)

# 设置随机种子
seed = 39
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 日志设置
logging.basicConfig(filename='expert_opponent_training_log.log', level=logging.INFO)

# PPO 超参数
GAMMA = 0.99
LAMBDA = 0.95
CLIP_PARAM = 0.2
NUM_EPOCHS = 5000
BATCH_SIZE = 2048
LEARNING_RATE = 3e-4
TIMESTEPS_PER_BATCH = 4096
MODEL_SAVE_PATH = './models/ppo_expert_opponents/'

# 创建模型保存目录
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# 初始化环境
config = {
    'agent_clz': FeatureAgent,
    'reward_norm': True,
    'variety': -1,
    'duplicate': True
}
env = MahjongGBEnv(config)

# 初始化策略网络，并将其转移到GPU
obs_space_shape = env.observation_space.shape
num_actions = env.action_space
policy_network = ResnetModel().to(device)
policy_network.train()
optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
replay_buffer = PrioritizedReplayBuffer(capacity=200)

# 加载预训练模型参数
expert_model_path = 'models/expert.pkl'
expert_network = ResnetModel().to(device)
if os.path.exists(expert_model_path):
    expert_network.load_state_dict(torch.load(expert_model_path))
    expert_network.eval()
    policy_network.load_state_dict(torch.load(expert_model_path))
    print("Loaded expert model parameters from", expert_model_path)
else:
    raise FileNotFoundError("Expert model file not found.")


def collect_trajectories(env, policy_network, timesteps_per_batch=1024):
    trajectories = []
    states = defaultdict(list)
    actions = defaultdict(list)
    rewards = defaultdict(list)
    dones = defaultdict(list)
    log_probs = defaultdict(list)
    values = defaultdict(list)
    action_masks = defaultdict(list)

    obs = env.reset()

    cumulative_rewards = {player_name: 0.0 for player_name in ["player_1", "player_2", "player_3", "player_4"]}
    cumulative_dones = {player_name: False for player_name in ["player_1", "player_2", "player_3", "player_4"]}

    pre_player = None

    for _ in range(timesteps_per_batch):
        actions_dict = {}
        temp_data = {}

        for player_name in obs.keys():
            state_tensor = torch.tensor(obs[player_name]['observation'], dtype=torch.float32).unsqueeze(0).to(device)
            action_mask = torch.tensor(obs[player_name]['action_mask'], dtype=torch.float32).unsqueeze(0).to(device)

            if player_name == "player_1":
                # 主智能体使用策略网络
                input_dict = {'is_training': True, 'obs': {'observation': state_tensor, 'action_mask': action_mask}}
                action_logits, value = policy_network(input_dict)
                action_prob = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_prob)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            else:
                # 其他智能体使用固定的专家模型
                input_dict = {'is_training': False, 'obs': {'observation': state_tensor, 'action_mask': action_mask}}
                action_logits, value = expert_network(input_dict)
                action = action_logits.detach().cpu().numpy().flatten().argmax()
                log_prob = torch.log(torch.tensor(1.0 / len(action_logits), dtype=torch.float32)).to(device)
                value = torch.tensor(0.0).to(device)  # 随机策略无价值函数

            actions_dict[player_name] = action.item()

            temp_data[player_name] = {
                "state": state_tensor,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "action_mask": action_mask
            }

        next_obs, reward, done, _ = env.step(actions_dict)

        # 累积奖励
        for player_name in reward.keys():
            cumulative_rewards[player_name] += reward[player_name]

        # 更新完成标志
        for player_name in done.keys():
            if player_name != '__all__':
                cumulative_dones[player_name] = cumulative_dones[player_name] or done[player_name]

        cur_player = f"player_{env.curPlayer + 1}"

        # 如果 pre_player 存在，记录数据
        if pre_player:
            if len(dones[pre_player]) == len(states[pre_player]):
                rewards[pre_player][-1] = cumulative_rewards[pre_player]
                dones[pre_player][-1] = cumulative_dones[pre_player]
            else:
                rewards[pre_player].append(cumulative_rewards[pre_player])
                dones[pre_player].append(cumulative_dones[pre_player])
            if not done['__all__']:
                cumulative_rewards[pre_player] = 0.0
                cumulative_dones[pre_player] = False

        # 如果 cur_player 是主智能体，并且游戏未结束，记录状态和动作
        if cur_player in temp_data.keys() and not done['__all__']:
            states[cur_player].append(temp_data[cur_player]["state"])
            actions[cur_player].append(temp_data[cur_player]["action"])
            log_probs[cur_player].append(temp_data[cur_player]["log_prob"])
            values[cur_player].append(temp_data[cur_player]["value"])
            action_masks[cur_player].append(temp_data[cur_player]["action_mask"])

        obs = next_obs
        pre_player = cur_player
        if done["__all__"]:
            for player_name in ["player_1", "player_2", "player_3", "player_4"]:
                dones[player_name][-1] = True
                rewards[player_name][-1] = cumulative_rewards[player_name]
            for player_name in ["player_1"]:
                trajectory = []
                for state, action, reward, done, log_prob, value, action_mask in \
                        zip(states[player_name], actions[player_name], rewards[player_name], dones[player_name],
                            log_probs[player_name], values[player_name], action_masks[player_name]):
                    trajectory.append((state.cpu().detach(), action.cpu().detach(), reward, done,
                                       log_prob.cpu().detach(), value.cpu().detach(), action_mask.cpu().detach()))
                trajectories.append(trajectory)
                states[player_name] = []
                actions[player_name] = []
                rewards[player_name] = []
                dones[player_name] = []
                log_probs[player_name] = []
                values[player_name] = []
                action_masks[player_name] = []
            obs = env.reset()
            pre_player = None
            cumulative_rewards = {player_name: 0.0 for player_name in ["player_1", "player_2", "player_3", "player_4"]}
            cumulative_dones = {player_name: False for player_name in ["player_1", "player_2", "player_3", "player_4"]}

    for trajectory in trajectories:
        # 计算优先级 (如基于轨迹总奖励)
        priority = np.abs(np.sum([step[2] for step in trajectory])) + 1e-5
        replay_buffer.add(trajectory, priority)

    return replay_buffer.sample(batch_size=BATCH_SIZE)


# 测评函数：使用一个policy网络与3个加载了expert.pkl的模型对战
def evaluate_model(env, policy_network, num_games=10):
    total_rewards = []

    for _ in range(num_games):
        obs = env.reset()
        done = False
        cumulative_reward = 0

        while not done:
            actions_dict = {}

            for i, player_name in enumerate(obs.keys()):
                state_tensor = torch.tensor(obs[player_name]['observation'], dtype=torch.float32).unsqueeze(0).to(
                    device)
                action_mask = torch.tensor(obs[player_name]['action_mask'], dtype=torch.float32).unsqueeze(0).to(device)
                input_dict = {'is_training': False, 'obs': {'observation': state_tensor, 'action_mask': action_mask}}

                if i == 0:  # 第一位玩家使用当前训练的policy_network
                    action_logits, _ = policy_network(input_dict)
                else:  # 其他玩家使用预训练的expert模型
                    action_logits, _ = expert_network(input_dict)

                action = action_logits.detach().cpu().numpy().flatten().argmax()

                actions_dict[player_name] = action.item()

            obs, reward, done, _ = env.step(actions_dict)

            if "player_1" in reward.keys():
                cumulative_reward += reward["player_1"]  # 记录当前训练模型的累计奖励

            if done['__all__']:
                total_rewards.append(cumulative_reward)

            done = done['__all__']

    sum_reward = np.sum(total_rewards)/num_games
    return sum_reward


# 训练函数
def train(env, policy_network, optimizer, num_epochs=1000):
    policy_loss_list = []
    value_loss_list = []
    evaluation_scores = []
    best_evaluation_score = -float('inf')
    evaluation_epochs = []

    evaluation_score = evaluate_model(env, policy_network, num_games=10)
    evaluation_scores.append(evaluation_score)
    evaluation_epochs.append(0)
    logging.info(f"Epoch {0} evaluation score: {evaluation_score:.2f}")
    print(f"Epoch {0} evaluation score: {evaluation_score:.2f}")

    for epoch in range(num_epochs):
        # 从经验池采样
        samples, indices = collect_trajectories(env, policy_network)

        all_states = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_old_log_probs = []
        all_action_masks = []

        for trajectory in samples:
            states, actions, rewards, dones, log_probs, values, action_masks = zip(*trajectory)

            # returns, advantages = compute_returns_and_advantages(rewards, values, dones)
            returns, advantages = compute_gae(rewards, values, dones)
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)
            all_advantages.extend(advantages)
            all_old_log_probs.extend(log_probs)
            all_action_masks.extend(action_masks)

        states_tensor = torch.cat(all_states, dim=0).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.long).to(device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(device)
        log_probs_tensor = torch.cat(all_old_log_probs, dim=0).to(device)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32).to(device)
        action_masks_tensor = torch.cat(all_action_masks, dim=0).to(device)

        policy_loss, value_loss = ppo_update(policy_network, optimizer, log_probs_tensor, states_tensor, actions_tensor,
                                             returns_tensor, advantages_tensor, action_masks_tensor, CLIP_PARAM)

        policy_loss_list.append(policy_loss)
        value_loss_list.append(value_loss)

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed. | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}")

        # 每20个epoch进行一次模型能力测评
        if (epoch + 1) % 20 == 0:
            evaluation_score = evaluate_model(env, policy_network, num_games=10)
            evaluation_scores.append(evaluation_score)
            evaluation_epochs.append(epoch + 1)
            logging.info(f"Epoch {epoch + 1} evaluation score: {evaluation_score:.2f}")
            print(f"Epoch {epoch + 1} evaluation score: {evaluation_score:.2f}")

            # 如果当前的测评分数超过了历史最高得分，则保存模型
            if evaluation_score > best_evaluation_score:
                best_evaluation_score = evaluation_score
                torch.save(policy_network.state_dict(),
                           os.path.join(MODEL_SAVE_PATH, f"best_policy_network_epoch.pkl"))
                print(f"New best model saved with evaluation score: {evaluation_score:.2f}")

    # 绘制奖励、损失和评估分数曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(policy_loss_list, label='Policy Loss')
    plt.plot(value_loss_list, label='Value Loss')
    plt.title('Losses')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(evaluation_epochs, evaluation_scores)
    plt.title('Evaluation Scores')
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Score')
    # plt.show()
    plt.savefig('training_results_expert_opponents.png')

# 主程序
if __name__ == "__main__":
    train(env, policy_network, optimizer, num_epochs=NUM_EPOCHS)
