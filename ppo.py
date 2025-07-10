import torch
import torch.optim as optim
import torch.nn.functional as F # 缩写该模块
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from env import MahjongGBEnv
from feature import FeatureAgent
from model import ResnetModel
from torch import nn
from collections import defaultdict
import random
import os
import logging
import matplotlib.pyplot as plt
import warnings

# 计算每个时间步的回报和优势
def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    advantages = []
    G = 0
    A = 0

    for i in reversed(range(len(rewards))):
        if dones[i]:
            G = 0
            A = 0

        G = rewards[i] + gamma * G
        returns.insert(0, G)

        if i < len(values) - 1:
            next_value = values[i + 1]
        else:
            next_value = 0

        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        A = delta + gamma * lam * A * (1 - dones[i])
        advantages.insert(0, A)

    return returns, advantages


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])

    return returns, advantages


# PPO 更新过程
def ppo_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
     # 可以增加 batch_size 以获得更稳定的梯度
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {'is_training': True, 'obs': {'observation': state_batch, 'action_mask': action_mask_batch}}
            action_logits, value = policy_network(input_dict)
            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            ###
            # TODO: 在此处补全 policy_loss 和 value_loss 的计算
            # 提示：
            # 1. 计算 `ratio`，即新策略的概率和旧策略概率的比值。
            # 2. 使用 `clip_param` 对 `ratio` 进行剪辑，并计算 PPO 损失函数中的裁剪损失。
            # 3. 计算 policy_loss，这是损失的策略部分。
            # 4. 计算 value_loss，这是预测值函数和实际回报之间的均方误差。
            ###

             ###
            # === PPO 损失计算开始 ===
            ###
            
            # 1. 计算 ratio，即新策略的概率和旧策略概率的比值。
            # PPO 的核心思想是控制新旧策略的差距。我们从 log 空间转换到正常概率空间。
            ratio = torch.exp(log_prob - old_log_prob_batch)

            # 2. 计算第一个代理目标 (surrogate objective)
            surr1 = ratio * advantage_batch

            # 3. 计算第二个代理目标，使用 clip_param (epsilon) 对 ratio 进行裁剪
            # torch.clamp 将 ratio 限制在 [1 - clip_param, 1 + clip_param] 区间内
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantage_batch

            # 4. 计算 policy_loss
            # PPO 的策略损失是两个代理目标中较小值的负数。
            # 取负号是因为优化器执行的是梯度下降（最小化损失），而我们的目标是最大化这个代理目标。
            # 最后取整个 batch 的平均值。
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # 5. 计算 value_loss
            # 这是 Critic 部分的损失，目标是让 V(s) 的预测值尽可能接近实际的回报 G_t。
            # 通常使用均方误差 (MSE) 作为损失函数。
            # value 的形状通常是 [batch_size, 1]，而 return_batch 是 [batch_size]，
            # 所以使用 .squeeze() 来匹配形状。
            value_loss = F.mse_loss(value.squeeze(), return_batch)
            
            ###
            # === PPO 损失计算结束 ===
            ###
            
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()  # 确保没有重复调用 backward
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)


def reinforce_update(policy_network, optimizer, old_log_probs, states, actions, returns, advantages, action_masks,
               clip_param):
    returns = returns.detach()
    advantages = advantages.detach()
    old_log_probs = old_log_probs.detach()

    dataset = TensorDataset(states, actions, returns, old_log_probs, advantages, action_masks)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy_losses = []
    value_losses = []

    for epoch in range(4):
        for batch in dataloader:
            state_batch, action_batch, return_batch, old_log_prob_batch, advantage_batch, action_mask_batch = batch

            input_dict = {'is_training': True, 'obs': {'observation': state_batch, 'action_mask': action_mask_batch}}
            action_logits, value = policy_network(input_dict)
            action_prob = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action_batch)

            policy_loss = -(log_prob * return_batch).mean()
            value_loss = policy_loss

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()  # 确保没有重复调用 backward
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)
