# Model part
import torch
from torch import nn
import numpy as np


class BottleNeck(nn.Module):

    def __init__(self, c):
        nn.Module.__init__(self)
        self._conv = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c // 4, 1, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(True),
            nn.Conv2d(c // 4, c // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(True),
            nn.Conv2d(c // 4, c, 1, bias=False)
        )

    def forward(self, x):
        out = self._conv(x)
        return out + x


class ResnetModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        super(ResnetModel, self).__init__()
        self._convs = nn.Sequential(
            nn.Conv2d(174, 256, 1, bias=False),
            *(BottleNeck(256) for i in range(18)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        self._hidden = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_mask = input_dict["obs"]["action_mask"].float()
        self._hidden = self._convs(obs)
        action_logits = self._logits(self._hidden)
        value = self._value_branch(self._hidden)
        action_mask = action_mask.float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask, value

    def value_function(self):
        assert self._hidden is not None, "must call forward() first"
        return torch.squeeze(self._value_branch(self._hidden), 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.in_channels = 64
        self._tower = nn.Sequential(
            nn.Conv2d(147, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer1 = self._make_layer(BasicBlock, 64, 9)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 235)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_logits = self._tower(obs)
        action_logits = self.layer1(action_logits)
        action_logits = self.flatten(action_logits)
        action_logits = self.fc1(action_logits)
        action_logits = self.relu(action_logits)
        action_logits = self.fc2(action_logits)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask


class RandomPolicy(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_dict):
        # return [np.random.choice(np.flatnonzero(obs)) for obs in obs_batch['action_mask']], [], {}
        return [np.random.choice(np.flatnonzero(obs['action_mask'])) for obs in input_dict]
