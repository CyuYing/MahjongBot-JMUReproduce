import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import random
from agent import MahjongGBAgent

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


class Error(Exception):
    pass


class MahjongGBEnv(gym.Env):
    agent_names = ['player_%d' % i for i in range(1, 5)]

    def __init__(self, config):
        assert 'agent_clz' in config, "must specify agent_clz to process features!"
        self.agentclz = config['agent_clz']
        assert issubclass(self.agentclz, MahjongGBAgent), "ageng_clz must be a subclass of MahjongGBAgent!"
        self.duplicate = config.get('duplicate', True)
        self.variety = config.get('variety', -1)
        self.r = random.Random()
        self.normalizeReward = config.get('reward_norm', False)
        self.observation_space = self.agentclz.observation_space
        self.action_space = self.agentclz.action_space

    def reset(self, prevalentWind=-1, tileWall=''):
        # Create agents to process features
        self.agents = [self.agentclz(i) for i in range(4)]
        self.reward = None
        self.done = False
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        # Init prevalent wind
        self.prevalentWind = random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevalentWind)
        # Prepare tile wall
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append('W' + str(i))
                    self.tileWall.append('B' + str(i))
                    self.tileWall.append('T' + str(i))
                for i in range(1, 5):
                    self.tileWall.append('F' + str(i))
                for i in range(1, 4):
                    self.tileWall.append('J' + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = ' '.join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34: (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        # Deal cards
        self._deal()
        return self._obs()

    def step(self, action_dict):
        try:
            if self.state == 0:
                # After Chi/Peng, prepare to Play
                response = self.agents[self.curPlayer].action2response(
                    action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Play':
                    self._discard(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
                self.isAboutKong = False
            elif self.state == 1:
                # After Draw, prepare to Hu/Play/Gang/BuGang
                response = self.agents[self.curPlayer].action2response(
                    action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Hu':
                    self.shownTiles[self.curTile] += 1
                    self._checkMahjong(self.curPlayer, isSelfDrawn=True, isAboutKong=self.isAboutKong)
                elif response[0] == 'Play':
                    self.hands[self.curPlayer].append(self.curTile)
                    self._discard(self.curPlayer, response[1])
                elif response[0] == 'Gang' and not self.myWallLast and not self.wallLast:
                    self._concealedKong(self.curPlayer, response[1])
                elif response[0] == 'BuGang' and not self.myWallLast and not self.wallLast:
                    self._promoteKong(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
            elif self.state == 2:
                # After Play, prepare to Chi/Peng/Gang/Hu/Pass
                responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if
                             i != self.curPlayer}
                t = {i: responses[i].split() for i in responses}
                # Priority: Hu > Peng/Gang > Chi
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if t[i][0] == 'Hu':
                        self._checkMahjong(i)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if t[i][0] == 'Gang' and self._canDrawTile(i) and not self.wallLast:
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.wallLast:
                            self._pung(i, self.curTile)
                            break
                    else:
                        i = (self.curPlayer + 1) % 4
                        if t[i][0] == 'Chi' and not self.wallLast:
                            self._chow(i, t[i][1])
                        else:
                            for j in range(1, 4):
                                i = (self.curPlayer + j) % 4
                                if t[i][0] != 'Pass': raise Error(i)
                            if self.wallLast:
                                # A draw
                                self.obs = {i: self.agents[i].request2obs('Huang') for i in range(4)}
                                self.reward = [0, 0, 0, 0]
                                self.done = True
                            else:
                                # Next player
                                self.curPlayer = (self.curPlayer + 1) % 4
                                self._draw(self.curPlayer)
            elif self.state == 3:
                # After BuGang, prepare to Hu/Pass
                responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if
                             i != self.curPlayer}
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if responses[i] == 'Hu':
                        self._checkMahjong(i, isAboutKong=True)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] != 'Pass': raise Error(i)
                    self._draw(self.curPlayer)
        except Error as e:
            player = e.args[0]
            self.obs = {i: self.agents[i].request2obs('Player %d Invalid' % player) for i in range(4)}
            self.reward = [10] * 4
            self.reward[player] = -30
            self.done = True
        return self._obs(), self._reward(), self._done(), self._info()

    def _obs(self):
        return {self.agent_names[k]: v for k, v in self.obs.items()}

    def _reward(self):
        if self.reward: return {self.agent_names[k]: self.reward[k] for k in self.obs}
        return {self.agent_names[k]: 0 for k in self.obs}

    def _done(self):
        done = {'__all__': self.done}
        for k in self.obs: done[self.agent_names[k]] = self.done
        return done

    def _info(self):
        return {self.agent_names[k]: {} for k in self.obs}

    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()

    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)

    def _deal(self):
        self.hands = []
        self.packs = []
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            self.hands.append(hand)
            self.packs.append([])
            self.agents[i].request2obs(' '.join(['Deal', *hand]))
        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)

    def _draw(self, player):
        tile = self._drawTile(player)
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player: self.agents[player].request2obs('Draw %s' % tile)}

    def _discard(self, player, tile):
        if tile not in self.hands[player]: raise Error(player)
        self.hands[player].remove(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i: self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(4) if
                    i != player}

    def _kong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs('Player %d Gang' % player)
        self._draw(player)

    def _pung(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('PENG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player: self.agents[player].request2obs('Player %d Peng' % player)}

    def _chow(self, player, tile):
        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]: raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        # offer: 123 for which tile is offered
        self.packs[player].append(('CHI', tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Chi %s' % (player, tile))
        self.obs = {player: self.agents[player].request2obs('Player %d Chi %s' % (player, tile))}

    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d AnGang' % player)
        self.agents[player].request2obs('Player %d AnGang %s' % (player, tile))
        self._draw(player)

    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == 'PENG' and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0: raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ('GANG', tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs('Player %d BuGang %s' % (player, tile))
        self.obs = {i: self.agents[i].request2obs('Player %d BuGang %s' % (player, tile)) for i in range(4) if
                    i != player}

    def _checkMahjong(self, player, isSelfDrawn=False, isAboutKong=False):
        try:
            # print(self.packs[player])
            # print(self.hands[player])
            # print(self.curTile)
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[player]),
                hand=tuple(self.hands[player]),
                winTile=self.curTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=self.shownTiles[self.curTile] == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=player,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Error('Not Enough Fans')
            self.obs = {i: self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}
            if self.normalizeReward:
                self.reward = [-1] * 4
                if isSelfDrawn:
                    self.reward[player] = ((8 + fanCnt) * 3 / 2) ** 0.5
                else:
                    self.reward[player] = ((8 * 3 + fanCnt) / 2) ** 0.5
                    self.reward[self.curPlayer] -= 1
            else:
                if isSelfDrawn:
                    self.reward = [-(8 + fanCnt)] * 4
                    self.reward[player] = (8 + fanCnt) * 3
                else:
                    self.reward = [-8] * 4
                    self.reward[player] = 8 * 3 + fanCnt
                    self.reward[self.curPlayer] -= fanCnt
            self.done = True
        except Exception as e:
            raise Error(player)


class MahjongGBEnv_Dense_Reward(gym.Env):
    agent_names = ['player_%d' % i for i in range(1, 5)]

    def __init__(self, config):
        assert 'agent_clz' in config, "must specify agent_clz to process features!"
        self.agentclz = config['agent_clz']
        assert issubclass(self.agentclz, MahjongGBAgent), "ageng_clz must be a subclass of MahjongGBAgent!"
        self.duplicate = config.get('duplicate', True)
        self.variety = config.get('variety', -1)
        self.r = random.Random()
        self.normalizeReward = config.get('reward_norm', False)
        self.observation_space = self.agentclz.observation_space
        self.action_space = self.agentclz.action_space

    def reset(self, prevalentWind=-1, tileWall=''):
        # Create agents to process features
        self.agents = [self.agentclz(i) for i in range(4)]
        self.reward = None
        self.done = False
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        # Init prevalent wind
        self.prevalentWind = random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevalentWind)
        # Prepare tile wall
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append('W' + str(i))
                    self.tileWall.append('B' + str(i))
                    self.tileWall.append('T' + str(i))
                for i in range(1, 5):
                    self.tileWall.append('F' + str(i))
                for i in range(1, 4):
                    self.tileWall.append('J' + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = ' '.join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34: (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        # Deal cards
        self._deal()
        return self._obs()

    def step(self, action_dict):
        try:
            self.reward = [0, 0, 0, 0]
            if self.state == 0:
                # After Chi/Peng, prepare to Play
                response = self.agents[self.curPlayer].action2response(
                    action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Play':
                    self._discard(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
                self.isAboutKong = False
            elif self.state == 1:
                # After Draw, prepare to Hu/Play/Gang/BuGang
                response = self.agents[self.curPlayer].action2response(
                    action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Hu':
                    self.shownTiles[self.curTile] += 1
                    self._checkMahjong(self.curPlayer, isSelfDrawn=True, isAboutKong=self.isAboutKong)
                elif response[0] == 'Play':
                    self.hands[self.curPlayer].append(self.curTile)
                    self._discard(self.curPlayer, response[1])
                elif response[0] == 'Gang' and not self.myWallLast and not self.wallLast:
                    self._concealedKong(self.curPlayer, response[1])
                elif response[0] == 'BuGang' and not self.myWallLast and not self.wallLast:
                    self._promoteKong(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
            elif self.state == 2:
                # After Play, prepare to Chi/Peng/Gang/Hu/Pass
                responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if
                             i != self.curPlayer}
                t = {i: responses[i].split() for i in responses}
                # Priority: Hu > Peng/Gang > Chi
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if t[i][0] == 'Hu':
                        self._checkMahjong(i)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if t[i][0] == 'Gang' and self._canDrawTile(i) and not self.wallLast:
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.wallLast:
                            self._pung(i, self.curTile)
                            break
                    else:
                        i = (self.curPlayer + 1) % 4
                        if t[i][0] == 'Chi' and not self.wallLast:
                            self._chow(i, t[i][1])
                        else:
                            for j in range(1, 4):
                                i = (self.curPlayer + j) % 4
                                if t[i][0] != 'Pass': raise Error(i)
                            if self.wallLast:
                                # A draw
                                self.obs = {i: self.agents[i].request2obs('Huang') for i in range(4)}
                                self.reward = [0, 0, 0, 0]
                                self.done = True
                            else:
                                # Next player
                                self.curPlayer = (self.curPlayer + 1) % 4
                                self._draw(self.curPlayer)
            elif self.state == 3:
                # After BuGang, prepare to Hu/Pass
                responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if
                             i != self.curPlayer}
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if responses[i] == 'Hu':
                        self._checkMahjong(i, isAboutKong=True)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] != 'Pass': raise Error(i)
                    self._draw(self.curPlayer)
        except Error as e:
            player = e.args[0]
            self.obs = {i: self.agents[i].request2obs('Player %d Invalid' % player) for i in range(4)}
            self.reward = [10] * 4
            self.reward[player] = -30
            self.done = True
        return self._obs(), self._reward(), self._done(), self._info()

    def _obs(self):
        return {self.agent_names[k]: v for k, v in self.obs.items()}

    def _reward(self):
        if self.reward: return {self.agent_names[k]: self.reward[k] for k in self.obs}
        return {self.agent_names[k]: 0 for k in self.obs}

    def _done(self):
        done = {'__all__': self.done}
        for k in self.obs: done[self.agent_names[k]] = self.done
        return done

    def _info(self):
        return {self.agent_names[k]: {} for k in self.obs}

    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()

    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)

    def _deal(self):
        self.hands = []
        self.packs = []
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            self.hands.append(hand)
            self.packs.append([])
            self.agents[i].request2obs(' '.join(['Deal', *hand]))
        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)

    def _draw(self, player):
        tile = self._drawTile(player)
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player: self.agents[player].request2obs('Draw %s' % tile)}

    def _discard(self, player, tile):
        if tile not in self.hands[player]: raise Error(player)
        self.hands[player].remove(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i: self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(4) if
                    i != player}

    def _kong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs('Player %d Gang' % player)
        self._draw(player)
        self.reward[player] = 0.2

    def _pung(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('PENG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player: self.agents[player].request2obs('Player %d Peng' % player)}
        self.reward[player] = 0.1

    def _chow(self, player, tile):
        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]: raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        # offer: 123 for which tile is offered
        self.packs[player].append(('CHI', tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Chi %s' % (player, tile))
        self.obs = {player: self.agents[player].request2obs('Player %d Chi %s' % (player, tile))}
        self.reward[player] = 0.1

    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d AnGang' % player)
        self.agents[player].request2obs('Player %d AnGang %s' % (player, tile))
        self._draw(player)
        self.reward[player] = 0.2

    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == 'PENG' and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0: raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ('GANG', tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs('Player %d BuGang %s' % (player, tile))
        self.obs = {i: self.agents[i].request2obs('Player %d BuGang %s' % (player, tile)) for i in range(4) if
                    i != player}
        self.reward[player] = 0.2

    def _checkMahjong(self, player, isSelfDrawn=False, isAboutKong=False):
        try:
            # print(self.packs[player])
            # print(self.hands[player])
            # print(self.curTile)
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[player]),
                hand=tuple(self.hands[player]),
                winTile=self.curTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=self.shownTiles[self.curTile] == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=player,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Error('Not Enough Fans')
            self.obs = {i: self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}
            if self.normalizeReward:
                self.reward = [-1] * 4
                if isSelfDrawn:
                    self.reward[player] = ((8 + fanCnt) * 3 / 2) ** 0.5
                else:
                    self.reward[player] = ((8 * 3 + fanCnt) / 2) ** 0.5
                    self.reward[self.curPlayer] -= 1
            else:
                if isSelfDrawn:
                    self.reward = [-(8 + fanCnt)] * 4
                    self.reward[player] = (8 + fanCnt) * 3
                else:
                    self.reward = [-8] * 4
                    self.reward[player] = 8 * 3 + fanCnt
                    self.reward[self.curPlayer] -= fanCnt
            self.done = True
        except Exception as e:
            raise Error(player)


class MahjongGBEnv_Print(gym.Env):
    agent_names = ['player_%d' % i for i in range(1, 5)]

    def __init__(self, config):
        assert 'agent_clz' in config, "must specify agent_clz to process features!"
        self.agentclz = config['agent_clz']
        assert issubclass(self.agentclz, MahjongGBAgent), "ageng_clz must be a subclass of MahjongGBAgent!"
        self.duplicate = config.get('duplicate', True)
        self.variety = config.get('variety', -1)
        self.r = random.Random()
        self.normalizeReward = config.get('reward_norm', False)
        self.observation_space = self.agentclz.observation_space
        self.action_space = self.agentclz.action_space

    def decode_tile(self, tile):
        """解码牌面，将英文代码转换为中文描述"""
        suits = {'W': '万', 'B': '筒', 'T': '条', 'F': '风', 'J': '箭'}
        numbers = {
            '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        if tile[0] in suits:
            if tile[0] in ['W', 'B', 'T']:
                return suits[tile[0]] + numbers[tile[1]]
            elif tile[0] == 'F':
                winds = {'1': '东', '2': '南', '3': '西', '4': '北'}
                return winds[tile[1]] + suits[tile[0]]
            elif tile[0] == 'J':
                dragons = {'1': '中', '2': '发', '3': '白'}
                return dragons[tile[1]] + suits[tile[0]]
        return tile  # 如果解码失败，返回原始代码

    def sort_hand(self, hand):
        """排序手牌"""
        order = {'W': 1, 'B': 2, 'T': 3, 'F': 4, 'J': 5}
        return sorted(hand, key=lambda tile: (order[tile[0]], tile[1]))

    def reset(self, prevalentWind=-1, tileWall=''):
        # Create agents to process features
        self.agents = [self.agentclz(i) for i in range(4)]
        self.reward = None
        self.done = False
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        # Init prevalent wind
        self.prevalentWind = random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        for agent in self.agents:
            agent.request2obs('Wind %d' % self.prevalentWind)
        # Prepare tile wall
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append('W' + str(i))
                    self.tileWall.append('B' + str(i))
                    self.tileWall.append('T' + str(i))
                for i in range(1, 5):
                    self.tileWall.append('F' + str(i))
                for i in range(1, 4):
                    self.tileWall.append('J' + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = ' '.join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34: (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        # Deal cards
        self._deal()
        return self._obs()

    def step(self, action_dict):
        try:
            self.reward = [0, 0, 0, 0]
            if self.state == 0:
                # After Chi/Peng, prepare to Play
                response = self.agents[self.curPlayer].action2response(
                    action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Play':
                    print(f"{self.agent_names[self.curPlayer]} 出了 {self.decode_tile(response[1])}")
                    self._discard(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
                self.isAboutKong = False
            elif self.state == 1:
                # After Draw, prepare to Hu/Play/Gang/BuGang
                response = self.agents[self.curPlayer].action2response(
                    action_dict[self.agent_names[self.curPlayer]]).split()
                if response[0] == 'Hu':
                    print(f"{self.agent_names[self.curPlayer]} 胡了！")
                    self.shownTiles[self.curTile] += 1
                    self._checkMahjong(self.curPlayer, isSelfDrawn=True, isAboutKong=self.isAboutKong)
                elif response[0] == 'Play':
                    print(f"{self.agent_names[self.curPlayer]} 出了 {self.decode_tile(response[1])}")
                    self.hands[self.curPlayer].append(self.curTile)
                    self._discard(self.curPlayer, response[1])
                elif response[0] == 'Gang' and not self.myWallLast and not self.wallLast:
                    print(f"{self.agent_names[self.curPlayer]} 杠了 {self.decode_tile(response[1])}")
                    self._concealedKong(self.curPlayer, response[1])
                elif response[0] == 'BuGang' and not self.myWallLast and not self.wallLast:
                    print(f"{self.agent_names[self.curPlayer]} 补杠了 {self.decode_tile(response[1])}")
                    self._promoteKong(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
            elif self.state == 2:
                # After Play, prepare to Chi/Peng/Gang/Hu/Pass
                responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if
                             i != self.curPlayer}
                t = {i: responses[i].split() for i in responses}
                # Priority: Hu > Peng/Gang > Chi
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if t[i][0] == 'Hu':
                        print(f"{self.agent_names[i]} 胡了！")
                        self._checkMahjong(i)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if t[i][0] == 'Gang' and self._canDrawTile(i) and not self.wallLast:
                            print(f"{self.agent_names[i]} 杠了 {self.decode_tile(self.curTile)}")
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == 'Peng' and not self.wallLast:
                            print(f"{self.agent_names[i]} 碰了 {self.decode_tile(self.curTile)}")
                            self._pung(i, self.curTile)
                            break
                    else:
                        i = (self.curPlayer + 1) % 4
                        if t[i][0] == 'Chi' and not self.wallLast:
                            print(f"{self.agent_names[i]} 吃了 {self.decode_tile(t[i][1])}")
                            self._chow(i, t[i][1])
                        else:
                            for j in range(1, 4):
                                i = (self.curPlayer + j) % 4
                                if t[i][0] != 'Pass': raise Error(i)
                            if self.wallLast:
                                # A draw
                                print("游戏和局。")
                                self.obs = {i: self.agents[i].request2obs('Huang') for i in range(4)}
                                self.reward = [0, 0, 0, 0]
                                self.done = True
                            else:
                                # Next player
                                self.curPlayer = (self.curPlayer + 1) % 4
                                self._draw(self.curPlayer)
            elif self.state == 3:
                # After BuGang, prepare to Hu/Pass
                responses = {i: self.agents[i].action2response(action_dict[self.agent_names[i]]) for i in range(4) if
                             i != self.curPlayer}
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if responses[i] == 'Hu':
                        print(f"{self.agent_names[i]} 胡了！")
                        self._checkMahjong(i, isAboutKong=True)
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] != 'Pass': raise Error(i)
                    self._draw(self.curPlayer)
        except Error as e:
            player = e.args[0]
            print(f"错误: {self.agent_names[player]} 进行了一个无效操作。")
            self.obs = {i: self.agents[i].request2obs('Player %d Invalid' % player) for i in range(4)}
            self.reward = [10] * 4
            self.reward[player] = -30
            self.done = True
        return self._obs(), self._reward(), self._done(), self._info()

    def _obs(self):
        return {self.agent_names[k]: v for k, v in self.obs.items()}

    def _reward(self):
        if self.reward: return {self.agent_names[k]: self.reward[k] for k in self.obs}
        return {self.agent_names[k]: 0 for k in self.obs}

    def _done(self):
        done = {'__all__': self.done}
        for k in self.obs: done[self.agent_names[k]] = self.done
        return done

    def _info(self):
        return {self.agent_names[k]: {} for k in self.obs}

    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()

    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)

    def _deal(self):
        self.hands = []
        self.packs = []
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            # 排序手牌
            sorted_hand = self.sort_hand(hand)
            self.hands.append(sorted_hand)
            self.packs.append([])
            hand_str = ' '.join(self.decode_tile(tile) for tile in sorted_hand)
            print(f"{self.agent_names[i]} 的起手牌: {hand_str}")
            self.agents[i].request2obs(' '.join(['Deal', *sorted_hand]))
        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)

    def _draw(self, player):
        tile = self._drawTile(player)
        print(f"{self.agent_names[player]} 摸到 {self.decode_tile(tile)}")
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Draw' % player)
        self.obs = {player: self.agents[player].request2obs('Draw %s' % tile)}

    def _discard(self, player, tile):
        if tile not in self.hands[player]: raise Error(player)
        self.hands[player].remove(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs('Player %d Play %s' % (player, tile))
        self.obs = {i: self.agents[i].request2obs('Player %d Play %s' % (player, tile)) for i in range(4) if
                    i != player}

    def _kong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs('Player %d Gang' % player)
        self._draw(player)
        self.reward[player] = 0.4

    def _pung(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3: raise Error(player)
        for i in range(3): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('PENG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Peng' % player)
        self.obs = {player: self.agents[player].request2obs('Player %d Peng' % player)}
        self.reward[player] = 0.3

    def _chow(self, player, tile):
        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]: raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        # offer: 123 for which tile is offered
        self.packs[player].append(('CHI', tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d Chi %s' % (player, tile))
        self.obs = {player: self.agents[player].request2obs('Player %d Chi %s' % (player, tile))}
        self.reward[player] = 0.3

    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4: raise Error(player)
        for i in range(4): self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(('GANG', tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs('Player %d AnGang' % player)
        self.agents[player].request2obs('Player %d AnGang %s' % (player, tile))
        self._draw(player)
        self.reward[player] = 0.4

    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == 'PENG' and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0: raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ('GANG', tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs('Player %d BuGang %s' % (player, tile))
        self.obs = {i: self.agents[i].request2obs('Player %d BuGang %s' % (player, tile)) for i in range(4) if
                    i != player}
        self.reward[player] = 0.4

    def _checkMahjong(self, player, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[player]),
                hand=tuple(self.hands[player]),
                winTile=self.curTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=self.shownTiles[self.curTile] == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=player,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8: raise Error('Not Enough Fans')
            print(f"{self.agent_names[player]} 胡了！共 {fanCnt} 番")
            self.obs = {i: self.agents[i].request2obs('Player %d Hu' % player) for i in range(4)}
            if self.normalizeReward:
                self.reward = [-1] * 4
                if isSelfDrawn:
                    self.reward[player] = ((8 + fanCnt) * 3 / 2) ** 0.5
                else:
                    self.reward[player] = ((8 * 3 + fanCnt) / 2) ** 0.5
                    self.reward[self.curPlayer] -= 1
            else:
                if isSelfDrawn:
                    self.reward = [-(8 + fanCnt)] * 4
                    self.reward[player] = (8 + fanCnt) * 3
                else:
                    self.reward = [-8] * 4
                    self.reward[player] = 8 * 3 + fanCnt
                    self.reward[self.curPlayer] -= fanCnt
            self.done = True
        except Exception as e:
            raise Error(player)
