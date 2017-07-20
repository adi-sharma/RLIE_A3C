# -*- coding: utf-8 -*-
import sys
import numpy as np
from game_env import GameEnv

from constants_a3c import ACTION_SIZE
from constants_a3c import PORT
from constants_a3c import MODE


class GameState(object):
    def __init__(self, thread_index):
        self.game_env = GameEnv(
            zmq_port=(PORT + thread_index), mode=MODE, thread_num=thread_index)

        # collect minimal action set
        self.real_actions = self.game_env.getActions()

        # start first new game
        self.reset()

    def reset(self):
        # start new game
        s, r, t = self.game_env.newGame()

        self.reward = r
        self.terminal = t
        self.s_t = s

    def process(self, action, query):
        # convert action set index to action
        real_action = self.real_actions[action]
        real_query = query + 1
        s, r, t = self.game_env.step(real_action, real_query)

        self.reward = r
        self.terminal = t
        self.s_t = s

    def evalStart(self):
	    self.game_env.evalStart()

    def evalEnd(self):
	    self.game_env.evalEnd()
