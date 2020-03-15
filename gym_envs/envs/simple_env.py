import gym
from gym import error, spaces, utils
from gym.utils import seeding

import torch

class SimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        width=2,
        height=2
    ):
        self.width = width
        self.height = height
        self._board = Board(width, height)
        self.possible_outputs = width*height
        self.input_layers = 2

    def A(self, state):
        if state is None:
            return Board(self.width, self.height).possible_moves()
        else:
            return self._board.possible_moves()

    def E(self, state_, action):
        reward = self.act(state_, action)
        state = self.x()
        done = self._board.full() or self._board.turns >= 16

        return state, reward, done, {}

    def step(self, a):
        return self.E(None, a)

    def act(self, state, action):
        # de-incentivize illegal moves
        if action is None:
            return -1

        # extract the coordinates of the move
        x, y = action

        # figure out whose turn it is
        turn = self._board.turn()

        # make the move
        self._board.place(x, y, turn)

        # calculate the difference in score
        reward = self._board.winner()
        reward = 1.0*reward # reward needs to be a float
        #reward = reward * turn #adjust sign depending on who made the move
        #reward = reward / abs(reward) # normalize
        return reward

    def x(self):
        state = [[[0 for r in range(self._board.hgt)] for t in range(self._board.wdt)] for l in range(self.input_layers)]
        x, y = 0, 0
        turn = self._board.turn()
        for row in self._board.board:
            for tile in row:
                if tile == XS:
                    state[0][y][x] = tile*1.0
                else:
                    state[0][y][x] = 0.0

                state[1][y][x] = self._board.turns/16.0
                x += 1
            x = 0
            y += 1

        return state

    def reset(self):
        self._board.reset()
        return self.step(None)[0]

    def render(self, mode='human'):
        print('render not implemented')
        pass

    def close(self):
        # clean up viewer
        pass

#########

'''Otello, by Tyler Leite'''

XS =  1 #x player
OS = -1 #o player
EM =  0 #empty
DS = 0.5 # disabled

reps = ['.', 'X', 'O']

class Board:
    def __init__(self, wdt, hgt):
        self.wdt = wdt
        self.hgt = hgt
        self.reset()

    def reset(self):
        self.turns = 0
        self.board = [[EM for x in range(self.wdt)] for y in range(self.hgt)]
        self.empty_squares = self.wdt * self.hgt

    def full(self):
        return self.empty_squares == 0

    def turn(self):
        return XS

    def winner(self):
        if self.empty_squares == 0:
            return 1
        elif self.turns >= 16:
            return -1
        else:
            return 1 - self.empty_squares/(self.wdt*self.hgt)

    def in_bounds(self, x, y):
        if x < 0 or x >= self.wdt or y < 0 or y >= self.hgt:
            return False
        else:
            return True

    def set(self, x, y, piece):
        if self.in_bounds(x, y):
            self.board[y][x] = piece
            return True
        else:
            return False

    def disable_tiles(self, disabled):
        for x_d, y_d in disabled:
            self.set(x_d, y_d, DS)

    def opp_turn(self, turn):
        return -turn

    def get(self, x, y):
        if self.in_bounds(x, y):
            return self.board[y][x]
        else:
            return DS

    def place(self, x, y, turn):
        self.turns += 1

        if not self.in_bounds(x, y) or self.get(x, y) == DS:
            return False

        if self.get(x, y) == EM:
            self.set(x, y, turn)
            self.empty_squares -= 1
        else:
            self.set(x, y, EM)
            self.empty_squares += 1

        return True

    def possible_moves(self):
        out = []
        for y in range(self.hgt):
            for x in range(self.wdt):
                out.append((x, y))
        return out

    def __str__(self, separator='\n'):
        out = ''
        for y in range(self.wdt):
            for x in range(self.hgt):
                if x == 0:
                    out += str(self.hgt-y)
                out += reps[self.board[y][x]]
            out += '\n'
        out += ' '
        for i in range(self.wdt):
            out += str(chr(97+i))
        out += separator
        return out
