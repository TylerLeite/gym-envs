import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import torch

class TicTacToeRoppEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        width=3,
        height=3
    ):
        self.width = width
        self.height = height
        self._board = Board(width, height)
        self.possible_outputs = width*height
        self.input_layers = 3

    def A(self, state):
        if state is None:
            return Board(self.width, self.height).possible_moves()
        else:
            return self._board.possible_moves()

    def E(self, state_, action):
        reward = self.act(state_, action)
        state = self.x()
        done = reward != 0 or self._board.full()

        return state, reward, done, {}

    def step(self, a):
        return self.E(None, a)

    def act(self, state, action):
        # de-incentivize illegal moves
        if action is None:
            # TODO: raise an exception. this shouldnt happen
            return -1

        # extract the coordinates of the move
        x, y = action

        # figure out whose turn it is
        turn = self._board.turn()

        # make the move
        self._board.place(x, y, turn)

        # calculate the difference in score
        reward = abs(self._board.winner())

        # if the game isn't over, make a random move for the opponent
        if reward == 0 and self._board.empty_squares > 0:
            self.take_random_action()
            reward = (-10.0)*abs(self._board.winner())
        else:
            reward = 1.0

        if self._board.empty_squares == 0 and self._board.winner() == 0:
          return -1.0
        return reward

    def take_random_action(self):
        possible_actions = self._board.possible_moves()
        x, y = random.choice(possible_actions)
        turn = self._board.turn()
        self._board.place(x, y, turn)

    def x(self):
        state = [[[0 for r in range(self._board.hgt)] for t in range(self._board.wdt)] for l in range(self.input_layers)]
        x, y = 0, 0
        turn = self._board.turn()
        for row in self._board.board:
            for tile in row:
                if tile == XS:
                    state[0][y][x] = 1.0
                else:
                    state[0][y][x] = 0.0

                if tile == OS:
                    state[1][y][x] = 1.0
                else:
                    state[1][y][x] = 0.0

                state[2][y][x] = turn*1.0
                x += 1
            x = 0
            y += 1

        return state

    def reset(self):
        self._board.reset()

        # randomly start as player two sometimes
        if random.random() < 0.5:
            # self.take_random_action()
            pass

        return self.step(None)[0]

    def render(self, mode='human'):
        print('render not implemented')
        pass

    def close(self):
        # clean up viewer
        pass

#########

XS =  1.0 #x player
OS = -1.0 #o player
EM =  0.0 #empty
DS = 0.5 # disabled

reps = ['.', 'X', 'O']

class Board:
    def __init__(self, wdt, hgt):
        self.wdt = wdt
        self.hgt = hgt
        self.reset()

    def reset(self):
        self.board = [[EM for x in range(self.wdt)] for y in range(self.hgt)]
        self.empty_squares = self.wdt * self.hgt

    def full(self):
        return self.empty_squares == 0

    def turn(self):
        if self.empty_squares % 2 == (self.wdt * self.hgt) % 2:
            return XS
        else:
            return OS

    def winner(self):
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        winner = 0.0

        for state in win_states:
            piece = 0.0
            check = None
            found = True
            for tile in state:
                x = tile % 3
                y = tile // 3

                piece = self.board[y][x]
                if check is None:
                    check = piece
                if piece == EM or piece != check:
                    found = False
                    break
            if found:
                return piece
        return 0

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
        if not self.in_bounds(x, y) or self.get(x, y) == DS:
            return False
        elif not self.get(x, y) == EM:
            return False

        self.set(x, y, turn)
        self.empty_squares -= 1

        return True

    def possible_moves(self):
        out = []
        for y in range(self.hgt):
            for x in range(self.wdt):
                if self.board[y][x] == EM:
                    out.append((x, y))
        return out

    def __str__(self, separator='\n'):
        out = ''
        for y in range(self.wdt):
            for x in range(self.hgt):
                if x == 0:
                    out += str(self.hgt-y)
                out += reps[int(self.board[y][x])]
            out += '\n'
        out += ' '
        for i in range(self.wdt):
            out += str(chr(97+i))
        out += separator
        return out
