import gym
from gym import error, spaces, utils
from gym.utils import seeding

import torch

XS =  1.0 # x player
OS = -1.0 # o player
EM =  0.0 # empty
DS =  0.5 # disabled

reps = ['.', 'X', 'O']

###

class FicFacFoeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        width=4,
        height=4
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

    def E(self, state_, action, flip_perspective=False):
        reward = self.act(state_, action)
        state = self.x()
        done = reward != 0 or self._board.full()

        return state, reward, done, {}

    def step(self, a):
        return self.E(None, a)

    def act(self, state, action):
        # de-incentivize illegal moves
        legal_moves = self._board.possible_moves()
        if action is None or action not in legal_moves:
            return -2.0

        # extract the coordinates of the move
        x, y = action

        # figure out whose turn it is
        turn = self._board.turn()

        # make the move
        self._board.place(x, y, turn)

        # calculate the difference in score
        reward = abs(self._board.winner())
        # reward = self._board.winner()

        # if reward == 0 and self._board.empty_squares == 0:
        #   reward = -0.01 # de-incentivize draws
        # else:
        reward = 1.0*reward # reward needs to be a float

        return reward

    def x(self, perspective=XS):
        state = [[[0 for r in range(self._board.hgt)] for t in range(self._board.wdt)] for l in range(self.input_layers)]
        x, y = 0, 0
        turn = self._board.turn()
        for row in self._board.board:
            for tile in row:
                if tile == perspective:
                    state[0][y][x] = 1.0
                else:
                    state[0][y][x] = 0.0

                if tile == -perspective:
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
        return self.step(None)[0]

    def render(self, mode='human'):
        print('render not implemented')
        pass

    def close(self):
        # clean up viewer
        pass

#########

class Board:
    def __init__(self, wdt, hgt):
        self.wdt = wdt
        self.hgt = hgt
        self.reset()

    def clone(self):
        b = Board(self.wdt, self.hgt)
        b.board = [[piece for piece in row] for row in self.board]
        b.empty_squares = self.empty_squares
        return b

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
        # check columns
        for x in range(self.wdt):
            check = self.board[0][x]
            win = True
            for y in range(self.hgt):
                if self.board[y][x] != check:
                    win = False
                    break

            if win:
                return check

        # check rows
        for y in range(self.hgt):
            check = self.board[y][0]
            win = True
            for x in range(self.hgt):
                if self.board[y][x] != check:
                    win = False
                    break

            if win:
                return check

        # diagonals
        if self.wdt == self.hgt:
            c1 = self.board[0][0]
            c2 = self.board[0][self.wdt-1]

            w1 = True
            w2 = True
            for d in range(self.wdt):
                if w1:
                    if self.board[d][d] != c1:
                        w1 = False
                if w2:
                    if self.board[d][self.wdt-1-d] != c2:
                        c2 = False

            if w1:
                return c1
            if w2:
                return c2

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
                # if x == 0:
                #     out += str(self.hgt-y)
                out += reps[int(self.board[y][x])]
            out += '\n'
        # out += ' '
        # for i in range(self.wdt):
        #     out += str(chr(97+i))
        # out += separator
        return out
