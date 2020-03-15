import gym
from gym import error, spaces, utils
from gym.utils import seeding

import torch

XS =  1  # x player
OS = -1  # o player
EM =  0  # empty
DS = 0.5 # disabled

reps = ['.', 'X', 'O']

###

class SkgEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
        width=9,
        height=9,
        disabled=[]
    ):
        self.width = width
        self.height = height
        self._board = Board(width, height)
        self.possible_outputs = width*height - len(disabled)
        self.input_layers = 5

        self._board.disable_tiles(disabled)

    def A(self, state):
        if state is None:
            return Board(self.width, self.height).possible_moves()
        else:
            return self._board.possible_moves()

    def E(self, state_, action):
        reward = self.act(state_, action)
        state = self.x()
        done = self._board.full()

        return state, reward, done, {}

    def step(self, a):
        return self.E(None, a)

    def act(self, state_, action):
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

        # no reward if game isn't over
        if len(self._board.possible_moves()) > 0:
            return 0.0

        # just reward based on winning / losing
        #   it may help to reward based on dominance
        reward = self._board.winner() # works because last move is always made by p1
        reward = 1.0*reward # reward needs to be a float
        return reward

    def x(self, perspective=XS):
        state = [[[0 for r in range(self._board.hgt)] for t in range(self._board.wdt)] for l in range(5)]
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

                if tile == DS:
                    state[2][y][x] = 1.0
                else:
                    state[2][y][x] = 0.0

                state[3][y][x] = 0.0
                state[4][y][x] = turn*1.0

                x += 1
            x = 0
            y += 1

        # figuring captures is a useful feature. maybe eschew this
        captures = self._board.capturing_moves()
        for x, y in captures:
            try:
                state[3][y][x] = 1.0
            except:
                print(x, state)
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

    def dominance(self):
        x_ct, o_ct = 0, 0
        for row in self.board:
            for tile in row:
                if tile == XS:
                    x_ct += 1
                elif tile == OS:
                    o_ct += 1
        return x_ct, o_ct

    def winner(self):
      x_ct, o_ct = self.dominance()
      if x_ct > o_ct:
          return XS
      elif o_ct > x_ct:
          return OS
      else:
          return EM

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

    def check_reversi(self, sx, sy, xdir, ydir, turn):
        nx, ny = sx+xdir, sy+ydir
        if not self.in_bounds(nx, ny):
            return False
        elif self.get(nx, ny) == EM:
            return False
        elif self.get(nx, ny) == turn:
            self.set(sx, sy, turn)
            return True
        elif self.get(nx, ny) == self.opp_turn(turn):
            if self.check_reversi(nx, ny, xdir, ydir, turn):
                self.set(sx, sy, turn)
                return True
            else:
                return False

    def place(self, x, y, turn):
        if not self.in_bounds(x, y) or self.get(x, y) == DS:
            return False
        elif not self.get(x, y) == EM:
            return False

        self.set(x, y, turn)
        self.empty_squares -= 1

        for j in range(-1, 2):
            for i in range(-1, 2):
                nx = x+i
                ny = y+j

                if self.in_bounds(nx, ny) and self.get(nx, ny) == self.opp_turn(turn):
                    self.check_reversi(nx, ny, i, j, turn)
                else:
                    continue

        return True

    def possible_moves(self):
        out = []
        for y in range(self.hgt):
            for x in range(self.wdt):
                if self.board[y][x] == EM:
                    out.append((x, y))
        return out

    def check_capture(self, sx, sy, xdir, ydir, depth, turn):
        nx, ny = sx+xdir, sy+ydir
        if not self.in_bounds(nx, ny):
            return False
        elif self.get(nx, ny) == EM:
            return False
        elif self.get(nx, ny) == turn and depth != 1:
            return True
        elif self.get(nx, ny) == self.opp_turn(turn):
            if self.check_capture(nx, ny, xdir, ydir, depth+1, turn):
                return True
            else:
                return False
        else:
            return False

    def capturing_moves(self):
        turn = self.turn()
        possible_moves = self.possible_moves()
        capture = []
        not_terrible = []

        for move in possible_moves:
            x, y = move

            for i in range (-1, 2):
                for j in range(-1, 2):
                    if self.check_capture(x, y, i, j, 1, turn):
                        #y gets flipped somewhere
                        new_move = (x, self.hgt-y-1)
                        capture.append(new_move)

        return capture

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
