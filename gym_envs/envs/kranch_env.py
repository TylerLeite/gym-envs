"""
  Kranch class: a bunch of tools for managing Kranch state
"""

import numpy as np
import random as r
import os
import copy

import sys
COMMIT_SUDOKU = sys.exit

import pdb

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import torch


"""
  Key for using the piece info
  Attack Power, Health, Attack Range, Move Count, Starting X for player 1, Starting Y for player 1, etc.
"""
ATK, HP, RNG, MV, SX1, SY1, SX2, SY2 = range(8)

DEV = True

OUTPUT = {}
OUTPUT['write-games-to-file'] = True
OUTPUT['out-directory'] = os.getcwd() + '/test_data/games'
try:
  OUTPUT['out-directory'] = os.getcwd() + '/test_data/' + sys.argv[1] + '/games'
except IndexError:
  pass

try:
  os.mkdir(OUTPUT['out-directory'], 0o777)
except OSError as e:
  print(e)
  pass

VERBOSE = {}
VERBOSE['silent-sanity-checks'] = True
VERBOSE['apply-move-sanity-check'] = False
VERBOSE['flip-board-sanity-check'] = False
VERBOSE['print-board-states'] = True
VERBOSE['print-board-state-separators'] = False
VERBOSE['print-board-on-flip'] = True
VERBOSE['print-possible-moves'] = False
VERBOSE['print-chosen-move'] = True
VERBOSE['print-compressed-state'] = False
VERBOSE['print-full-state'] = False
VERBOSE['print-state-details'] = False
VERBOSE['print-simulated-game-progress'] = True

class KranchEnv(gym.Env):
  metadata = {'render.modes': ['human', 'text']}

  def __getitem__(self, item):
    """ This just lets us use square brackets to access properties"""
    return getattr(self, item)

  def __init__(self,
    width=7,
    height=7,
  ):
    board_dims = (width, height)
    self.init(board_dims)

    self.possible_outputs = self.get_num_dense_moves()
    self.input_layers = 13
    self.input_details = 16

  def A(self, state):
    if state is None:
      state = self.x()

    return self.get_moves_dense(state)

  def E(self, state, action):
    sf, r = self.apply_move_dense(state, action)
    done = r != 0
    return sf, r, done, {}

  def step(self, action):
    return self.E(None, action)

  def x(self):
    return self.start_state()

  def reset(self):
    pass

  def render(self, mode='human', state=None):
    if state is None:
      return

    if mode == 'text':
      self.print_board(state)
    else:
      print('render mode "' + mode + '" not implemented')

  def close(self):
    pass

  def init(self, board_dims):
    """
      board_dims is a 2-tuple of the game board's dimensions (width, height)
      pieces is a 2-tuple of information about the game pieces (total_troops_per_side, ramping_troops_per_side)

      Troop order is King, Ranger, Assassin, Knight, Cannibal, Healer
      Move order is
        MoveNorth, MoveSouth, MoveEast, MoveWest,
        AttackNorth, AttackSouth, AttackEast, AttackWest,
        AttackNorthWest, AttackNorthEast, AttackSouthWest, AttackSouthEast,
        RangerAttackUURR, RangerAttackUUR, RangerAttackUUU, RangerAttackUUL, RangerAttackUULL,
        RangerAttackDDRR, RangerAttackDDR, RangerAttackDDD, RangerAttackDDL, RangerAttackDDLL,
        RangerAttackRR, RangerAttackRRU, RangerAttackRRD,
        RangerAttackLL, RangerAttackLLU, RangerAttackLLD,
        Pass
    """
    # Board dimension info
    self.width = board_dims[0]
    self.w_mid = int(self.width/2)
    self.height = board_dims[1]
    self.h_mid = int(self.height/2)

    # North, South, East, West, in (dx, dy) form
    self.directions = [
      ( 0,  1), ( 0, -1), ( 1,  0), (-1, 0),
      ( 0,  1), ( 0, -1), ( 1,  0), (-1, 0),
      (-1,  1), ( 1,  1), (-1, -1), ( 1, -1),
      (-2,  2), (-1,  2), ( 0,  3), ( 1,  2), (2,  2),
      (-2, -2), (-1, -2), ( 0, -3), ( 1, -2), (2, -2),
      (-2,  0), (-2,  1), (-2, -1),
      ( 2,  0), (-2,  1), (-2, -1),
    ]
    self.move_ct = len(self.directions) + 1
    self.range_exlusive_move_ct = 16

    # Troop info
    self.TroopKinds = ['King', 'Ranger', 'Assassin', 'Knight', 'Cannibal', 'Healer']
    self.RampingKinds = ['Cannibal']
    self.RangedKinds = ['Ranger']
    self.BackstabKinds = ['Assassin']

    self.num_pieces = len(self.TroopKinds)
    self.ramp_pieces = len(self.RampingKinds)

    w  = self.width
    wm = self.w_mid
    h  = self.height
    hm = self.h_mid

    self.Boulder  = ( 0, 10)
    self.King     = ( 2,  5, 1, 2, 0,   hm, w-1,   hm)
    self.Ranger   = ( 2,  8, 2, 2, 0, hm-1, w-1, hm+1)
    self.Assassin = ( 2,  8, 1, 3, 1, hm+1, w-2, hm-1)
    self.Knight   = ( 4, 10, 1, 2, 1,   hm, w-2,   hm)
    self.Cannibal = ( 2,  6, 1, 2, 1, hm-1, w-2, hm+1)
    self.Healer   = (-2,  8, 1, 2, 0, hm+1, w-1, hm-1)

    # Turn info
    self.num_moves = 1+4+8+16 # TODO: pass in an array of moves to constructor?
    self.max_moves = self.num_pieces * self.num_moves

  # TODO: There may be a more efficient way to do this
  def compress_state(self, state):
    """Return a 2D array of booleans, whether a square is occupied"""
    M, N = self.width, self.height

    out = np.zeros([M, N])
    for i in range(M):
      for j in range(N):
        layer = state['Positions'][i][j]
        if np.count_nonzero(layer) > 0:
          out[i][j] = 1
        else:
          continue

    return out

  def tuplify_state(self, state):
    """Output the state as a tuple"""
    return (state['Positions'], state['Details'])

  def densify_move(self, move):
    """Move array is sparsely populated, this mapping makes it dense"""
    P = self.num_pieces
    X = self.move_ct
    Y = self.range_exlusive_move_ct
    Z = X - Y

    troop_index = int(move / X)
    move_index = move % X
    if move_index == 28:
      move_index = Z-1

    if move_index >= Z:
      # Dealing with a ranger here
      troop_name = self.TroopKinds[troop_index]
      range_index = self.RangedKinds.index(troop_name)

      # To avoid gaps, start counting ranged moves at 0 rather than 14
      move_index -= Z

      return P * Z + range_index * Y + move_index
    else:
      return troop_index * Z + move_index

  def sparsify_move(self, move):
    """So that the output from the NN can be used in this sim"""
    P = self.num_pieces
    X = self.move_ct
    Y = self.range_exlusive_move_ct
    Z = X - Y

    troop_index = int(move / Z)
    move_index = move % Z

    if move_index == Z-1 and troop_index < P:
      move_index = X-1
    if troop_index >= P:
      # This is where the ranged-only moves start
      move = move - P * Z

      # Recalculate troop and move index with new move
      range_index = int(move / Y)
      move_index = move % Y + Z # we subtracted 13 earlier

      # Get troop index from range index
      troop_name = self.RangedKinds[range_index]
      troop_index = self.TroopKinds.index(troop_name)

    return troop_index * X + move_index

  def start_state(self):
    """
      The structure of our state is as follows
      let P = number of pieces per side
      let C = number of ramping troops per side (troops with mutable attack stat)
      Positions: MxNxZ matrix, where
        M = width of the board
        N = height of the borad
        Z = 2P+1

        The order of the layers is as follows:
          Depth 0: Boulders
          Depth 1 -> P: Current player's pieces in order
          Depth P+1 -> 2P: Other player's pieces in order

      Details: vector of length 2P+2C
        The order of the vector is as follows:
          Depth 0 -> P-1: Move count for current player's pieces in order
          Depth P -> 2P-1: Attack boolean for current player's pieces in order
          Depth 2P -> 2P+C-1: Current player's ramping attack troops in order
          Depth 2P+C -> 2P+2C-1: Other player's ramping attack troops in order

    """
    # These are defined in config or via command line
    M, N = self.width, self.height
    P = self.num_pieces
    C = self.ramp_pieces
    Z = 2*P + 1

    # Initialize matrices with zeros
    positions = np.zeros([M, N, Z])
    details = np.zeros(2*P + 2*C)

    # Positions for boulders (exact code used by server)
    for i in range(2, M-2):
      for j in range(0, N):
        if r.randrange(100) < 20:
          positions[i][j][0] = self.Boulder[HP]
          positions[M-i-1][N-j-1][0] = self.Boulder[HP]

    for i, kind in enumerate(self.TroopKinds):
      # Get the starting stats for this troop
      stats = self[kind]

      # Positions for pieces (constants in __init__ taken from server code)
      positions[stats[SX1]][stats[SY1]][i+1] = stats[HP]
      positions[stats[SX1]][stats[SY1]][0]   = 0 # In case a boulder was generated here

      positions[stats[SX2]][stats[SY2]][P+i+1] = stats[HP]
      positions[stats[SX2]][stats[SY2]][0]   = 0 # In case a boulder was generated here

      # Details for pieces (constants in __init__ taken from server code)
      details[i]   = stats[MV]
      details[P+i] = True # Attack boolean

    # Set ramper attacks
    for i, kind in enumerate(self.RampingKinds):
      # Get the starting stats for this troop
      stats = self[kind]

      # Set the starting ramper attacks
      details[2*P+i]   = stats[ATK]
      details[2*P+C+i] = stats[ATK]

    # Return the state object
    return {'Positions': positions, 'Details': details}

  def get_num_dense_moves(self):
    """ Return how many moves exist (even illegal ones) """
    P = self.num_pieces
    X = self.move_ct
    Y = self.range_exlusive_move_ct
    Z = X - Y

    # Number of pieces * number of moves all pieces can make plus number of ranged troops times the number of moves only rangers can make
    return P*Z + len(self.RangedKinds)*Y

  # returns empty if turn over
  def get_moves(self, state, ff=True):
    """
      Moves are encoded as follows:
        troop_index * max_possible_moves + move_index
    """
    M, N = self.width, self.height
    P = self.num_pieces
    X = self.move_ct

    # All possible moves (for all pieces)
    possible_moves = []

    # WARNING: severe nesting incoming. I refuse to refactor, though
    # For each occupied square
    compressed = self.compress_state(state)
    for i in range(M):
      for j in range(N):
        if compressed[i][j] == 0:
          # Square is empty, ignore it
          continue
        else:
          # If that square is occupied by one of your troops
          for n in range(P):
            if not state['Positions'][i][j][n+1] == 0:
              # If that troop has actions
              can_attack = state['Details'][P+n]
              can_move = state['Details'][n] > 0
              if can_attack or can_move:
                # Get the starting position for this move
                start_index = n * X
                # You can always pass
                possible_moves.append(start_index + X-1)

                # For each direction
                for d in range(4, X-1):
                  if d > 11 and not self.TroopKinds[n] in self.RangedKinds:
                    # Ignore the last 16 actions if you aren't a ranged troop
                    break

                  # Get the target square
                  nx, ny = i + self.directions[d][0], j + self.directions[d][1]

                  # Check if it's in bounds
                  if nx < 0 or nx >= M or ny < 0 or ny >= N:
                    continue
                  else:
                    # Check if you can attack there (target is not empty)
                    if compressed[nx][ny] != 0:
                      if can_attack:
                        friend_on_sq = False
                        for t in range(P):
                          if not state['Positions'][nx][ny][t+1] == 0:
                            friend_on_sq = True
                            break
                        if (ff or self.TroopKinds[n] == 'Healer') or not friend_on_sq:
                          possible_moves.append(start_index + d)
                    elif can_move and d < 8:
                      # If you can't attack there, you can move there (if it's in range)
                      possible_moves.append(start_index + d-4)

    return possible_moves

  def get_moves_no_friendly_fire(self, state):
    return self.get_moves(state, False)

  def get_moves_dense_no_friendly_fire(self, state):
    all_moves = self.get_moves(state, False)
    return [self.densify_move(move) for move in all_moves]

  def get_moves_dense(self, state):
    all_moves = self.get_moves(state)
    return [self.densify_move(move) for move in all_moves]

  def get_moves_dense_bitmask(self, state):
    every_move = self.get_num_dense_moves()
    legal_moves = self.get_moves_dense(state)
    return [int(i in legal_moves) for i in range(every_move)]

  def apply_move(self, state, move):
    """
      Simulate the game for the given state + action, return state' and reward.
      For efficiency, assume that the move is legal
    """
    M, N = self.width, self.height
    P = self.num_pieces
    X = self.move_ct

    new_state = copy.deepcopy(state)
    reward = 0

    # Decipher the move
    troop_index = int(move / X)
    troop_name = self.TroopKinds[troop_index]
    move_index = move % X

    # If the move is "pass" zero out move_ct and attack_boolean. No other changes
    if move_index == X-1:
      new_state['Details'][troop_index] = 0
      new_state['Details'][troop_index + P] = False

    # Otherwise, it's a move or an attack
    else:
      # Either way, need to find the target square
      loop_done = False
      for i in range(M):
        for j in range(N):
          # This is the current location of the piece you are moving
          if state['Positions'][i][j][troop_index+1] != 0:
            # Get the new position for the piece / position of the piece to attack
            nx = i+self.directions[move_index][0]
            ny = j+self.directions[move_index][1]

            # To be clear, player did one of two things
            #   1) moved a piece on (i, j) to (nx, ny)
            #   2) attacked the piece on (nx, ny) with the piece on (i, j)

            # If the move is "move", change location of piece, lower move_ct, no other changes
            if move_index < 4:
              # Unfortunately, need to loop through the whole board to find the piece that is moving
              if state['Positions'][i][j][troop_index+1] != 0:
                # Move the piece, clear the old one
                new_state['Positions'][nx][ny][troop_index+1] = state['Positions'][i][j][troop_index+1]
                new_state['Positions'][i][j][troop_index+1] = 0

                # Also need to not forget to reduce the move count, since that would be just silly
                new_state['Details'][troop_index] -= 1

                loop_done = True # Also need to break out of the loop 1 level up
                break

            # If the move is "attack", run combat
            else:
              # Figure out what the target was
              target_layer = state['Positions'][nx][ny]
              target_index = -1
              for k, piece in enumerate(target_layer):
                if piece != 0:
                  target_index = k
                  break
                else:
                  continue
              target_name = self.TroopKinds[(target_index-1)%P]

              # Get the attack of your troop
              attack = 0
              # Check if your troop is a ramper
              ramp_index = 0
              is_ramper = False
              if troop_name in self.RampingKinds:
                # Get attack from state
                ramp_index = self.RampingKinds.index(troop_name)
                is_ramper = True
                attack = state['Details'][2*P+ramp_index]
              else:
                # Get attack from constants
                attack = self[troop_name][ATK]

              if troop_name in self.BackstabKinds:
                # Check if you get a backstab bonus
                if self.directions[move_index][0] < 0:
                  attack *= 4

              # Perform the combat
              health = max(0, target_layer[target_index]-attack) # Prevent overkill
              health = min(health, self[target_name][HP]) # Prevent overhealing
              new_state['Positions'][nx][ny][target_index] = health

              # Check if the opponent is dead
              if new_state['Positions'][nx][ny][target_index] == 0:
                # Check if the dead guy is a king
                if target_index-1 == self.TroopKinds.index('King'):
                  # You killed your own king
                  reward = -1
                elif target_index-1 == self.TroopKinds.index('King') + P:
                  # You killed the other king
                  reward = 1
                # Check if you ramped the attacker
                elif is_ramper:
                  # Gain +1 attack, restore to full health
                  new_state['Details'][2*P+ramp_index] += 1
                  new_state['Positions'][i][j][troop_index+1] = self[troop_name][HP]

              # Zero out move_ct and attack_boolean
              new_state['Details'][troop_index] = 0
              new_state['Details'][troop_index + P] = False

            loop_done = True # Also need to break out of the loop 1 level up
            break

        if loop_done:
          # Already executed the move
          break
        else:
          # Keep looking
          continue

    if DEV and VERBOSE['apply-move-sanity-check']:
      if not VERBOSE['silent-sanity-checks']:
        print('Apply move', move, 'sanity check: ')
      self.sanity_check(new_state)

    return new_state, reward

  def apply_move_dense(self, state, move):
    return self.apply_move(state, self.sparsify_move(move))

  # TODO: make sure this is all correct
  def flip_board(self, state):
    M, N = self.width, self.height
    P = self.num_pieces
    C = self.ramp_pieces
    Z = 2*P + 1

    flipped_state = {}

    # Zero-out the flipped_state for now, we will fill it in
    #  For some reason, deepcpoying and mutating caused unexpected behavior
    flipped_state['Details'] = [0 for i in state['Details']]
    flipped_state['Positions'] = np.zeros([M, N, Z])

    # Flip the position layers
    for i in range(M):
      for j in range(N):
        # Add the boulders in
        flipped_state['Positions'][i][j][0] = state['Positions'][i][j][0]
        # Flip the piece perspective
        layer = flipped_state['Positions'][i][j]
        layer[1:P+1], layer[P+1:] = state['Positions'][i][j][P+1:], state['Positions'][i][j][1:P+1]

    # Flip the board horizontally
    flipped_state['Positions'] = flipped_state['Positions'][::-1]
    # Flip the board vertically
    for layer in flipped_state['Positions']:
      layer = layer[::-1]

    # Details, reset move / attack counts, flip ramper attacks
    for i, kind in enumerate(self.TroopKinds):
      # Get the starting stats for this troop
      stats = self[kind]

      # Reset the move count and attack boolean
      flipped_state['Details'][i]   = stats[MV]
      flipped_state['Details'][P+i] = True # Attack boolean

    # Flip the ramper attacks
    dets = flipped_state['Details']
    dets[2*P:2*P+C], dets[2*P+C:] = state['Details'][2*P+C:], state['Details'][2*P:2*P+C]

    if DEV and VERBOSE['flip-board-sanity-check']:
      if not VERBOSE['silent-sanity-checks']:
        print("flip_board sanity check...")
      self.sanity_check(flipped_state)

    return flipped_state

  def print_board(self, state):
    """Print out the game state in a human-readable form"""
    M, N = self.width, self.height
    P = self.num_pieces

    out = ''

    if DEV:
      if VERBOSE['print-compressed-state']:
        print('Compressed: ')
        compressed = self.compress_state(state)
        for j in range(N):
          for i in range(M):
            if compressed[i][j] == 1:
              out += '@\n',
            else:
              out += '.\n',
          out += '\n'
        out += '\n'

      if VERBOSE['print-full-state']:
        out += 'Full: \n'
        out += str(state['Positions']) + '\n'

      if VERBOSE['print-state-details']:
        out += str(state['Details']) + '\n'

    piece_reps = '# K R A N C H k r a n c h .'.split()
    # Rotate board for our puny human minds
    for j in range(N):
      for i in range(M):
        piece_index = 0
        while True:
          if state['Positions'][i][j][piece_index] != 0:
            break
          else:
            piece_index += 1

          if piece_index >= 2*P+1:
            piece_index = 2*P+1
            break
        out += piece_reps[piece_index]
      out += '\n'

    if VERBOSE['print-board-state-separators']:
      out += '-'*75 + '\n'

    if not OUTPUT['write-games-to-file']:
      print(out)
    return out

  def sanity_check(self, state):
    """Make sure the state is valid"""
    M, N = self.width, self.height

    valid = True

    # Check that no layer has more than one object on it
    for i in range(M):
      for j in range(N):
        layer = state['Positions'][i][j]
        if np.count_nonzero(layer) > 1:
          valid =  False

    if not valid:
      print('Invalid Game State:')
      self.print_board(state)
      COMMIT_SUDOKU(1)

    return valid

  def run_game(self, state, p1_ai, p2_ai):
    """
      p1_ai and p2_ai are functions that take in a state and output a move
      If one of the ai functions is None, make a random move instead
      Once the state has no legal moves, the game passes to the next player
      This continues until someone dies
    """

    # Next game file name
    files = [int(g.split('.')[0]) for g in os.listdir(OUTPUT['out-directory'])]
    next_game = 0
    if files:
      next_game = max(files) + 1
    out_file = None
    if DEV and OUTPUT['write-games-to-file']:
      out_fname = OUTPUT['out-directory'] + '/' + str(next_game) + '.kranch'
      out_file = open(out_fname, 'a')
      print('out=%d.kranch...' % next_game,)
      if p1_ai is not None and p2_ai is None:
        out_file.write('-10\n')
      elif p1_ai is None and p2_ai is not None:
        out_file.write('-20\n')
      else:
        print(str(p1_ai).split(' ')[1], 'vs.', str(p2_ai).split(' ')[1])
        out_file.write('-1\n')

    if DEV and VERBOSE['print-board-states']:
      if not out_file:
        print('Starting state:')
      state_str = self.print_board(state)
      if out_file:
        out_file.write(state_str)

    p1_move = True
    move_ct = 0
    while True:
      move = None

      all_moves = self.get_moves_no_friendly_fire(state)
      if not all_moves:
        p1_move = not p1_move
        state = self.flip_board(state)
        all_moves = self.get_moves_no_friendly_fire(state)

        if DEV and VERBOSE['print-board-on-flip']:
          state_str = self.print_board(state)
          if out_file:
            out_file.write('-2\n')
            out_file.write(state_str)

      if p1_move:
        if p1_ai:
          move = p1_ai(state)
        else:
          move = r.choice(all_moves)
      else:
        if p2_ai:
          move = p2_ai(state)
        else:
          move = r.choice(all_moves)

      state, reward = self.apply_move(state, move)
      move_ct += 1
      if move_ct > 4000:
        return -1, move_ct
      # Print the new state
      if DEV and VERBOSE['print-board-states']:
        if VERBOSE['print-possible-moves']:
          print('Possible moves: ')
          print(all_moves)

        if VERBOSE['print-chosen-move']:
          if out_file:
            out_file.write(str(move) + '\n')
          else:
            print('Move: ', move)

        state_str = self.print_board(state)
        if out_file:
          out_file.write(state_str)

      if reward != 0:
        winner = 1 # Player 1
        if p1_move and reward == -1:
          winner = 2
        elif not p1_move and reward == 1:
          winner = 2

        if out_file:
          out_file.close()
        return winner, move_ct

  def simulate_n_games(self, n, ai_function, cpu_function, step=100):
    wins = 0
    draws = 0
    for i in range(n):
      if DEV and VERBOSE['print-simulated-game-progress']:
        if i%step == 0:
          print('Game %d:' % i,)

      state = self.start_state()

      # Equal chance to play first / second
      ai_player = None
      winner = None
      if r.randrange(100) < 50:
        ai_player = 1
        winner, moves = self.run_game(state, ai_function, cpu_function)
      else:
        ai_player = 2
        winner, moves = self.run_game(state, cpu_function, ai_function)

      if winner == ai_player:
        wins += 1
        print('AI wins!')
      elif winner == -1:
        draws += 1
        print('AI draws...')
      else:
        print('AI loses :(')

    print("Won %d/%d games. Drew %d" % (wins, n, draws))
