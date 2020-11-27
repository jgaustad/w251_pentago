import numpy as np
import random
from datetime import datetime
import time
from collections import defaultdict
from helper_func import *
import pickle

class pentago:
    """
    
    """

    def __init__(self, state = None):
        """Initializes the class reservation"""
        #print('initializing')
        
        if state == None:
            self.state = state = np.zeros((6,6), dtype=np.int)
        self.history = []
        self.winner = None
        self.gameover = False
        self.player_turn = 1
    
    def current_board_state(self):
        # need to return a copy or bad stuff happens
        return copy.copy(self.state)
    
    def game_history(self, player, move, cuad, rotatation):
        self.history.append((boardstate_to_ideal_key(self.state), player, move, cuad, rotatation))
        #return self.history

    def find_winner(self, board_state):
        player1_win = False
        player_min1_win = False
        diagonal1 = board_state.diagonal()
        diagonal2 = np.fliplr(board_state).diagonal()
        winning_slices =  np.vstack([board_state[1:,:].T, board_state[:-1,:].T, # all columns
                              board_state[:,1:], board_state[:,:-1], # all rows
                              diagonal1[1:], diagonal1[:-1], # diagonal 1
                              diagonal2[1:],diagonal2[1:], # diagonal 2
                              board_state.diagonal(offset=1), board_state.diagonal(offset=-1), # diagonal offsets 
                              np.fliplr(board_state).diagonal(offset=1), np.fliplr(board_state).diagonal(offset=-1)] ) # diagonal offsets
        sums = np.dot(winning_slices, np.array([1,1,1,1,1]))
        if 5 in sums: player1_win = True
        if -5 in sums: player_min1_win = True
        if player1_win == True or player_min1_win == True:
           # print("Player 1 winner?", player1_win, "Player -1 winner?", player_min1_win)
            self.gameover = True
            if player1_win == True:
                self.winner = 1
            elif player_min1_win ==True:
                self.winner = -1
            self.history.append(self.winner)
        return "Win"

    def check_gameover(self):
        if not 0 in self.state:
              self.gameover = True
              print("The game board is full!")
        
    def full_move(self, move, cuad, direction, player, dtype=np.int):
        if player != self.player_turn:
            print( "error, wrong player turn. No move taken.")
            return 'Error, wrong player turn.'
        self.state = fullmove(self.state,move, cuad, direction, player)


        self.game_history(move, player, cuad, direction)
        self.find_winner(self.state) #return in find_winner if a winner is found
        self.check_gameover() #return in check_gameover
        if player == 1:
            self.player_turn = -1
        else:
            self.player_turn = 1
        #print('Successful Move')
        return self.state
        
class q_table:

    def __init__(self,length=0, games_played=0):
        """Initializes the class reservation"""
        self.time = datetime.now()
        self.length = length
        self.q_dict = {}
        self.games_played = games_played

  #def time(self):
    #self.time = time

    def length(self):
        self.length += 1
    #self.length = length  
    
    def get_q_value(self, boardstate):
        return self.q_dict.get(boardstate, (0, 0))
    
    def update_q_value(self, boardstate, new_val, update_function = None):
        q_val, n = self.get_q_value(boardstate) 
        if update_function:
            #print('using custom function')
            self.q_dict[boardstate] = update_function(q_val, n, new_val)
        else:
            self.q_dict[boardstate] = [new_val, n+1]
        return self.q_dict[boardstate]
    
    def update_post_game(self, history, update_fn):
        winner = history[-1]
        
        for boardposition in history[-2::-1]:
            key = boardposition[0]
            #print(key, winner)
            self.update_q_value(key, winner, update_fn)
    
def my_func(q, n, nn):
    #print('here',q, n, nn, 'end')
    #q, n = cv
    return (q*n+nn)/(n+1), n+1

def dampen_func(q, n, nn):
    #print('here',q, n, nn, 'end')
    #q, n = cv
    return (q*(n+1)+nn)/(n+2), n+1
    
class qtable_agent:
    
    def __init__(self, player = 1, epsilon = 1, epsilon_decay = .99995, epsilon_min = .5, q_table = q_table()):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = q_table
        self.player = player
        self.epsilon_min = epsilon_min
        
            
    def get_avail_moves(self,boardstate):
        """
        This method creates a list with available spaces in the board and combination of quadrant and rotation
        The input is the board state (6x6) numpy array
        """
        x = np.where(boardstate == 0)
        #print(x)
        available_positions_for_placement = list(zip(x[0], x[1]))
        
        # all available positions (p), quadrants(q), rotations(r)
        available_moves = [(p,q,r) for p in available_positions_for_placement for q in [1,2,3,4] for r in [-1,1]]
        #print(len(available_moves))
        return available_moves
    
    def get_possible_next_boardstates(self, boardstate):
        next_possible_boardstates = defaultdict(list)
        for move in self.get_avail_moves(boardstate):
            possible_boardstate = fullmove(boardstate,*move, self.player)
            key = boardstate_to_ideal_key(possible_boardstate)
            #print(key)
            next_possible_boardstates[key].append(move)
            
        return next_possible_boardstates
    
    def make_move(self, game):
        
        # get the current boardstate from the pentago class
        boardstate = game.current_board_state()
        
        # get possible next possible boardstates
        next_possible_boardstates = self.get_possible_next_boardstates(boardstate)
        key_list = list(next_possible_boardstates.keys())
        
        # determine if to take random move
        if np.random.rand() < self.epsilon:
            random_bs = random.choice(key_list)
            random_mv = next_possible_boardstates[random_bs][0]
            
            game.full_move(*random_mv,self.player)
            
        else:
            #print("not random", self.player)
            q_values_list = [self.q_table.get_q_value(bs)[0]*self.player for bs in key_list] # *player flips the q's for -1 player to allow max calc
            #print(q_values_list)
            
            # get random index of a max value
            max_q = (max(q_values_list))
            index_of_all_max = [i for i in range(len(q_values_list)) if q_values_list[i] == max_q]
            random_max_q_index = random.choice(index_of_all_max)
            
            mv_to_take = next_possible_boardstates[key_list[random_max_q_index]][0]
            game.full_move(*mv_to_take, self.player)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
        else:
            self.epsilon = self.epsilon_min
            
def big_sim(n_games, agent1, agent2, qtables_to_update = [], update_cadence = 1):
    game_times = []
    q_dict_update_times = []
    winner_list = []
    
    for n in range(n_games):
        print('game', n, end = ' ')
        game_start = time.time()
        g = pentago() # initialize game
        
        while g.gameover == False:
            agent1.make_move(g)
            if g.gameover ==True:
                break
            agent2.make_move(g)
            
        game_times.append(time.time()-game_start)
        
        # check for winner and update q_table(s)
        if g.winner:
            winner_list.append(g.winner)
            print('winner: ', g.winner)
            if n%update_cadence == 0:
                # update time
                update_start = time.time()
                for q_tab in qtables_to_update:
                    q_tab.update_post_game(g.history, dampen_func)
                q_dict_update_times.append(time.time()-update_start)
        else:
            print('No winner!')
    # end of simulation runs, save q_table(s) to disk
    qt_num = 1
    time_str = str(datetime.now())[:19].replace(':','_')
    for q_tab in qtables_to_update:
        with open(f'q_table{qt_num}_'+time_str+'.pickle2.2', 'wb') as file:
            pickle.dump(q_tab, file, protocol = pickle.HIGHEST_PROTOCOL)
        qt_num += 1
    
    print('game_times:', game_times)
    print('q_dict_update_times:', q_dict_update_times)
    print('winners:', winner_list)
    winner1 = len([w for w in winner_list if w == 1])
    winner_min1 = len([w for w in winner_list if w == -1])
    print("Player 1 wins: ", winner1)
    print("Player -1 wins:", winner_min1)
    
    return game_times, q_dict_update_times, winner_list
    
# Note you will overwrite this q_table and agents if you run this cell again.    Verify you won't lose your data!
qtable1 = q_table()  
agent1 = qtable_agent(player = 1,  q_table=qtable1)
agent2 = qtable_agent(player = -1, q_table=qtable1)


    
##################################################
## Change number of games to simulate here
n_games = 5000
##################################################


time0 = time.time()
for x in range(10):
    game_times, q_update_times, winners = big_sim(n_games, agent1, agent2, qtables_to_update=[qtable1])
print(time.time()-time0, 'seconds.')