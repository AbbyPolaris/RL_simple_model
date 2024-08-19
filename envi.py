import time
import random
import numpy as np
def flatten_array(matrix):
    flattned = []
    for row in matrix:
        flattned += row
    return flattned

def treegonal_to_decimal(trigonal_array):
    decimal = 0
    for index ,i in enumerate(trigonal_array):
        decimal += i*(3**index)
    return decimal

class environment:
    def __init__(self, space_len=10):
        self.space_len = space_len
        self.adamak_state = 0
        self.all_states = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        self.observable_space = [self.all_states[0][5:7], self.all_states[1][5:7]]
        self.score = 0
        self.terminal = False
        #self.gravity_counter = 0
    def step(self,action):
        reward = 0
        new_state_1 = random.randint(0,8)
        new_state_2 = random.randint(0,6)
        res_adam_state = self.adamak_state
        self.all_states[1] = self.all_states[1][1:] + [1 if new_state_1 == 0 else 2 if new_state_1 == 1 else 0] 
        
        if self.all_states[1][-1] == 2 or self.all_states[1][-2] == 2:
            self.all_states[0] = self.all_states[0][1:] + [0]
        else:
            self.all_states[0] = self.all_states[0][1:] + [1 if new_state_2 == 0 else 2 if new_state_2 == 1 else 0] 
        #if self.adamak_state == 1:
            #self.gravity_counter += 1
        self.observable_space = [self.all_states[0][5:7], self.all_states[1][5:7]]

        if action == 1:
            self.adamak_state = 1
            #self.gravity_counter = 0
        elif action == 0:
            #if self.gravity_counter > 0:
            self.adamak_state = 0
        else:
            pass
        ## here can be replaced with a reward function.
        if self.observable_space[1][0] == 2 and self.adamak_state == 0:
            #reward = -1
            self.terminal = True
            #print("terminate")
        elif self.observable_space[0][0] == 2 and self.adamak_state == 1:
            #reward = -1
            self.terminal = True
            #print("terminate")
        elif self.observable_space[0][0] == 1 and self.adamak_state == 1:
            self.score += 1
            reward = 1
        elif self.observable_space[1][0] == 1 and self.adamak_state == 0:
            reward = 1
            self.score += 1
            #terminal = True
        elif self.adamak_state != res_adam_state:
            if self.observable_space[1-self.adamak_state][0] != 2:
                reward = -.3
        else:
            pass

        
        
        return (treegonal_to_decimal([self.adamak_state] + flatten_array(self.observable_space)), 2*reward, self.terminal)
    def reset(self):
        self.score = 0
        self.adamak_state = 0
        self.terminal = False
        self.all_states = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        self.observable_space = [self.all_states[0][5:7], self.all_states[1][5:7]]
        return (treegonal_to_decimal([self.adamak_state] + flatten_array(self.observable_space)) , self.terminal)
    def print_all_env(self):
        print(self.all_states)
    def print_observable(self):
        print(self.observable_space)

