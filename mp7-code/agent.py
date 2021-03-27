import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        self.points = 0
        self.s = None
        self.a = None

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        discrete = self.discretize_state(state)

        if dead:
            reward = -1
        elif points > self.points:
            reward = 1
        else:
            reward = -0.1


        if self._train:
            next_action = self.exploration(discrete)

            if not (self.s == None or self.a == None):
                best_action = np.argmax(self.Q[discrete])
                value = self.Q[discrete + (best_action,)]
                for i in range(3, -1, -1):
                    if value == self.Q[discrete + (i,)] and i > best_action:
                        best_action = i
                alpha = self.C / (self.C + self.N[self.s + (self.a,)])
                self.Q[self.s + (self.a,)] += (alpha * (reward + (self.gamma * self.Q[discrete + (best_action,)]) - self.Q[self.s + (self.a,)]))

            if not dead:
                self.N[discrete + (next_action,)] += 1
            self.points = points
            self.s = discrete
            self.a = next_action
        else:
            next_action = np.argmax(self.Q[discrete])

        if dead:
            self.reset()

        return next_action



    def discretize_state(self, state):

        discrete = np.zeros(8, dtype=int)

        pos_x = state[0]
        pos_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        if pos_x == 40:
            discrete[0] = 1
        elif pos_x == 480:
            discrete[0] = 2

        if pos_y == 40:
            discrete[1] = 1
        elif pos_y == 480:
            discrete[1] = 2

        if food_x == pos_x:
            pass
        elif food_x < pos_x:
            discrete[2] = 1
        elif food_x > pos_x:
            discrete[2] = 2

        if food_y == pos_y:
            pass
        elif food_y < pos_y:
            discrete[3] = 1
        elif food_y > pos_y:
            discrete[3] = 2

        if (pos_x, pos_y - 40) in snake_body:
            discrete[4] = 1
        if (pos_x, pos_y + 40) in snake_body:
            discrete[5] = 1
        if (pos_x - 40, pos_y) in snake_body:
            discrete[6] = 1
        if (pos_x + 40, pos_y) in snake_body:
            discrete[7] = 1

        return tuple(discrete)

    def exploration(self, discrete):
        for i in range(3, -1, -1):
            if self.N[discrete + (i,)] < self.Ne:
                return i

        action = np.argmax(self.Q[discrete])
        value = self.Q[discrete + (action,)]
        if value >= 1:
            print(value)

        for i in range(3, -1, -1):
            if value == self.Q[discrete + (i,)] and i > action:
                action = i

        return action