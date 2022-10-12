from genericpath import sameopenfile
from operator import le
from game import SnakeGame
from constant import *
from network import *
from plot import plot
import random
import numpy as np
import torch
from collections import deque

class Agent:
    def __init__(self):
        self.num_game = 0
        self.network = DQN(11, 3)
        self.trainer = Trainer(self.network, lr = 0.001, gamma = 0.9)
        self.replay_buffer = deque(maxlen = STATES_TO_TRAIN)
    
    def get_state(self, game):
        self.game = game
        x = game.head.x
        y = game.head.y
        dir = game.direction
        points = self.get_points(x, y)
        
        straight_is_danger, right_is_danger, left_is_danger = self.get_dangers(dir, points)
        
        dir_is_right = (dir == RIGHT)
        dir_is_left = (dir == LEFT)
        dir_is_up = (dir == UP)
        dir_is_down = (dir == DOWN)

        food_is_left = (game.food.x < x)
        food_is_right = (game.food.x > x)
        food_is_up = (game.food.y < y)
        food_is_down = (game.food.y > y)
        
        state = [straight_is_danger, right_is_danger, left_is_danger,
                 dir_is_left, dir_is_right, dir_is_up, dir_is_down,
                 food_is_left, food_is_right, food_is_up, food_is_down]
        
        return np.array(state, dtype = int)

    def get_points(self, x, y):
        point_right = point(x + BLOCK_SIZE, y)
        point_left = point(x - BLOCK_SIZE, y)
        point_up = point(x, y - BLOCK_SIZE)
        point_down = point(x, y + BLOCK_SIZE)
        points = [point_right, point_down, point_left, point_up]
        return points
    
    def get_dangers(self, dir, points):
        directions = [RIGHT, DOWN, LEFT, UP]

        straight_index = directions.index(dir) 
        right_index = (straight_index + 1) % 4
        left_index = (straight_index - 1) % 4
        
        straight_new_point = points[straight_index]
        right_new_point = points[right_index]
        left_new_point = points[left_index]
        return self.game.is_collision(straight_new_point), self.game.is_collision(right_new_point), self.game.is_collision(left_new_point)
    
    def get_action(self, state):
        epsilon = max(80 - self.num_game, min(2, 120 - self.num_game))

        action = [0, 0, 0]
        if np.random.randint(0, 200) < epsilon:
            move = np.random.randint(0, 3)
            action[move] = 1
            print("random")
            return action
        
        state = torch.tensor(state, dtype = torch.float)
        prediction = self.network(state)
        move = torch.argmax(prediction).item()
        action[move] = 1
        print("best")
        return action

    def train_step(self, state, action, reward, next_state, game_over):
        self.trainer.step(state, action, reward, next_state, game_over)
        self.replay_buffer.append((state, action, reward, next_state, game_over))
    
    def train_replay_buffer(self):
        if len(self.replay_buffer) > BATCH_SIZE:
            sample = random.sample(self.replay_buffer, BATCH_SIZE)
        else:
            sample = self.replay_buffer
        
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.step(states, actions, rewards, next_states, game_overs)
    
def train():
    game = SnakeGame()
    agent = Agent()
    best_score = -1
    total_score = 0
    all_score = []
    all_mean_score = []
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)
        next_state = agent.get_state(game)
        agent.train_step(state, action, reward, next_state, game_over)

        if not game_over:
            continue
        
        agent.train_replay_buffer()
        game.reset()
        agent.num_game += 1
        
        if score > best_score:
            best_score = score
            agent.network.save()
    
        all_score.append(score)
        total_score += score
        all_mean_score.append(total_score/agent.num_game)
        
        plot(agent.num_game, all_score, all_mean_score)

if __name__ == "__main__":
    train()