import pygame
from constant import *
import random
import numpy as np

pygame.init()
font = pygame.font.Font("arial.ttf", 25)

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = RIGHT
        self.head = point(self.w/2, self.h/2)
        self.snake = [self.head,
                      point(self.head.x - BLOCK_SIZE, self.head.y),
                      point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food = None
        self.place_food()
        
    def place_food(self):
       index_x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)
       index_y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)
       x = index_x * BLOCK_SIZE
       y = index_y * BLOCK_SIZE
       self.food = point(x, y)
       if self.food in self.snake:
           self.place_food()
    
    def play_step(self, action):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               pygame.quit()
               quit()

        # move
        self.move(action)
        
        # check if game over
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -1
            return reward, game_over, self.score 
        elif self.head == self.food:
            self.score += 1
            reward = 1
            self.place_food()
        else:
            self.snake.pop()
        
        # update ui
        self.update()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
        
    def move(self, action):
        # [straight, right, left]
        
        directions = [RIGHT, DOWN, LEFT, UP]
        index = directions.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[index]
        elif np.array_equal(action, [0, 1, 0]):
            new_index = (index+1) % 4
            new_dir = directions[new_index]
        else:
            new_index = (index-1) % 4
            new_dir = directions[new_index]
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if new_dir == RIGHT:
            x += BLOCK_SIZE
        elif new_dir == LEFT:
            x -= BLOCK_SIZE
        elif new_dir == DOWN:
            y += BLOCK_SIZE
        else:
            y -= BLOCK_SIZE
        
        self.head = point(x, y)
        self.snake.insert(0, self.head)
             
    def is_collision(self, location = None):
        if location == None:
            x = self.head.x
            y = self.head.y
            location = self.head
        else:
            x = location.x
            y = location.y

        if x < 0 or x > self.w - BLOCK_SIZE:
            return True
        if y < 0 or y > self.h - BLOCK_SIZE:
            return True
        if location in self.snake[1:]:
            return True
        return False      
    
    def update(self):
        self.display.fill(GRAY)
       
        for body in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(body.x, body.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(body.x+2, body.y+2, 17, 16))
            
        pygame.draw.rect(self.display, ORANGE, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: {}".format(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()