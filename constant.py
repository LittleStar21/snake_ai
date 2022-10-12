#snake
from collections import namedtuple
point = namedtuple("point", "x, y") 

# directions
RIGHT = 1
LEFT = 2
UP = 3
DOWN = 4

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (117, 117, 117)
YELLOW = (254, 228, 64)
GREEN1 = (1, 133, 117)
GREEN2 = (0, 245, 212)
ORANGE = (235, 137, 52)

BLOCK_SIZE = 20
SPEED = 30

#replay buffer
STATES_TO_TRAIN = 100000
BATCH_SIZE = 1000