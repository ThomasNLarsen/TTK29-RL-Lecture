import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 10


def collision_with_apple(apple_position, score):
    # Spawn new apple
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnakeEnv(gym.Env):

    def __init__(self, render_mode="human"):
        super(SnakeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Define action space: we let the snake move in 4 discrete directions: [up, down, right, left]
        self.action_space = spaces.Discrete(4)
        #
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float32)

        self.render_mode = render_mode  # Toggle rendering for training vs. eval.


    def render(self):
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10), (0, 255, 0), 3)

        # Takes step after fixed time
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

    def step(self, action):
        if self.render_mode == "human":
            self.render()

        # Change (or not) the snake's direction
        self._dynamics(action=action)

        # Get observation
        obs = self._observe()

        # Check success / failure conditions
        self.done = self._isDone()

        # Calculate immediate reward
        self.reward = self._calcReward()

        info = {}

        return obs, self.reward, self.done, info

    def reset(self):
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        self.prev_reward = 0

        self.done = False



        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)  # to create history

        obs = self._observe()

        return obs


    ### HELPER FUNCTIONS ###
    def _dynamics(self, action):
        self.prev_actions.append(action)
        # Change the head position based on the button direction
        if action == 1:
            self.snake_head[0] += 10  # Up
        elif action == 0:
            self.snake_head[0] -= 10  # Down
        elif action == 2:
            self.snake_head[1] += 10  # Right
        elif action == 3:
            self.snake_head[1] -= 10  # Left

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

    def _observe(self):
        # Get an observation of the current state of the game
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # Compile observation:
        obs = np.array([head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions))

        return obs

    def _isDone(self):
        # The episode ends when the snake is out of bounds, or bites itself:
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            # Show end result
            if self.render_mode == "human":
                font = cv2.FONT_HERSHEY_SIMPLEX
                self.img = np.zeros((500, 500, 3), dtype='uint8')
                cv2.putText(self.img, 'Your Score is {}'.format(self.score), (140, 250), font, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('a', self.img)
                cv2.waitKey(0)  # Wait until key is pressed

            # Episode is over
            return True

        # Else: episode is not over
        return False

    def _calcReward(self):
        # If snake is out-of-bounds or has bitten itself, reward is -10
        if self.done:
            self.reward = -10
            return self.reward

        # Cumulative reward is proportional to the length of the snake.
        self.total_reward = len(self.snake_position) - 3  # default length is 3

        # Immediate reward when the snake has eaten an apple
        self.reward = self.total_reward - self.prev_reward

        # Recall the current total reward for the next step.
        self.prev_reward = self.total_reward

        return self.reward

    def _getInfo(self):
        return self.info