import numpy as np
import gymnasium as gym
import cv2
from collections import deque

class DiscreteCarRacingActions(gym.ActionWrapper):
    ACTIONS = np.array([
        [0.0, 0.0, 0.0],   # lift
        [-1.0, 0.0, 0.0],  # left
        [1.0, 0.0, 0.0],   # right
        [0.0, 1.0, 0.0],   # accelerate
        [0.0, 0.0, 0.1],   # brake
    ], dtype=np.float32)

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))

    def action(self, index):
        return self.ACTIONS[index]

class DiscreteCarRacingObservations(gym.ObservationWrapper):
    def __init__(self, env, stacked_frames=3):
        super().__init__(env)
        self.stacked_frames = stacked_frames
        self.frames = deque(maxlen=stacked_frames)
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                                shape=(48, 48, stacked_frames), 
                                                dtype=np.float32)
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)                       #Turning the image to grayscale
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA) #Resizing the image to 48x48
        normed = resized.astype(np.float32) / 255.0                        #Normalizing the image
        self.frames.append(normed)                                         #Adding the new frame to the deque
        while len(self.frames) < self.stacked_frames:
            self.frames.append(np.zeros_like(normed))            
        return np.stack(self.frames, axis=-1)