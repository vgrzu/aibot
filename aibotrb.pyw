import cv2
import numpy as np
import random
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym
from tensorflow.keras.layers import Input, Dense, GRU, Conv2D, MaxPooling2D, Flatten, TimeDistributed
from tensorflow.keras.models import Model
from collections import deque
import keyboard  # For sending real keyboard inputs
import mss  # For screen capture
import tkinter as tk  # For GUI


class AdvancedGameEnv(gym.Env):
    def __init__(self):
        super(AdvancedGameEnv, self).__init__()
        self.action_space = spaces.Discrete(8)  # 8 possible actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype=np.uint8)
        self.memory = deque(maxlen=10)  # Short-term memory for state history
        
    def capture_screen(self):
        """Capture the game screen using mss."""
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 800, "height": 600}  # Adjust according to your setup
            img = sct.grab(monitor)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, (100, 100))
        return img
    
    def step(self, action):
        """Perform an action and get the new observation, reward, and done signal."""
        self.perform_action(action)
        observation = self.capture_screen()
        reward = self.calculate_reward(observation)
        done = False  # Game is continuous
        return observation, reward, done, {}
    
    def reset(self):
        self.memory.clear()  # Reset memory on environment reset
        return self.capture_screen()
    
    def calculate_reward(self, observation):
        """Calculate a reward based on the current observation."""
        return np.mean(observation) / 255.0 - 0.5  # Adjust based on actual game conditions
    
    def perform_action(self, action):
        """Perform an action using W, A, S, D keys."""
        if action == 0:  # Move forward
            keyboard.press('w')
            sleep(0.1)
            keyboard.release('w')
        elif action == 1:  # Move backward
            keyboard.press('s')
            sleep(0.1)
            keyboard.release('s')
        elif action == 2:  # Rotate left
            keyboard.press('a')
            sleep(0.1)
            keyboard.release('a')
        elif action == 3:  # Rotate right
            keyboard.press('d')
            sleep(0.1)
            keyboard.release('d')
        elif action == 4:  # Jump
            keyboard.press('space')
            sleep(0.1)
            keyboard.release('space')
        elif action == 5:  # Crouch
            keyboard.press('ctrl')
            sleep(0.1)
            keyboard.release('ctrl')
        elif action == 6:  # Sprint
            keyboard.press('shift')
            sleep(0.1)
            keyboard.release('shift')
        elif action == 7:  # Attack
            keyboard.press('f')  # Assuming 'F' is the attack key
            sleep(0.1)
            keyboard.release('f')


class MemoryAugmentedModel:
    def __init__(self):
        inputs = Input(shape=(10, 100, 100, 3))  # Input sequence of images
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)  # Process images with Conv2D
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)  # Apply MaxPooling
        x = TimeDistributed(Flatten())(x)  # Flatten the output for GRU
        x = GRU(64, return_sequences=False)(x)  # GRU expects 3D input after flattening
        outputs = Dense(8, activation="softmax")(x)  # Predict next action probabilities
        self.model = Model(inputs, outputs)

    def remember_and_predict(self, state_sequence):
        """Predict action based on recent states."""
        state_sequence = np.array(state_sequence).reshape((1, 10, 100, 100, 3))
        return self.model.predict(state_sequence)

memory_model = MemoryAugmentedModel()


env = DummyVecEnv([lambda: AdvancedGameEnv()])
agent = PPO("CnnPolicy", env, verbose=1)


class SymbolicModule:
    def __init__(self):
        self.rules = {
            "attack": lambda x: x["enemy_nearby"],
            "navigate": lambda x: x["path_clear"],
        }
    
    def interpret(self, game_state):
        """Interpret game state and suggest a task based on rules."""
        for task, rule in self.rules.items():
            if rule(game_state):
                return task
        return "explore"

# Main bot poop
def run_bot():
    global running
    running = True
    observation = env.reset()
    state_sequence = deque(maxlen=10)  # Limited memory of recent states

    while running:
        game_state = {
            "enemy_nearby": random.choice([True, False]),
            "path_clear": random.choice([True, False])
        }
        symbolic_task = SymbolicModule().interpret(game_state)
        
        # Decide actions using memory-augmented model if enough states are remembered
        state_sequence.append(observation)
        if len(state_sequence) == 10:
            action_prediction = memory_model.remember_and_predict(list(state_sequence))
            action = np.argmax(action_prediction)
        else:
            action, _states = agent.predict(observation)
        
        
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()

# turn the bot on
def start_bot():
    global running
    running = True
    status_label.config(text="Bot is running...")
    run_bot()

# fuck off bot
def stop_bot():
    global running
    running = False
    status_label.config(text="Bot has stopped.")


root = tk.Tk()
root.title("Game Bot Control")
root.geometry("300x150")

status_label = tk.Label(root, text="Bot is stopped.", font=("Helvetica", 12))
status_label.pack(pady=20)

start_button = tk.Button(root, text="Start Bot", command=start_bot, width=15)
start_button.pack(pady=5)

stop_button = tk.Button(root, text="Stop Bot", command=stop_bot, width=15)
stop_button.pack(pady=5)


running = False


root.mainloop()
