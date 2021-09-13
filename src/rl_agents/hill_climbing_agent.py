import numpy as np
from src.rl_agents.agent import Agent
from tf_agents.trajectories import PolicyStep
import tensorflow as tf
import os


class HillClimbingAgent(Agent):

    def __init__(self, state_size, number_of_actions, gamma, noise_scale):
        super().__init__()
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.w =  1e-4*np.random.rand(state_size, number_of_actions)
        self.rewards = []
        self.gamma = gamma
        self.best_R = -np.inf
        self.best_w = np.random.rand(state_size, number_of_actions)
        self.noise_scale = noise_scale
        self.progress_counter = 0

        
    def select_action(self, time_step, deploy):
        state = time_step.observation
        x = None
        if deploy:
            x = np.dot(state, self.best_w)
        else:
            x = np.dot(state, self.w)
        probs = np.exp(x)/sum(np.exp(x))
        action_index = np.argmax(probs)
        return PolicyStep(action=tf.constant(action_index, dtype=tf.int64))
        

    def store_experience(self, time_step, action_step, n_time_step):
        reward = float(n_time_step.reward)
        self.rewards.append(reward)


    def episodic_learn(self):
        discounts = [self.gamma**i for i in range(len(self.rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, self.rewards)])
        if R >= self.best_R: # found better weights
            self.best_R = R
            self.best_w = self.w
            self.noise_scale = max(1e-3, self.noise_scale / 2)
            self.w += self.noise_scale * np.random.rand(*self.w.shape)
        else: # did not find better weights
            self.noise_scale = min(2, self.noise_scale * 2)
            self.w = self.best_w + self.noise_scale * np.random.rand(*self.w.shape)
            self.progress_counter += 1
            if self.progress_counter >= 10000:
                self.w = self.noise_scale * np.random.rand(*self.w.shape)
        self.rewards = []
        


    def save(self, root_folder):
        np.save(os.path.join(root_folder, 'model.npy'), self.best_w)

    def load(self, root_folder):
        try:
            self.best_w = np.load(os.path.join(root_folder, 'model.npy'))
            self.w = np.load(os.path.join(root_folder, 'model.npy'))
        except:
            self.best_w = np.random.rand(self.state_dimension, self.number_of_actions)
            self.w = np.random.rand(self.state_dimension, self.number_of_actions)

