import random
import tensorflow as tf
from tf_agents.trajectories import PolicyStep
from src.rl_agents.agent import Agent

class RandomAgent(Agent):

    def __init__(self, number_of_actions):
        super().__init__()
        self.number_of_actions = number_of_actions
        self.state_action_memory = {}
        

    def select_action(self, time_step, deploy):
        state = str(time_step.observation.tolist())
        # In Verification, use memory
        if state in self.state_action_memory.keys():
            action_index = self.state_action_memory[state]
        else:
            action_index = random.randint(0, self.number_of_actions-1)
            self.state_action_memory[state] = action_index
        return PolicyStep(action=tf.constant(action_index, dtype=tf.int64))

        

    def eval_clear(self):
        # Clean memory
        self.state_action_memory = {}
