from src.rl_agents.agent import Agent
from joblib import dump, load
import os
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import PolicyStep

class DecisionTreeAgent(Agent):

    def __init__(self, environment):
        super().__init__()
        self.environment = environment


    def select_action(self, time_step, deploy):
        x = time_step.observation
        action_name = self.clf.predict(x.reshape(1,x.shape[0]))[0]
        action_idx = self.environment.action_mapper.action_name_to_action_index(action_name)
        return PolicyStep(action=tf.constant(action_idx, dtype=tf.int64))



    def load(self, root_folder):
        self.clf = load(os.path.join(root_folder,'decision_tree.joblib'))




