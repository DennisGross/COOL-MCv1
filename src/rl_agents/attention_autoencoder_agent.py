from src.rl_agents.agent import Agent
import os
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import PolicyStep
import json

class AttentionAutoencoderAgent(Agent):

    def __init__(self, project, autoencoder):
        super().__init__()
        self.environment = project.environment
        self.autoencoder = autoencoder
        self.features = self.get_feature_names()

    def get_feature_names(self):
        features = []
        d = json.loads(self.environment.storm_bridge.state_json_example)
        for f in d.keys():
            features.append(f)
        return features

    def numpy_to_attention_autonencoder_input(self, x):
        state = ''
        for i in range(len(self.features)):
            state+= self.features[i] + str(int(x[i])) + ','
        return state[0:-1]

    def get_action_index(self,attention_input_str):
        input_text = tf.constant([
            str(attention_input_str), # "It's really cold here."
        ])
        result = self.autoencoder.tf_translate(input_text = input_text)
        i = 0
        action_name = str(result['text'][0].numpy(), 'utf-8')
        action_name = action_name.upper()
        print(action_name)
        action_idx = self.environment.action_mapper.action_name_to_action_index(action_name)
        if action_idx == None:
            action_idx = 0
        return action_idx

        


    def select_action(self, time_step, deploy):
        x = time_step.observation
        x = self.numpy_to_attention_autonencoder_input(x)
        action_idx = self.get_action_index(x)
        return PolicyStep(action=tf.constant(action_idx, dtype=tf.int64))


