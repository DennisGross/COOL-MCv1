import os
import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import TimeStep
from src.storm_environment.storm_bridge import StormBridge
from src.storm_environment.action_mapper import ActionMapper

class StormEnvironment(py_environment.PyEnvironment):

    def __init__(self, prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag, disabled_features=''):
        '''
        Initialize Storm Bridge.
        :param prism_file_path, path to prism file
        :constant_definitions, constant definitions
        :max_steps, maximal steps
        :wrong_action_penalty, penalty for taking wrong action
        :reward_flag, reward (True) or costs (False)

        '''
        self.storm_bridge = StormBridge(prism_file_path, constant_definitions, wrong_action_penalty, reward_flag, disabled_features)
        self.action_mapper = ActionMapper.collect_actions(self.storm_bridge)
        self.steps = 0
        self.max_steps = max_steps
        time_step = self._reset()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=len(self.action_mapper.actions)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=time_step.observation.shape, dtype=np.float32, minimum=0, name='observation')
        self._reward_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, name='reward')
        self._discount_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, name='discount')
        self.tf_environment = tf_py_environment.TFPyEnvironment(self)

        
        

    
    def _step(self, action_index):
        '''
        Take action action index in environment
        :param action_index, action index
        :return Transition
        '''
        action_name = self.action_mapper.action_index_to_action_name(action_index)
        n_state, reward, done = self.storm_bridge.step(action_name)
        self.steps+=1
        if done or self.steps >= self.max_steps:
            return ts.termination(n_state, reward)
        else:
            return ts.transition(n_state, reward=reward, discount=1.0)

    def _reset(self) -> TimeStep:
        '''
        Resets the environment
        :return: init state
        '''
        self.steps = 0
        n_state = self.storm_bridge.reset()
        return ts.restart(n_state)

    def action_spec(self) -> BoundedArraySpec:
        '''
        Action Spec.
        :return: Action Spec.
        '''
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        '''
        Observation Spec.
        :return: Observation Spec.
        '''
        return self._observation_spec

