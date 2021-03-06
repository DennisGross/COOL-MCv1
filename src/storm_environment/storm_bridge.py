import random
import json
import numpy as np
import stormpy
import stormpy as sp
import stormpy.examples.files
import stormpy.simulator as sps
from src.storm_environment.model_checker import ModelChecker
'''
This class should be the only class that has contact with Storm.
'''
class StormBridge:

    def __init__(self, path, constant_definitions, wrong_action_penalty, reward_flag, disabled_features):
        '''
        Initialize Storm Bridge.
        :param path, path to prism file
        :constant_definitions, constant definitions
        :wrong_action_penalty, penalty for taking wrong action
        :reward_flag, reward (True) or costs (False)
        '''
        self.disabled_features = disabled_features
        self.simulator = self.create_simulator(path, constant_definitions)
        self.model_checker = ModelChecker()
        self.constant_definitions = constant_definitions
        self.state_json_example = self.__preprocess_state_json_example(self.simulator.restart()[0])
        self.wrong_action_penalty = wrong_action_penalty
        self.reward_flag = reward_flag
        self.path = path
        

    def __preprocess_state_json_example(self, json_example):
        '''
        Preprocess the state_json_example and ignore all the disabled actions.
        :param json_example
        :return state_json_example
        '''
        dummy_state = {}
        json_example = str(json_example)
        for k in json.loads(json_example):
            if k in self.disabled_features:
                continue
            else:
                value = json.loads(json_example)[k]
                if type(value) == type(True):
                    if value:
                        value = 1
                    else:
                        value = 0
                dummy_state[k] = value
        json_object = json.dumps(dummy_state)  
        return json_object





    def create_simulator(self, path, constant_definitions):
        '''
        Create the simulator for PRISM file (path) and constant definitions
        :param path, path to PRISM file
        :param constant_definitions, constant definitions for PRISM file
        :return simulator
        '''
        prism_program = stormpy.parse_prism_program(path)
        prism_program = stormpy.preprocess_symbolic_input(prism_program, [], constant_definitions)[0].as_prism_program()
        suggestions = dict()
        i = 0
        for m in prism_program.modules:
            for c in m.commands:
                if not c.is_labeled:
                    suggestions[c.global_index] = "tau_" + str(i) #str(m.name)
                    i+=1

        prism_program = prism_program.label_unlabelled_commands(suggestions)

        simulator = stormpy.simulator.create_simulator(prism_program)
        simulator.set_action_mode(stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)
        return simulator

    

    def step(self, action_name):
        '''
        Take action ACTION_NAME in simulator
        :param action_name, action name
        :return (new state, reward, done)
        '''
        penalty = 0
        current_available_actions = sorted(self.simulator.available_actions())
        if len(current_available_actions)==0:
            return self._state, self.reward, True
        if action_name not in current_available_actions:
            action_name = current_available_actions[0]
            penalty = -self.wrong_action_penalty
       
        data = self.simulator.step(action_name)
        self._state = str(data[0])
        self.reward = data[1]
        #self._state, self.reward = self.simulator.step(action_name)
        
        
        self._state = self.parse_state(self._state)
        self.reward = self.reward[0]
        if self.reward_flag==False:
            self.reward *= (-1)
        if penalty!=0:
            self.reward = penalty
        return self._state, self.reward, self.simulator.is_done()

    def reset(self):
        '''
        Reset simulator
        :return state
        '''
        data = self.simulator.restart()
        #_state, reward = self.simulator.restart()
        _state = str(data[0])
        reward = data[1]
        self._state = self.parse_state(str(_state))
        return self._state

    def parse_state(self, state):
        '''
        Parse json state to numpy state
        :state, json state
        :return numpy state
        '''
        arr = []
        for k in json.loads(state):
            if k in self.disabled_features:
                continue
            value = json.loads(state)[k]
            if type(value) == type(True):
                if value:
                    value = 1
                else:
                    value = 0
            arr.append(value)
        return np.array(arr, dtype=np.float32)


    