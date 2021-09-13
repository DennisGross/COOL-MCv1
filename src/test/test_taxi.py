import unittest
import gym
import math
from collections import deque
from src.storm_environment.environment import StormEnvironment
from src.rl_agents.sarsa_max_agent import SarsaMaxAgent

class TaxiTest(unittest.TestCase):

    
    def test_taxi_collected_actions(self):
        prism_file_path = 'prism_files/taxi.prism'
        constant_definitions = 'passenger_location=0,passenger_destination=1'
        actions =  ['DROP', 'EAST', 'NORTH', 'PICK_UP', 'SOUTH', 'WEST'] 
        max_steps = 9
        wrong_action_penalty=1000
        reward_flag = False
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        self.assertEqual(env.action_mapper.actions, actions)


    def test_taxi_optimal_path(self):
        prism_file_path = 'prism_files/taxi.prism'
        constant_definitions = 'passenger_location=0,passenger_destination=1'
        max_steps = 9
        wrong_action_penalty=1000
        reward_flag = False
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        # ['DROP', 'EAST', 'NORTH', 'PICK_UP', 'SOUTH', 'WEST']
        # Start x=2,y=2
        # Passenger Location: x=0, y=4
        # Passenger Destination: x=0, y=0
        print(env.reset())
        t = env._step(5) # x=1
        total_reward+=t.reward
        print(t)
        t = env._step(5) # x=0
        total_reward+=t.reward
        print(t)
        t = env._step(2) # x=0 y=3
        total_reward+=t.reward
        print(t)
        t = env._step(2) # x=0 y=4
        total_reward+=t.reward
        print(t)
        t = env._step(3) # PICK UP
        total_reward+=t.reward
        print(t)
        t = env._step(4) # x=0 y=3
        total_reward+=t.reward
        print(t)
        t = env._step(4) # x=0 y=2
        total_reward+=t.reward
        print(t)
        t = env._step(4) # x=0 y=1
        total_reward+=t.reward
        print(t)
        t = env._step(4) # x=0 y=0
        total_reward+=t.reward
        print(t)
        t = env._step(0) # DROP
        total_reward+=t.reward
        print(t)
        self.assertEqual(-189,total_reward)


        

        


if __name__ == '__main__':
    unittest.main()