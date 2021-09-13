import unittest
from src.storm_environment.environment import StormEnvironment

class StormEnvironmentTest(unittest.TestCase):

    def test_frozen_lake_4x4_waterhole_terminal(self):
        prism_file_path = 'prism_files/frozen_lake_4x4.prism'
        constant_definitions = 'slippery=0'
        max_steps = 20
        wrong_action_penalty=1000
        reward_flag = False
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        env._step(0)
        env._step(0)
        env._step(0)
        last_state = env._step(0)
        self.assertEqual(-100,last_state.reward)
        self.assertTrue(last_state.is_last())

    def test_frozen_lake_4x4_frisbee_terminal(self):
        prism_file_path = 'prism_files/frozen_lake_4x4.prism'
        constant_definitions = 'slippery=0'
        max_steps = 20
        wrong_action_penalty=1000
        reward_flag = False
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        env._step(0)
        env._step(0)
        env._step(2)
        env._step(0)
        env._step(2)
        env._step(2)
        last_state = env._step(2)
        self.assertEqual(0,last_state.reward)
        self.assertTrue(last_state.is_last())

    def test_frozen_lake_4x4_frisbee_terminal_reward(self):
        prism_file_path = 'prism_files/frozen_lake_4x4.prism'
        constant_definitions = 'slippery=0'
        max_steps = 20
        wrong_action_penalty=1000
        reward_flag = False
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        t = env._step(0)
        total_reward += float(t.reward)
        env._step(0)
        total_reward += float(t.reward)
        env._step(2)
        total_reward += float(t.reward)
        env._step(0)
        total_reward += float(t.reward)
        env._step(2)
        total_reward += float(t.reward)
        env._step(2)
        total_reward += float(t.reward)
        t = env._step(2)
        total_reward += float(t.reward)
        self.assertEqual(-60,total_reward)
        
    def test_flying_bees_2_no_sting(self):
        prism_file_path = 'prism_files/flying_bees_2.prism'
        constant_definitions = 'xMax=40,yMax=40,slickness=0'
        max_steps = 20
        wrong_action_penalty=1000
        reward_flag = True
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        total_reward += env._step(0).reward
        total_reward += env._step(0).reward
        total_reward += env._step(0).reward
        total_reward += env._step(0).reward
        self.assertEqual(400,total_reward)
    
    def test_flying_bees_2_terminal_sting(self):
        prism_file_path = 'prism_files/flying_bees_2.prism'
        constant_definitions = 'xMax=2,yMax=2,slickness=0'
        max_steps = 100
        wrong_action_penalty=1000
        reward_flag = True
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        time_step = env._step(0)
        counter = 0
        while time_step.is_last()==False:
            time_step = env._step(0)
            counter+=1
        self.assertTrue(True)
        self.assertLessEqual(counter,100)

    def test_flying_bees_2_terminal_max_steps(self):
        prism_file_path = 'prism_files/flying_bees_2.prism'
        constant_definitions = 'xMax=200,yMax=200,slickness=0'
        max_steps = 5
        wrong_action_penalty=1000
        reward_flag = True
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        time_step = env._step(0)
        counter = 1
        while time_step.is_last()==False:
            time_step = env._step(0)
            counter+=1
        self.assertTrue(True)
        self.assertEqual(counter,5)

    def test_taxi_optimal_path(self):
        prism_file_path = 'prism_files/taxi.prism'
        constant_definitions = 'passenger_location=0,passenger_destination=1'
        max_steps = 20
        wrong_action_penalty=1000
        reward_flag = False
        total_reward = 0
        env = StormEnvironment(prism_file_path, constant_definitions, max_steps, wrong_action_penalty, reward_flag)
        # ['DROP', 'EAST', 'NORTH', 'PICK_UP', 'SOUTH', 'WEST']
        # Start x=2,y=2
        # Passenger Location: x=0, y=4
        # Passenger Destination: x=0, y=0
        t = env._step(1) # x=1
        t = env._step(1) # x=0
        t = env._step(2) # x=0 y=3
        t = env._step(2) # x=0 y=4
        print(t)
        t = env._step(3) # PICK UP
        t = env._step(4) # x=0 y=3
        t = env._step(4) # x=0 y=2
        t = env._step(4) # x=0 y=1
        t = env._step(4) # x=0 y=0
        print(t)
        t = env._step(0) # DROP
        print(t)
        self.assertEqual(-60,total_reward)

if __name__ == '__main__':
    unittest.main()