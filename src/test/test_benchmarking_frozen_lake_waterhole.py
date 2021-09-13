import unittest
import os
from src.utilities.project import Project
from src.utilities.monitor import Monitor
from src.utilities.training import train

class BenchmarkingFrozenLakeWaterholeTest(unittest.TestCase):


    def setUp(self):
        # init all common command line arguments for each test
        self.command_line_arguments = {}
        self.command_line_arguments['prism_dir'] = 'prism_files'
        self.command_line_arguments['project_dir'] = 'projects'
        self.command_line_arguments['prism_file_path'] = 'frozen_lake_nxn.prism'

        self.command_line_arguments['constant_definitions'] = 'slippery=0.1,SIZE=20'
        self.command_line_arguments['max_steps'] = 100
        self.command_line_arguments['reward_flag'] = False
        self.command_line_arguments['wrong_action_penalty'] = 70

        self.command_line_arguments['learning_rate'] = 0.0001

        self.command_line_arguments['epsilon'] = 1
        self.command_line_arguments['epsilon_dec'] = 0.9999
        self.command_line_arguments['epsilon_min'] = 0.05
        
        self.command_line_arguments['num_episodes'] = 100000
        self.command_line_arguments['eval_interval'] = 5000
        
        self.command_line_arguments['prop'] = 'Pmax=? [F "water"]'
        self.command_line_arguments['prop_type'] = 'min_prop'
        
        
        self.command_line_arguments['disabled_features'] = ''


    def test_frozen_lake_20x20_not_waterhole_sarsamax(self):
        self.command_line_arguments['project_name'] = 'test_frozen_lake_20x20_not_waterhole_sarsamax'
        self.command_line_arguments['architecture'] = 'sarsamax'
        self.command_line_arguments['gamma'] = 1
        self.command_line_arguments['alpha'] = 0.6
        project_folder = os.path.join(self.command_line_arguments['project_dir'], self.command_line_arguments['project_name'])

        project = Project(self.command_line_arguments)
        monitor = Monitor(project_folder, self.command_line_arguments['architecture'])
        train(project, monitor, self.command_line_arguments['num_episodes'], self.command_line_arguments['eval_interval'])
        self.assertEqual(0,0)


    def test_frozen_lake_20x20_not_waterhole_reinforce(self):
        self.command_line_arguments['project_name'] = 'test_frozen_lake_20x20_not_waterhole_reinforce'
        self.command_line_arguments['architecture'] = 'reinforce'
        self.command_line_arguments['layers'] = 3
        self.command_line_arguments['neurons'] = 128
        self.command_line_arguments['episode_collections'] = 1
        self.command_line_arguments['batch_size'] = 64
        self.command_line_arguments['gamma'] = 1
        self.command_line_arguments['replay_buffer_size'] = 10000
        project_folder = os.path.join(self.command_line_arguments['project_dir'], self.command_line_arguments['project_name'])
        project = Project(self.command_line_arguments)
        monitor = Monitor(project_folder, self.command_line_arguments['architecture'])
        train(project, monitor, self.command_line_arguments['num_episodes'], self.command_line_arguments['eval_interval'])
        self.assertEqual(0,0)


    def test_frozen_lake_20x20_not_waterhole_hillclimbing(self):
        self.command_line_arguments['project_name'] = 'test_frozen_lake_20x20_not_waterhole_hillclimbing'
        self.command_line_arguments['architecture'] = 'hillclimbing'
        self.command_line_arguments['noise_scale'] = 1e-2
        self.command_line_arguments['gamma'] = 1
        project_folder = os.path.join(self.command_line_arguments['project_dir'], self.command_line_arguments['project_name'])
        project = Project(self.command_line_arguments)
        monitor = Monitor(project_folder, self.command_line_arguments['architecture'])
        train(project, monitor, self.command_line_arguments['num_episodes'], self.command_line_arguments['eval_interval'])
        self.assertEqual(0,0)
    

    def test_frozen_lake_20x20_not_waterhole_dqn(self):
        self.command_line_arguments['project_name'] = 'test_frozen_lake_20x20_not_waterhole_ddqn'
        self.command_line_arguments['architecture'] = 'dqn'
        self.command_line_arguments['layers'] = 3
        self.command_line_arguments['neurons'] = 128
        self.command_line_arguments['batch_size'] = 64
        self.command_line_arguments['replace'] = 500
        self.command_line_arguments['gamma'] = 1
        self.command_line_arguments['replay_buffer_size'] = 10000
        self.command_line_arguments['epsilon'] = 0.1
        project_folder = os.path.join(self.command_line_arguments['project_dir'], self.command_line_arguments['project_name'])
        project = Project(self.command_line_arguments)
        monitor = Monitor(project_folder, self.command_line_arguments['architecture'])
        train(project, monitor, self.command_line_arguments['num_episodes'], self.command_line_arguments['eval_interval'])
        self.assertEqual(0,0)

    
if __name__ == '__main__':
    unittest.main()