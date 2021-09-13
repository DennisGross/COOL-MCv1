import unittest
from src.rl_agents.agent_builder import AgentBuilder
from tf_agents.specs import array_spec, BoundedArraySpec
import numpy as np
import os

class AgentTest(unittest.TestCase):

    def test_build_agent(self):
        action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=3, name='action')
        observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0, name='observation')
        agent_builder = AgentBuilder()
        report = dict()
        report['project_dir'] = 'bla'
        report['project_name'] = 'li'
        report['architecture'] = 'sarsamax'
        report['epsilon'] = 0.1
        report['epsilon_dec'] = 0.6
        report['epsilon_min'] = 0.1
        report['gamma'] = 0.5
        report['alpha'] = 0.2
        agent = agent_builder.build_agent(report, observation_spec, action_spec)
        self.assertEqual(0.1, agent.epsilon)
        self.assertEqual(4, agent.number_of_actions)

    
    def test_save_and_load_sarsa_max_model_1(self):
        action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=3, name='action')
        observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0, name='observation')
        agent_builder = AgentBuilder()
        report = dict()
        report['project_dir'] = 'src/test/models'
        report['project_name'] = 'sarsa_max_model_1'
        report['architecture'] = 'sarsamax'
        report['epsilon'] = 0.1
        report['epsilon_dec'] = 0.6
        report['epsilon_min'] = 0.1
        report['gamma'] = 0.5
        report['alpha'] = 0.2
        agent = agent_builder.build_agent(report, observation_spec, action_spec)
        agent.Q = 2
        agent.save(os.path.join(report['project_dir'], report['project_name']))
        agent.Q = 3
        agent.load(os.path.join(report['project_dir'], report['project_name']))
        #agent_builder.build_agent(report, observation_spec, action_spec)
        self.assertEqual(2, agent.Q)

    def test_save_and_load_sarsa_max_model_1(self):
        action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=3, name='action')
        observation_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=0, name='observation')
        agent_builder = AgentBuilder()
        report = dict()
        report['project_dir'] = 'src/test/models'
        report['project_name'] = 'sarsa_max_model_1'
        report['architecture'] = 'sarsamax'
        report['epsilon'] = 0.1
        report['epsilon_dec'] = 0.6
        report['epsilon_min'] = 0.1
        report['gamma'] = 0.5
        report['alpha'] = 0.2
        agent = agent_builder.build_agent(report, observation_spec, action_spec)
        agent.Q = 4
        agent.save(os.path.join(report['project_dir'], report['project_name']))
        agent2 = agent_builder.build_agent(report, observation_spec, action_spec)
        #agent_builder.build_agent(report, observation_spec, action_spec)
        self.assertEqual(4, agent2.Q)

    
if __name__ == '__main__':
    unittest.main()