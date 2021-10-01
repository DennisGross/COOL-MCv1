import os
from src.rl_agents.sarsa_max_agent import SarsaMaxAgent
from src.rl_agents.dqn_agent import DQNAgent
from src.rl_agents.hill_climbing_agent import HillClimbingAgent
from src.rl_agents.reinforce_agent import ReinforceAgent
from src.rl_agents.random_agent import RandomAgent
from src.rl_agents.decision_tree_agent import DecisionTreeAgent
from src.rl_agents.attention_autoencoder_agent import *

'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class AgentBuilder():

    @staticmethod
    def build_agent(report, observation_spec, action_spec, environment):
        '''
            IMPORTANT: WE ONLY BUILD AGENTS AND NO POLICIES!
        '''
        agent = None
        state_dimension = int(observation_spec.shape[0])
        number_of_actions = int(action_spec.maximum + 1)
        project_folder = os.path.join(report['project_dir'], report['project_name'])
        # Build Agents (for each agent a new elif-branch)
        if report['architecture'] == 'sarsamax':
            agent = SarsaMaxAgent(number_of_actions, report['epsilon'], report['epsilon_dec'], report['epsilon_min'], report['alpha'], report['gamma'])
            agent.load(project_folder)
        elif report['architecture'] == 'hillclimbing':
            agent = HillClimbingAgent(state_dimension, number_of_actions,  report['gamma'],  report['noise_scale'])
            agent.load(project_folder)
        elif report['architecture'] == 'dqn':
            agent = DQNAgent(number_of_actions, environment.tf_environment, report)
            agent.load(project_folder)
        elif report['architecture'] == 'reinforce':
            agent = ReinforceAgent(environment.tf_environment, report)
            agent.load(project_folder)
        elif report['architecture'] == 'random':
            agent = RandomAgent(number_of_actions)
        else:
            raise NotImplementedError('Architecture is not supported')
        return agent

    @staticmethod
    def build_decision_tree_agent(project_folder, environment):
        agent = DecisionTreeAgent(environment)
        agent.load(project_folder)
        return agent

    @staticmethod
    def build_attention_autoencoder_agent(project, autoencoder):
        agent = AttentionAutoencoderAgent(project, autoencoder)
        return agent