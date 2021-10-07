'''
Experimental Way to load the trained RL policy into the OpenAI Gym environment.
'''
import gym
from tf_agents.trajectories import time_step as ts
from src.utilities.project import *
from tf_agents.trajectories import PolicyStep

def state_to_json(state):
    json_state = {'done':0}
    if state == 0:
        json_state['x'] = 0
        json_state['y'] = 3
    elif state == 1:
        json_state['x'] = 1
        json_state['y'] = 3
    elif state == 2:
        json_state['x'] = 2
        json_state['y'] = 3
    elif state == 3:
        json_state['x'] = 3
        json_state['y'] = 3
    elif state == 4:
        json_state['x'] = 0
        json_state['y'] = 2
    elif state == 5:
        json_state['x'] = 1
        json_state['y'] = 2
    elif state == 6:
        json_state['x'] = 2
        json_state['y'] = 2
    elif state == 7:
        json_state['x'] = 3
        json_state['y'] = 2
    if state == 8:
        json_state['x'] = 0
        json_state['y'] = 1
    elif state == 9:
        json_state['x'] = 1
        json_state['y'] = 1
    elif state == 10:
        json_state['x'] = 2
        json_state['y'] = 1
    elif state == 11:
        json_state['x'] = 3
        json_state['y'] = 1
    if state == 12:
        json_state['x'] = 0
        json_state['y'] = 0
    elif state == 13:
        json_state['x'] = 1
        json_state['y'] = 0
    elif state == 14:
        json_state['x'] = 2
        json_state['y'] = 0
    elif state == 15:
        json_state['x'] = 3
        json_state['y'] = 0
    return json_state

        

def get_time_step(project, state, reward, done = False):
    json_state = state_to_json(state)
    state = project.environment.storm_bridge.parse_state(str(json_state).replace("'",'"'))
    if done:
        return ts.termination(state, reward)
    else:
        return ts.transition(state, reward=reward, discount=1.0)

def map_action(policy_step):
    action = int(policy_step.action)
    # DOWN(0), LEFT(1), RIGHT(2), UP(3)
    # Gym: LEFT(0), DOWN(1), RIGHT(2), UP(3)
    if action == 0:
        action = 1
    elif action == 1:
        action = 0
    elif action == 2:
        action = 2
    elif action == 3:
        action = 3
    return action



command_line_args = {'architecture':'sarsamax', 'learning_rate':0.001, 'epsilon_dec':0.99999, 'epsilon_min':0.4, 'alpha':0.6, 'batch_size':32, 'epsilon':1,'gamma':0.99, 'replay_buffer_size':10000, 'layers':3,'neurons':64, 'disabled_features' : '', 'prism_dir' : 'prism_files', 'prism_file_path':'frozen_lake_4x4.prism','project_dir':'projects', 'project_name':'example1_vice_versa','constant_definitions':'slippery=0.04','max_steps':10, 'wrong_action_penalty':1000, 'reward_flag':False}
project = Project(command_line_args)
success = False
MAX_ITERATIONS = 20000
env = gym.make('FrozenLake-v0',is_slippery=False)
for epoch in range(0,100000):
    state = env.reset()
    state = get_time_step(project, state, 0)
    #print(epoch)
    #env.render()
    for i in range(MAX_ITERATIONS):
        action = map_action(project.agent.select_action(state, False))
        new_state, reward, done, info = env.step(action)
        #env.render()
        new_state = get_time_step(project, new_state, reward, done)
        project.agent.store_experience(state, PolicyStep(action=tf.constant(action, dtype=tf.int64)), new_state)
        project.agent.step_learn()
        state = new_state
        if done:
            project.save_agent()
            break
    
