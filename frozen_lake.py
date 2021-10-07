'''
Experimental Way to load the trained RL policy into the OpenAI Gym environment.
'''
import gym
from tf_agents.trajectories import time_step as ts
from src.utilities.project import *

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

        

def get_time_step(project, state):
    json_state = state_to_json(state)
    state = project.environment.storm_bridge.parse_state(str(json_state).replace("'",'"'))
    return ts.transition(state, reward=0, discount=0)

def map_action(policy_step):
    action = policy_step.action
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



command_line_args = {'project_dir':'projects', 'project_name':'example1','constant_definitions':'slippery=0.04','max_steps':10, 'wrong_action_penalty':1000, 'reward_flag':False}
project = Project(command_line_args)
success = False
MAX_ITERATIONS = 1000
env = gym.make('FrozenLake-v0',is_slippery=False)
while success == False:
    state = env.reset()
    state = get_time_step(project, state)
    env.render()
    for i in range(MAX_ITERATIONS):
        action = map_action(project.agent.select_action(state, True))
        new_state, reward, done, info = env.step(action)
        env.render()
        state = get_time_step(project, new_state)
        if done:
            print(state.observation[1], state.observation[2])
            success = True
            if state.observation[1] == 3 and state.observation[1] == 0:
                success = True
            break
    
    
