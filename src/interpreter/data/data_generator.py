import random
import io
import tensorflow as tf
import pandas as pd
import json
from tf_agents.trajectories import time_step as ts

class DataGenerator:

    def __init__(self, project):
        self.df = pd.read_csv(project.data_collector.csv_file_path)
        self.features = []
        for feature in self.df.columns:
            self.features.append(feature)
        self.project = project

    def __row_to_time_step(self, row, features):
        state_dict = {}
        # row to json
        for feature in features:
            state_dict[feature] = int(row[feature])
        # dict to json
        json_state = json.dumps(state_dict)
        # json to state
        state = self.project.environment.storm_bridge.parse_state(json_state)
        # state to time_step
        return ts.transition(state, reward=0, discount=1.0)


    def label_each_row_with_rl_agent_action(self):
        actions = []
        for index, row in self.df.iterrows():
            # Create TimeStep
            time_step = self.__row_to_time_step(row, self.features)
            # Get rl agent action
            action_step = self.project.agent.select_action(time_step, True)
            # Action to action label
            action_name = self.project.environment.action_mapper.action_index_to_action_name(action_step.action)
            # Action to actions
            actions.append(action_name)

        # Add List to Data Frame as column
        self.df['action'] = actions

        # Print head
        print(self.df.head(100))

    def generate_dataset(self, destination_path):
        f = open(destination_path, 'w')
        for index, row in self.df.iterrows():
            tmp_str = ''
            for feature in self.features:
                tmp_str += feature + str(int(row[feature])) + ' '
            tmp_str = tmp_str.strip()
            tmp_str += '\t'
            tmp_str += row['action']
            tmp_str += '\n'
            f.write(tmp_str)
        f.close()




    

