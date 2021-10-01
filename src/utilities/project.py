import os
import json
from shutil import copyfile
from src.utilities.constant_definition_parser import ConstantDefinitionParser
from src.utilities.data_collector import DataCollector
from src.rl_agents.agent_builder import AgentBuilder
from src.storm_environment.environment import *
import math
class Project():

    def __init__(self, command_line_args):
        '''
        Initialize the current project with the command line arguments.
        It tries to load an already existing project.
        If the project does not exist, it will initialize a new project (report = command_line_args).
        It creates the environment and the agent based in the report dictionary.
        :param command_line_args: dictionary with command line arguments
        '''
        self.project_folder_path = os.path.join(command_line_args['project_dir'], command_line_args['project_name'])
        self.report_file_path = os.path.join( self.project_folder_path, 'report.json')
        self.prism_file_path = os.path.join(self.project_folder_path, 'mdp.prism')
        if os.path.exists(self.project_folder_path):
            self.report = self.__load_report(self.report_file_path)
        else:
            prism_file_path = os.path.join(command_line_args['prism_dir'], command_line_args['prism_file_path'])
            self.report = {}
            self.report.update(command_line_args)
            self.report['return'] = -math.inf
            self.report['prop_result'] = None
            self.report['disabled_features'] = self.__extract_disabled_features(self.report['disabled_features'])
            self.__create_project_folder(self.project_folder_path, prism_file_path, self.report)
        # The environment will be always created based on the command line arguments and the initial disabled features (otherwise problems with the neural network dimensions...)
        self.environment = self.__create_environment(self.prism_file_path, command_line_args, self.report['disabled_features'])
        # The agent is always builded based on the report in the project folder
        self.agent = self.__create_agent(self.report, self.environment.observation_spec(), self.environment.action_spec(), self.environment)
        # Data Collector
        self.data_collector = DataCollector(self.project_folder_path, self.environment.storm_bridge.state_json_example)

    def __extract_disabled_features(self, disabled_feature_string):
        '''
        This method extracts all the features from the disabled_feature_string argument which should be not visible by the RL agent.
        :param disabled_feature_string, string of disabled features (seperated by commatas)
        :return list of disabled features

        '''
        features = disabled_feature_string.split(',')
        for i in range(len(features)):
            features[i] = features[i].strip()
        return features

    def save_report_and_data(self):
        '''
        Save Report and Data
        '''
        self.__save_report(self.project_folder_path, self.report)
        self.data_collector.save()

    def save_agent(self):
        '''
        Save Agent
        '''
        self.agent.save(self.project_folder_path)

    def save_model_checking_result(self, prop_specification, prop_result, rl, model_size, model_checking_time, checking_time, constant_definitions):
        '''
        Saves the model checking result
        :param prop_specification, property specification
        :param prop_result, result of model checking
        :param rl, is reinforcement learning used?
        '''
        if 'model_checking_results' in self.report.keys():
            self.report['model_checking_results'].append([prop_specification, prop_result, rl, model_size, model_checking_time,checking_time, constant_definitions])
        else:
            self.report['model_checking_results'] = ([[prop_specification, prop_result, rl, model_size, model_checking_time, checking_time, constant_definitions]])
        self.__save_report(self.project_folder_path, self.report)
        

        
    def __create_project_folder(self, project_folder_path, prism_file_path, report):
        '''
        This method creates the project folder.
        :param project_folder_path: path to the project folder
        :prism_file_path: path to the prism file path
        :report: dictionary with the report
        '''
        os.mkdir(project_folder_path)
        self.__save_report(self.project_folder_path, report)
        copyfile(prism_file_path, os.path.join(project_folder_path, "mdp.prism"))

    def __save_report(self, project_folder_path, report):
        '''
        This method saves the report into the project folder.
        :param project_folder_path: path to the project folder
        :param report: report as dictionary
        ''' 
        with open( os.path.join(project_folder_path, "report.json"), 'w') as fp:
            json.dump(self.report, fp)

            
    def __load_report(self, report_file_path):
        '''
        This method loads the report as dictionary
        :param report_file_path: path to the report file
        :return report
        '''
        self.report = {}
        with open(report_file_path) as json_file:
            self.report = json.load(json_file)
        return self.report

    def __create_environment(self, prism_file_path, command_line_args, disabled_features):
        '''
        This method creates the environment via a prism file and constant definitions:
        :param prism_file_path: path to the prism file
        :param constant_definitions: constant definitions
        :return StormEnvironment
        '''
        tmp_constant_definition = command_line_args['constant_definitions']
        if command_line_args['constant_definitions'].count('[')==1:
            print(tmp_constant_definition)
            tmp_constant_definition, _, _ = ConstantDefinitionParser.parse_constant_definition(tmp_constant_definition)
            tmp_constant_definition = tmp_constant_definition[0]

        storm_env = StormEnvironment(prism_file_path, tmp_constant_definition, command_line_args['max_steps'],  command_line_args['wrong_action_penalty'], command_line_args['reward_flag'], disabled_features)
        return storm_env

    def __create_agent(self, report, observation_spec, action_spec, environment):
        '''
        This method creates the agent via the hyperparameters from the report
        :param report: dictionary of hyperparameters
        :return Agent
        '''
        return AgentBuilder.build_agent(report, observation_spec, action_spec, environment)
    
    
    
    

    



        