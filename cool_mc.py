import argparse
import sys
import os
from src.utilities.project import Project
from src.utilities.monitor import Monitor
from src.utilities.constant_definition_parser import ConstantDefinitionParser
from src.utilities.training import train
from src.interpreter.interpreter import *
from src.rl_agents.agent_builder import *
from src.utilities.logger import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_arguments():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    unknown_args: List[str] = list()
    arg_parser.add_argument('--task', help='What kind of task? - Training Model (training) - RL Model Checking (rl_model_checking) - Model Checking (storm_model_checking)', type=str,
                            default='training')
    arg_parser.add_argument('--prism_dir', help='Folder with all PRISM files', type=str,
                            default='prism_files')
    arg_parser.add_argument('--prism_file_path', help='Prism file path', type=str,
                            default='frozen_lake_4x4.prism')
    arg_parser.add_argument('--project_dir', help='Project folder', type=str,
                            default='projects')
    arg_parser.add_argument('--project_name', help='Project name', type=str,
                            default='test_frozen_lake_4x4_dqn')
    arg_parser.add_argument('--constant_definitions', help='Constant definition for environment (var=VALUE), (var1=VALUE1,var2=VALUE2), or (var1=[START,STEP,END])', type=str,
                            default='slippery=0.04')
    arg_parser.add_argument('--max_steps', help='Maximal number of steps', type=int,
                            default=10000)
    arg_parser.add_argument('--reward_training', help='Normal RL training with property checking', type=bool,
                            default=False)
    arg_parser.add_argument('--reward_flag', help='Rewards (true) or costs (false)', type=bool,
                            default=False)
    arg_parser.add_argument('--wrong_action_penalty', help='Penalty for choosing wrong action', type=int,
                            default=70)
    arg_parser.add_argument('--architecture', help='(Deep) RL Architecture (dqn, reinforce, sarsamax, ...)', type=str,
                            default='dqn')
    arg_parser.add_argument('--replace', help='Replace target network every REPLACE epochs.', type=int,
                            default=103)
    arg_parser.add_argument('--episode_collections', help='Number of episode collections for episodic learning', type=int,
                            default=2)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)
    arg_parser.add_argument('--replay_buffer_size', help='Replay Buffer Size', type=int,
                            default=10000)
    arg_parser.add_argument('--learning_rate', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--noise_scale', help='Noise Scale for Hillclimbing', type=float,
                            default=1e-2)
    arg_parser.add_argument('--epsilon', help='Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decay', type=float,
                            default=0.99999)
    arg_parser.add_argument('--epsilon_min', help='Minimal epislon value', type=float,
                            default=0.4)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=1)
    arg_parser.add_argument('--alpha', help='Alpha', type=float,
                            default=0.6)
    arg_parser.add_argument('--layers', help='Number of neural network layers', type=int,
                            default=3)
    arg_parser.add_argument('--neurons', help='Number of neurons in each layer', type=int,
                            default=512)
    arg_parser.add_argument('--num_episodes', help='Number of training episodes', type=int,
                            default=10000)
    arg_parser.add_argument('--num_supervised_epochs', help='Number of supverised learning epochs', type=int,
                            default=10)
    arg_parser.add_argument('--eval_interval', help='Evaluation each NUMBER of steps', type=int,
                            default=10)
    arg_parser.add_argument('--prop', help='Property Specification', type=str,
                            default='Pmin=? [F "water"]')
    arg_parser.add_argument('--prop_type', help='Maximal Reward (max_reward), Minimal Reward (min_reward), Property minimization (min_prop) or maximization (max_prop).', type=str,
                            default='min_reward')
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--attention_input', help='Input for attention map plotting', type=str,
                            default='x=1,y=2')
    arg_parser.add_argument('--no_gpu', help='GPU for neural network?', type=bool,
                            default=False)
    arg_parser.add_argument('--random_seed', help='Noise Scale for Hillclimbing', type=int,
                            default=0)
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    return vars(args)


if __name__ == '__main__':
    command_line_arguments = get_arguments()
    if command_line_arguments['no_gpu'] == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    project_folder = os.path.join(
        command_line_arguments['project_dir'], command_line_arguments['project_name'])
    command_line_arguments['prop'] = command_line_arguments['prop'].replace(
        '\\', '')   # Cleaning Escape Characters from command line
    formula_str = command_line_arguments['prop']
    project = Project(command_line_arguments)

    cool_mc_logger.info("Init Project", extra={'project':project_folder,'task':command_line_arguments['task']})
    if command_line_arguments['task'] == 'training':
        # Training RL agent
        monitor = Monitor(
            project_folder, command_line_arguments['architecture'])
        train(project, monitor,
              command_line_arguments['num_episodes'], command_line_arguments['eval_interval'])
    elif command_line_arguments['task'] == 'rl_model_checking':
        # Reinforcement Learning Model Checking
        # Check if constant_definitions string contains ranges
        # if no
        if command_line_arguments['constant_definitions'].count(
                '[') == 0 and command_line_arguments['constant_definitions'].count(']') == 0:
            monitor = Monitor(
                project_folder, command_line_arguments['architecture'], training=False)

            prop_result, model_size, model_checking_time, checking_time = project.environment.storm_bridge.model_checker.induced_markov_chain(
                project.agent, project.environment, command_line_arguments['constant_definitions'], formula_str)
            monitor.monitor_model_checking(
                formula_str, prop_result, model_size, model_checking_time)
            project.save_model_checking_result(formula_str, prop_result, True, model_size,
                                               model_checking_time, checking_time, command_line_arguments['constant_definitions'])
        elif command_line_arguments['constant_definitions'].count('[') == 1:
            # if yes
            # For each step make model checking and save results in list
            # Save results to report
            # Plot results
            all_constant_definitions, range_tuple, range_state_variable = ConstantDefinitionParser.parse_constant_definition(
                command_line_arguments['constant_definitions'])
            all_prop_results = []
            all_model_sizes = []
            all_model_checking_times = []
            all_checking_time = []
            monitor = Monitor(
                project_folder, command_line_arguments['architecture'], training=False)
            for constant_definitions in all_constant_definitions:

                prop_result, model_size, model_checking_time, checking_time = project.environment.storm_bridge.model_checker.induced_markov_chain(
                    project.agent, project.environment, constant_definitions, formula_str)
                monitor.monitor_model_checking(
                    formula_str, prop_result, model_size, model_checking_time)
                project.save_model_checking_result(
                    formula_str, prop_result, True, model_size, model_checking_time, checking_time, constant_definitions)
                all_prop_results.append(prop_result)
                all_model_sizes.append(model_size)
                all_model_checking_times.append(model_checking_time)
                all_checking_time.append(checking_time)
            monitor.plot_prop_ranges(
                all_prop_results, range_tuple, range_state_variable, formula_str, project_folder)

        else:
            cool_mc_logger.error('We only support plotting for one state variable.', extra={'project':project_folder,'task':command_line_arguments['task']})
            raise ValueError(
                "We only support plotting for one state variable...")
    elif command_line_arguments['task'] == 'attention_training':
        interpreter = Interpreter(project)
        interpreter.train_autoencoder()
    elif command_line_arguments['task'] == 'attention_mapping':
        interpreter = Interpreter(project)
        interpreter.load_autoencoder()
        interpreter.get_attention_map_for_attention_input_str(
            command_line_arguments['attention_input'])
    elif command_line_arguments['task'] == 'attention_model_checking':
        monitor = Monitor(
            project_folder, command_line_arguments['architecture'], training=False)
        interpreter = Interpreter(project)
        interpreter.load_autoencoder()
        agent = AgentBuilder.build_attention_autoencoder_agent(
            project, interpreter.autoencoder)
        prop_result, model_size, model_checking_time, checking_time = project.environment.storm_bridge.model_checker.induced_markov_chain(
            agent, project.environment, command_line_arguments['constant_definitions'], formula_str)
        monitor.monitor_model_checking(
            formula_str, prop_result, model_size, model_checking_time)
        project.save_model_checking_result(
            formula_str, prop_result, None, model_size, model_checking_time, checking_time, command_line_arguments['constant_definitions'])
    elif command_line_arguments['task'] == 'decision_tree':
        interpreter = Interpreter(project)
        interpreter.decision_tree()
    elif command_line_arguments['task'] == 'dt_model_checking':
        # Decision Tree Model Checking
        monitor = Monitor(
            project_folder, command_line_arguments['architecture'], training=False)
        # Load Decision Tree
        dt_agent = AgentBuilder.build_decision_tree_agent(
            project.project_folder_path, project.environment)
        prop_result, model_size, model_checking_time, checking_time = project.environment.storm_bridge.model_checker.induced_markov_chain(
            dt_agent, project.environment, command_line_arguments['constant_definitions'], formula_str)
        monitor.monitor_model_checking(
            formula_str, prop_result, model_size, model_checking_time)
        project.save_model_checking_result(
            formula_str, prop_result, None, model_size, model_checking_time, checking_time, command_line_arguments['constant_definitions'])
    elif command_line_arguments['task'] == 'plot_rewards' or command_line_arguments['task'] == 'plot_props':
        monitor = Monitor(project_folder, command_line_arguments['architecture'], training=False)
        monitor.plot_training(project.report, command_line_arguments['task'])
    else:
        # Storm Model Checking
        monitor = Monitor(
            project_folder, command_line_arguments['architecture'], training=False)
        prop_result, model_size, model_checking_time, checking_time = project.environment.storm_bridge.model_checker.optimal_checking(
            project.environment, formula_str)
        monitor.monitor_model_checking(
            formula_str, prop_result, model_size, model_checking_time)
        project.save_model_checking_result(formula_str, prop_result, False, model_size,
                                           model_checking_time, checking_time, command_line_arguments['constant_definitions'])
