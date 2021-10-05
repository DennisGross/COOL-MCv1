import datetime
import sys
import os
import tensorflow as tf
from tf_agents.utils import common
import matplotlib.pyplot as plt
import numpy as np
import json
import os
class Monitor:

    def __init__(self, root_dir, architecture, training=True):
        '''
        Init the monitor.
        :param root_dir, for all monitor logs 
        :param training, true if monitoring training
        '''
        if training:
            current_time = datetime.datetime.now().strftime(architecture+"%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(root_dir, 'logs/' + current_time) 
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    def monitor(self, episode, average_return, best_return, prop_result):
        '''
        This method is the only function with std-output.
        It monitors the training process.
        :param episode, current episode
        :param average_return, average return
        :param best_return, best return
        :param prop_result, propert specification result from model checking
        '''
        # TensorBoard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('return', average_return, step=episode)
            if prop_result!=None:
                tf.summary.scalar('prop_result', prop_result, step=episode)
        #STDOUT
        #sys.stdout.flush()
        print("\rEpisode: {}\t|| Last Property Result: {}\t|| Best Return: {}\t|| Average Return: {}".format(episode, prop_result, best_return, average_return), end="")

    def monitor_model_checking(self, formula_str, prop_result, model_size, model_checking_time):
         print("\rProperty Result: {} with Model Size: {} checked in {} seconds\n".format(prop_result, model_size, model_checking_time), end="")

    def plot_prop_ranges(self,prop_results, range_tuple, xlabel, formula_str, project_folder):


        # Data for plotting
        x = np.arange(range_tuple[0], range_tuple[2], range_tuple[1])
        y = np.array(prop_results)

        fig, ax = plt.subplots()
        ax.plot(x,y)

        ax.set(xlabel=xlabel, ylabel=formula_str,
            title='Properties')
        ax.grid()

        fig.savefig(os.path.join(project_folder, "properties.png"))
        plt.show()

    def plot_training(self, report, task):
        y1 = None
        file_name = ''
        if task=='plot_props':
            y1 = report['prop_results']
            plt.ylabel(report['prop'])
            file_name =  'training_props.png'
        else:
            y1 = report['rewards']
            plt.ylabel('Rewards')
            file_name =  'training_rewards.png'
        x = list(range(1,len(y1)+1))
        plt.title('RL Training Progress')
        plt.xlabel("Epochs x " + str(report['eval_interval']))
        
        plt.plot(x, y1)
        
        plt.savefig(os.path.join(report['project_dir'], report['project_name'], file_name))
        plt.show()

    



