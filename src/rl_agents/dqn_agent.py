import tensorflow as tf
from src.rl_agents.agent import Agent, to_tuple
from src.rl_agents.experience_replay import ExperienceReplay
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import PolicyStep
import os

def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

class DQNAgent(Agent):

    def __init__(self, number_of_actions, tf_env, report):
        super().__init__(True)
        fc_layer_params = to_tuple(report['layers'], report['neurons'])
        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            number_of_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        self.q_net = sequential.Sequential(dense_layers + [q_values_layer])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=report['learning_rate'])
        self.sample_counter = 0
        self.batch_size = report['batch_size']
        train_step_counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,epsilon_greedy=report['epsilon'])

        self.agent.initialize()
        self.experience_replay =  ExperienceReplay(tf_env, self.agent, report['batch_size'], report['replay_buffer_size'])

    def select_action(self, time_step, deploy):
        # Convert time_step to tensor
        if deploy:
            # Reshaping
            a = time_step.observation
            a = a.reshape(1,a.shape[0])
            time_step = ts.transition(tf.constant(a), reward=0, discount=1.0)
            action_step = self.agent.policy.action(time_step)
            action_step = PolicyStep(action=int(action_step.action))
        else:
            action_step = self.agent.collect_policy.action(time_step)
        return action_step

    def store_experience(self, time_step, action_step, n_time_step):
        # Convert all parameters to tensors
        self.experience_replay.store_experience(time_step, action_step, n_time_step)


    def step_learn(self):
        if self.sample_counter<self.batch_size:
            self.sample_counter+=1
            return
        else:
            experience, unused_info = self.experience_replay.step_replay()
            self.agent.train(experience)

    def get_hyperparameters(self):
        pass

    def save(self, root_folder):
        self.train_checkpointer.save(0)

    def load(self, root_folder):
        checkpoint_dir = os.path.join(root_folder, 'checkpoint')
        self.train_checkpointer = common.Checkpointer(
                ckpt_dir=checkpoint_dir,
                max_to_keep=1,
                agent=self.agent,
                policy=self.agent.policy,
                replay_buffer=self.experience_replay.replay_buffer,
            )
        self.train_checkpointer.initialize_or_restore()