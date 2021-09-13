import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.utils import common
from src.rl_agents.agent import Agent, to_tuple
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from src.rl_agents.experience_replay import ExperienceReplay
import os

        

    


class ReinforceAgent(Agent):

    def __init__(self, tf_env, report):
        super().__init__(True)
        fc_layer_params = to_tuple(report['layers'], report['neurons'])
        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fc_layer_params)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.agent = reinforce_agent.ReinforceAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            actor_network=self.actor_net,
            optimizer=self.optimizer,
            gamma=report['gamma'],
            normalize_returns=True)
        self.agent.initialize()
        self.experience_replay =  ExperienceReplay(tf_env, self.agent, report['batch_size'], report['replay_buffer_size'])
        self.episode_counter = 0
        self.episode_counter_interval = report['episode_collections']

    def select_action(self, time_step, deploy):
        # Convert time_step to tensor
        if deploy:
            action_step = self.agent.policy.action(time_step)
        else:
            action_step = self.agent.collect_policy.action(time_step)
        return action_step

    def store_experience(self, time_step, action_step, n_time_step):
        # Convert all parameters to tensors
        self.experience_replay.store_experience(time_step, action_step, n_time_step)


    def episodic_learn(self):
        self.episode_counter += 1
        if self.episode_counter % self.episode_counter_interval == 0 and self.episode_counter != 0:
            experience = self.experience_replay.episodic_replay()
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