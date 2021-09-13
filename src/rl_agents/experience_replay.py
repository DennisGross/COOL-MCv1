import tensorflow as tf
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

class ExperienceReplay:

    def __init__(self, tf_env, agent, batch_size, replay_buffer_max_length):
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_max_length)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=batch_size, 
            num_steps=2).prefetch(3)
        self.iterator = iter(dataset)

    def episodic_replay(self):
        experience = self.replay_buffer.gather_all()
        self.replay_buffer.clear()
        return experience

    def step_replay(self):
        return next(self.iterator)

        

    def store_experience(self, time_step, action_step, next_time_step):
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

