from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import pygame

from ai_game_env import *
from snake import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as K
import keras.backend.common as common

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy, Policy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

grid = Grid(10, 10, 10)
env = IAGameEnv(grid)

INPUT_SHAPE = (40, 40)
WINDOW_LENGTH = 2


class SnakeProcessor(Processor):

    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch


class MaxwellQPolicy(Policy):
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """

    def __init__(self, tau=1., clip=(-500., 500.), eps=0.3):
        super(MaxwellQPolicy, self).__init__()
        self.tau = tau
        self.eps = eps
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """

        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        max_val = np.max(np.abs(q_values))
        # print("## max value = ", max_val)
        max_clip = max(np.abs(self.clip))
        if max_val > max_clip:
            q_values = q_values / max_val * max_clip

        # print(max_val, " ######## ", q_values)
        nb_of_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            action = np.random.choice(range(nb_of_actions), p=probs)
        else:
            action = np.argmax(q_values)

        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super(MaxwellQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--retrain', type=int, default=0)
parser.add_argument('--step', type=int, default=0)
parser.add_argument('--episodes', type=int, default=5)
parser.add_argument('--initial_eps', type=float, default=0.3)
args = parser.parse_args()

# Get the environment and extract the number of actions.
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

model = Sequential()
if common.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif common.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (5, 5), strides=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
# model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = SnakeProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=args.initial_eps, value_min=.1, value_test=.05,
                              nb_steps=1000000)

# policy = BoltzmannQPolicy(tau=100)
test_policy = MaxwellQPolicy(tau=0.5, eps=.2)
# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.995, target_model_update=10000,
               train_interval=4, delta_clip=1., test_policy=test_policy)
dqn.compile(Adam(lr=.00050), metrics=['mae'])

if args.mode == 'train_':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_snake_weights.h5f'
    checkpoint_weights_filename = 'dqn_snake_weights_{step}.h5f'
    log_filename = 'dqn_snake_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=4000000, log_interval=10000, visualize=False, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    env.set_fps(20)
    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=1, nb_max_episode_steps=10, visualize=True)
    pygame.quit()

elif args.mode == 'test':
    if args.weights:
        weights_filename = args.weights
    else:
        weights_filename = 'dqn_snake_weights_{}.h5f'.format(args.retrain)

    dqn.load_weights(weights_filename)
    env.set_fps(20)
    dqn.test(env, nb_episodes=10, nb_max_episode_steps=500, visualize=True, verbose=3)

elif args.mode == 'train':
    if args.weights:
        weights_filename = args.weights
    else:
        weights_filename = 'dqn_snake_weights_{}.h5f'.format(args.retrain - 1)
    try:
        dqn.load_weights(weights_filename)
    except OSError:
        pass

    new_weights_filename = 'dqn_snake_weights_{}.h5f'.format(args.retrain)
    new_checkpoint_weights_filename = 'dqn_snake_weights_{step}.h5f'
    log_filename = 'dqn_snake_log.json'
    callbacks = [ModelIntervalCheckpoint(new_checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    dqn.fit(env, callbacks=callbacks, nb_steps=args.step, log_interval=10000, visualize=False, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(new_weights_filename, overwrite=True)
