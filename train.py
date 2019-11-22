from ai_game_env import *
from snake import *
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Input
from keras.layers import MaxPooling2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

grid = Grid(20, 20, 20)
env = IAGameEnv(grid)
np.random.seed(123)
env.seed(123)

print(env.observation_space.shape)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=20, strides=(4, 4), padding="valid",
                 input_shape=env.observation_space.shape, data_format="channels_last"))
# model.add(Conv2D(16, 20, input_shape=env.observation_space.shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(env.action_space.n))

print(model.summary())
print(env.observation_space.shape)

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
env.set_fps(10)
dqn.test(env, nb_episodes=5, visualize=True)
