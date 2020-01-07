from __future__ import division
from env.ai_game_env import *
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from dqn import models
from dqn.snake_processor import SnakeProcessor
from dqn.policy import MaxwellQPolicy


class DQNSnake(DQNAgent):

    def __init__(self, env, input_shape, version, initial_eps):
        # Get the environment and extract the number of actions.
        np.random.seed(123)
        env.seed(123)
        nb_actions = env.action_space.n

        # Next, we build our model. We use the same model that was described by Mnih et al. (2015).

        model = models.model(input_shape, nb_actions, version)

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=1000000, window_length=input_shape[0])
        processor = SnakeProcessor(input_shape[1:])

        # Select a policy. We use eps-greedy action selection, which means that a random action is selected
        # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
        # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
        # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
        # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=initial_eps, value_min=.01,
                                      value_test=0.,
                                      nb_steps=2000000)

        # policy = BoltzmannQPolicy(tau=100)
        test_policy = MaxwellQPolicy(tau=1., eps=.05)
        # The trade-off between exploration and exploitation is difficult and an on-going research topic.
        # If you want, you can experiment with the parameters or use a different policy. Another popular one
        # is Boltzmann-style exploration:
        # policy = BoltzmannQPolicy(tau=1.)
        # Feel free to give it a try!

        super(DQNSnake, self).__init__(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                                       processor=processor, nb_steps_warmup=50000, gamma=.995,
                                       target_model_update=10000,
                                       train_interval=4, delta_clip=1., test_policy=test_policy)
        self.compile(Adam(lr=.00025), metrics=['mae'])

    def test(self, env: IAGameEnv, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        env.visualize = True
        return super().test(env, nb_episodes, action_repetition, callbacks, visualize, nb_max_episode_steps,
                            nb_max_start_steps, start_step_policy, verbose)
