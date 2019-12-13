from __future__ import division
from ai_game_env import *
from rl.policy import Policy


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
