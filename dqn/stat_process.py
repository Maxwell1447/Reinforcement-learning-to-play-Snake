from keras.callbacks import History
import numpy as np


def stat_process(history: History):

    rewards = np.array(history.history["episode_reward"])
    steps = np.array(history.history["nb_steps"])

    print("rewards stats", get_stats(rewards))
    print("steps stats", get_stats(steps))


def get_stats(array: np.ndarray):

    stats = {}
    stats["min"] = array.min()
    stats["max"] = array.max()
    stats["mean"] = array.mean()
    stats["std"] = array.std()
    stats["25% percentile"] = np.percentile(array, 25)
    stats["median"] = np.percentile(array, 50)
    stats["75% percentile"] = np.percentile(array, 75)

    return stats
