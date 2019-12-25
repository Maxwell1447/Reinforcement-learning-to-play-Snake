from __future__ import division
import argparse
from env.ai_game_env import *
from snake import *
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from dqn.dqn import DQNSnake
from dqn.stat_process import stat_process


grid = Grid(10, 10, 10)
env = IAGameEnv(grid)

INPUT_SHAPE = (40, 40)
WINDOW_LENGTH = 2
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--retrain', type=int, default=-1)
parser.add_argument('--step', type=int, default=0)
parser.add_argument('--episodes', type=int, default=5)
parser.add_argument('--initial_eps', type=float, default=0.3)
parser.add_argument('--version', type=str, default="v1")
parser.add_argument('--FPS', type=int, default=25)
args = parser.parse_args()

dqn = DQNSnake(env, input_shape, args.version, args.initial_eps)

if args.mode == 'test':
    if args.weights:
        weights_filename = "data\\"+args.weights
    else:
        weights_filename = 'data\\dqn_snake_weights_{}_{}.h5f'.format(args.version, args.retrain)

    dqn.load_weights(weights_filename)
    env.set_fps(args.FPS)
    history = dqn.test(env, nb_episodes=args.episodes, nb_max_episode_steps=10000, visualize=True, verbose=3)
    stat_process(history)

elif args.mode == 'train':
    if args.weights:
        weights_filename = "data\\"+args.weights
    else:
        weights_filename = 'data\\dqn_snake_weights_{}_{}.h5f'.format(args.version, args.retrain)
    try:
        dqn.load_weights(weights_filename)
    except OSError:
        pass

    new_weights_filename = 'data\\dqn_snake_weights_{}_{}.h5f'.format(args.version, args.retrain+1)
    new_checkpoint_weights_filename = 'data\\dqn_snake_weights_{}_{}.h5f'.format(args.version, "{step}")
    log_filename = 'data\\dqn_snake_log_{}_{}.json'.format(args.version, args.retrain+1)
    callbacks = [ModelIntervalCheckpoint(new_checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    dqn.fit(env, callbacks=callbacks, nb_steps=args.step, log_interval=10000, visualize=False, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(new_weights_filename, overwrite=True)
