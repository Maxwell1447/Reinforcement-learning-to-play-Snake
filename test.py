from ai_game_env import *
from snake import *

grid = Grid(20, 20, 20)

env = IAGameEnv(grid)
for i_episode in range(20):
    obs = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if reward != -1:
            print(reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
pyg.quit()
