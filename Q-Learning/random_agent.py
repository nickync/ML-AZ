import gym as gym
env = gym.make("FrozenLake-v1")
import numpy as np
import matplotlib.pyplot as plt

np.bool = np.bool_

win_pct = []
scores = []

policy = {0:1, 1:2, 2:1, 3:0, 4:1, 6:1, 8:2, 9:1, 10:1, 13:2, 14:2}

for i in range(1000):
    # while True:
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
        
    #     print(observation, reward, terminated, truncated, info)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    #         break
        
    done = False
    obs = env.reset()
    score = 0
    obs = obs[0]
    while not done:
        #action = env.action_space.sample() # ramdon agent
        action = policy[obs] # some policy
        obs, reward, done,t, info = env.step(action)
        score += reward

    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()
