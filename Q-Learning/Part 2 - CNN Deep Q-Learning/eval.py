import torch
import deep_q_convolutional_pacman as m
import numpy as np
import imageio


state_dic = torch.load('checkpoint.pth')

import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v4', render_mode='rgb_array', full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n

model = m.Network(number_actions)
model.load_state_dict(state_dic)
model.eval()

state, _ = env.reset()
done = False
frames = []
while not done:
    frame = env.render()
    frames.append(frame)
    state = m.preprocess_frame(state)
    actions = model(state)
    action = np.argmax(actions.cpu().data.numpy())
    state, reward, done, _, _ = env.step(action)
env.close()
imageio.mimsave('eval_pac.mp4', frames, fps=30)