import matplotlib.pyplot as plt
import numpy as np
import time
import gym
import torch
from PIL import Image

def main():
    env = gym.make("CarRacing-v2", render_mode='human')
    state, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        gray = np.dot(state[..., :], [0.299, 0.587, 0.114])
        # gray = gray / 128. - 1.
        # gray = gray[0:84, 6:90]
        state = state[0:84, 6:90]
        print(gray.shape)
        plt.imshow(state)
        plt.pause(2)
        if terminated:
            print("Episode is terminated!")
            break
        if truncated:
            print("Episode is truncated!")
            break
        # time.sleep(1)
    # 环境结束
    env.close()


main()


# def preprocess(obs, is_start=False):
#     # First convert to grayscale
#     grayscale = obs.astype('float32').mean(2)
#     # Resize the image to w x h and scale to between 0 and 1
#     s = np.array(Image.fromarray(grayscale).resize((42,42))).astype('float32')*(1.0/255.0)
#     # Next reshape the image to a 4D array with 1st and 4th dimensions of
#     # size 1
#     return s
#
# def test.py():
#     for i in range(4):
#         img = plt.imread("{}.png".format(i+1))
#         ax = plt.subplots(1,2)
#         ax[0]=plt.imshow(img)
#         im = np.array(Image.fromarray(img).resize((96, 96)))
#         ax[1]=plt.imshow(im)
#
#
# import numpy as np
# import time
# import gym
# import torch
# from PIL import Image
# env = gym.make("CarRacing-v2")
# state, _ = env.reset()
# for i in range(1000):
#     action = env.action_space.sample()
#     state, _, terminated, truncated, _ = env.step(action)




