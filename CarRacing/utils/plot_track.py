from car_racing import CarRacing
import torch
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)

fig = plt.figure()
seeds = [79474, 67463, 50944, 1210, 3606, 95069, 33682, 33866, 79953, 48529, 32886, 64887, 76047, 32673, 8960, 20659, 72706, 59810, 77344, 5106, 1377, 53333, 46133, 50285, 86239]
times = 1
for seed in seeds:
    # seed = torch.randint(0, 100000, (1,)).item()
    print(seed)
    env = CarRacing(render_mode="human")
    env.reset(seed=seed)
    x = []
    y = []
    for i in range(len(env.track)):
        x.append(env.track[i][2])
        y.append(env.track[i][3])
    x.append(x[0])
    y.append(y[0])
    ax = fig.add_subplot(5,5,times)
    ax.plot(x, y)
    ax.scatter(x[0], y[0], s=5, color='red')
    ax.set_title("{}".format(seed))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    times += 1
plt.show()

