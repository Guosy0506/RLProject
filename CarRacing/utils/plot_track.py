from car_racing import CarRacing
import torch
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(0)

fig = plt.figure()
for times in range(16):
    # seed = torch.randint(0, 100000, (1,)).item()
    seed = 3357
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
    ax = fig.add_subplot(4,4,times+1)
    ax.plot(x, y)
    ax.set_title("{}".format(seed))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
plt.show()

