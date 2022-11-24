import matplotlib.pyplot as plt
import numpy as np


def main():
    record = np.load("D:/git/project/CarRacing/PPOinv2/training_records.npy")
    i_ep = record[:, 0]
    score = record[:, 1]
    running_score = record[:, 2]
    seed = record[:, 3]
    # plt.subplot(121)
    # plt.plot(i_ep,running_score)
    # plt.subplot(122)
    s = [1 for n in range(len(i_ep))]
    plt.figure(figsize=(12, 16))
    plt.scatter(i_ep, score, s=s)
    plt.plot(i_ep, running_score, color='k')
    plt.legend(['reward in each episode', 'average reward'])
    plt.title('PPO in CarRacing-v2')
    plt.xlabel('episode')
    plt.show()


if __name__ == "__main__":
    main()
