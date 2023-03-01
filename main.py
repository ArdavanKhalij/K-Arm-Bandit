#!/usr/bin/python3
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

import bandit

def main():
    # timesteps = int(sys.argv[1])
    timesteps = 10000
    b = bandit.Bandit()

    regret = 0.
    ret = 0
    rewards = []
    rewards_for_plot = []
    rewards2 = []
    rewards_for_plot2 = []
    regrets = []

    # exploration probability
    epsilon = 0
    decay = 1 / (timesteps - timesteps/20)
    epsilon2 = 2
    decay2 = 0.9999

    for i in range(b.num_arms()):
        rewards.append([0])
        rewards2.append([0])

    for t in range(timesteps):
        # Choose an arm
        # a = 0
        a = random.randrange(b.num_arms())
        if np.random.random() > epsilon:
            rewards[a].append(b.trigger(a))
            ret = random.choice(rewards[a])
        else:
            ret = max(rewards[a])

        # Pull the arm, obtain a reward

        rew_plot = 0
        for i in range(len(rewards)):
            rewards[i] = [*set(rewards[i])]
        for i in rewards:
            rew_plot = rew_plot + (sum(i) / len(i))

        rewards_for_plot.append((rew_plot / b.num_arms()))

        regret += b.opt() - (rew_plot / b.num_arms())
        regrets.append(regret)

        # Learn from a and ret
        epsilon = min([1, epsilon + decay])
        if (t % 1000) == 0:
            print(t / 1000, "/", timesteps / 1000)
            print('Reward', ret, 'regret', regret)
            print(epsilon)
            print("\n\n\n")

    for t in range(timesteps):
        # Choose an arm
        # a = 0
        a = random.randrange(b.num_arms())
        if np.random.random() < epsilon2:
            rewards2[a].append(b.trigger(a))
            ret = random.choice(rewards2[a])
        else:
            ret = max(rewards2[a])

        # Pull the arm, obtain a reward
        rew_plot = 0
        for i in range(len(rewards2)):
            rewards2[i] = [*set(rewards2[i])]
        for i in rewards2:
            rew_plot = rew_plot + (sum(i) / len(i))

        rewards_for_plot2.append((rew_plot / b.num_arms()))

        regret += b.opt() - (rew_plot / b.num_arms())
        regrets.append(regret)

        # Learn from a and ret
        epsilon2 = min([1, epsilon2 * decay2])
        if (t % 1000) == 0:
            print(t / 1000, "/", timesteps / 1000)
            print('Reward', ret, 'regret', regret)
            print(epsilon2)
            print("\n\n\n")
        # continue
    # plt.plot(regrets, label="Regrets")
    plt.plot(rewards_for_plot, label="Rewards Average with linear decay")
    plt.plot(rewards_for_plot2, label="Rewards Average with non-linear decay")
    plt.xlabel('Time')
    plt.ylabel('Reward Average and Regrets')
    plt.title('Learning Curve for ' + str(timesteps) + " timesteps")
    leg = plt.legend();
    plt.show()

if __name__ == '__main__':
    main()
