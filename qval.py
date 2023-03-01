#!/usr/bin/python3
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

import bandit

def main():
    # timesteps = int(sys.argv[1])
    timesteps = 1000
    b = bandit.Bandit()
    AverageEvery = 100

    regret = 0.
    rewards_for_plot = []
    regrets = []

    # exploration probability
    epsilon = 0.9
    decay = 1
    # decay = 0.999
    min_epsilon = 0.1

    QVal = [0] * b.num_arms()
    ActionCount = [0] * b.num_arms()

    Rewards = [0]
    k = []
    delta = 0.00001
    bandits_p = []

    policy = []
    for i in range(b.num_arms()):
        policy.append(1/b.num_arms())
        bandits_p.append([])

    for t in range(timesteps):
        # Choose an arm
        a = 0
        if np.random.random() <= epsilon:
            p = []
            for i in QVal:
                if sum(QVal) == 0:
                    for j in range(b.num_arms()):
                        p.append(1/b.num_arms())
                    break
                p.append(i/sum(QVal))
            a = np.random.choice(list(range(0, b.num_arms())), p=p)
            a = random.randrange(b.num_arms())
        else:
            a = QVal.index(max(QVal))

        if policy[a] <= (1 - delta):
            policy[a] = policy[a] + delta
            for i in range(b.num_arms()):
                if i != a:
                    if policy[i] >= (delta/(b.num_arms() - 1)):
                        policy[i] = policy[i] - (delta/(b.num_arms() - 1))

        for i in range(b.num_arms()):
            bandits_p[i].append(policy[i])

        # Pull the arm, obtain a reward
        rew = b.trigger(a)
        ActionCount[a] = ActionCount[a] + 1

        # Update the Q-Value
        QVal[a] = QVal[a] = QVal[a] + ((1 / ActionCount[a]) * (rew - QVal[a]))

        # Save the mean Reward
        rewards_for_plot.append(rew)

        regret += b.opt() - rew
        regrets.append(regret)

        # Learn from a and ret
        epsilon = max([min_epsilon, epsilon * decay])
        if (t % 1000) == 0:
            print(str(t / (timesteps/100)) + " from " + str(timesteps / (timesteps/100)) + ' Reward', rew, 'regret', regret, "Action", a+1, "epsilon", epsilon)

    # plt.plot(regrets, label="Regrets")
    # rew_plot_average = []
    # k = []
    print(QVal)
    # for i in range(len(rewards_for_plot)):
    #     k.append(rewards_for_plot[i])
    #     if (i % AverageEvery) == 0:
    #         rew_plot_average.append(sum(k)/len(k))
    #         k = []
    # plt.plot(rew_plot_average, label="Rewards Average with linear decay")
    # # plt.plot(rewards_for_plot2, label="Rewards Average with non-linear decay")
    # plt.xlabel('Time')
    # plt.ylabel('Reward Average and Regrets')
    # plt.title('Learning Curve for ' + str(timesteps) + " timesteps and average every " + str(AverageEvery) + " timesteps")
    # leg = plt.legend();
    # plt.show()
    for i in range(b.num_arms()):
        plt.plot(bandits_p[i], label="Arm " + str(i))
    # plt.plot(rewards_for_plot2, label="Rewards Average with non-linear decay")
    plt.xlabel('Time')
    plt.ylabel('Possibility of Arm')
    plt.title('Possibility of each Arm ' + str(timesteps) + " timesteps")
    leg = plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
