import os
import random
import numpy as np

class Bandit(object):
    def __init__(self):
        self._arms = {}

        # Read distribution from files
        dirfile = os.path.dirname(os.path.abspath(__file__)) + '/distribution'

        for filename in os.listdir(dirfile):
            if not filename.endswith('.txt'):
                continue

            arm = int(filename.split('.')[0])
            l = []
            f = open(dirfile + '/' + filename)

            for line in f:
                # Each file contains a list of returns, keep it and sample from
                # it later
                value = float(line.strip())
                l.append(value)

            self._arms[arm] = l

        # Find what the optimal return is
        means = [np.mean(l) for l in self._arms.values()]
        self._opt = np.max(means)

    def num_arms(self):
        return len(self._arms)
    
    def opt(self):
        return self._opt

    def trigger(self, arm):
        samples = self._arms[arm]

        # Uniformly sample from the distribution of the arm, which will return
        # a reward according to the true distribution of the arm
        return random.choice(samples)
