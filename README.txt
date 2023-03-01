Assignment 1: Bandits

This directory contains the implementation of a bandit (bandit.py), and the skeleton of a bandit learning program (main.py). The goal of this assignment is to implement learning rules in main.py so that the agent learns the bandit. The second goal is to learn fast: having the highest learning curve, or lowest cumulative regret, among the students doing this exercise, gives bonus points.

The bandit
==========

For the agent, the bandit is just a set of arms that give outcomes. You are not allowed to use any additional knowledge you have on the bandit in your solution. Your algorithm must be general. However, for information, here are details about the bandit:

- The bandit represents what happens in a big population of potentially-sick individuals when vaccination strategies are implemented. Pulling one arm maps, in the real world, to the infection of a couple of people, and the assignation of vaccines to specific age groups. The reward given by the arm is computed from the total amount of people who became sick in 2 or 3 years. You can see why sample-efficiency is important here: in the real world, those kinds of simulations (pulling a single arm) takes about 1 hour on a supercomputer. For this assignment, thousands of outcomes for every arm have been generated and kept in the distributions/ directory, so that your computer does not have to run hot.
- The outcome of each arm is bi-modal (but, remember, you cannot use this information to tune your algorithm). This means that either the epidemic dies out (with a small probability), or it takes over the population. Some fancy work done at the AI lab uses this hypothesis to learn the bandit a bit faster.
