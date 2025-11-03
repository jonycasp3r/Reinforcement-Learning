import time
import kuimaze
import os
import random
import gym
import numpy as np

# Function to find the best action based on Q-values (highest reward)
def best_action(env, place, x, y):
    best_reward = float('-inf')
    for move in range(env.action_space.n):
        actual_reward = place[x][y][move]
        if actual_reward > best_reward:
            best_reward = actual_reward
            best_move = move
    return best_move

# Function to check if a state has any unexplored actions (Q-value == 0 means unexplored)
def spotted(env, place, x, y):
    actions = list(range(env.action_space.n))
    random.shuffle(actions)  # shuffle actions to explore in random order
    for index in actions:
        if place[x][y][index] == 0:  # if Q-value of an action is 0, it hasn’t been tried yet
            return index
    return "spotted"  # all actions have been explored

# Main function to learn a policy using Q-learning
def learn_policy(env):
    discount = 0.6      # discount factor (how much future rewards matter)
    epsilon = 1         # exploration factor (not used here, but common in epsilon-greedy methods)
    alpha = 0.6         # learning rate
    x_dims = env.observation_space.spaces[0].n  # size of the X dimension of the environment
    y_dims = env.observation_space.spaces[1].n  # size of the Y dimension of the environment
    real_time = 0
    max_time = 19       # maximum training time in seconds
    start_time = time.time()

    maze_size = tuple((x_dims, y_dims))
    num_actions = env.action_space.n
    # Initialize Q-table with zeros
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)

    # Training loop runs until max_time is reached
    while max_time > real_time:
        state = env.reset()
        while True:
            # Choose action: explore new moves first, otherwise random
            if spotted(env, q_table, state[0], state[1]) != "spotted":
                action = spotted(env, q_table, state[0], state[1])
            else:
                action = env.action_space.sample()

            # Take the action and observe the result
            new_obv, reward, done, _ = env.step(action)
            new_state = new_obv[0:2]

            # Determine the best possible action from the new state
            new_action = best_action(env, q_table, new_state[0], new_state[1])

            # Calculate value update (Q-learning rule)
            value = discount * q_table[new_state[0]][new_state[1]][new_action] + reward
            delta = value - q_table[state[0]][state[1]][action]
            q_table[state[0]][state[1]][action] += delta * alpha

            # If the episode ends, break out of the loop
            if done == 1:
                break

            # Update elapsed time
            real_time = time.time() - start_time
            state = new_state

            # Stop training if time limit is reached
            if max_time <= real_time:
                break

    # Derive the final policy from the learned Q-table
    policy = dict()
    for k in range(x_dims):
        for l in range(y_dims):
            pol = np.where(q_table[k][l] == np.amax(q_table[k][l]))
            poli = pol[0][0]
            policy[k, l] = poli

    return policy
