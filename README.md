Implement the learn_policy (env) method in a rl_agent.py. env is of type HardMaze this time. The expected output is policy, dictionary keyed states, values ​​can be [0,1,2,3] which corresponds to up, right, down, left (N, E, S, W). The learning limit on one tile is 20 seconds. Be sure to turn off visualizations before submitting, see VERBOSITY in rl_sandbox.py.

Again, we will use the cubic environment. Visualization methods are the same, as well as initialization, but the basic idea of working with the environment is different. We do not have a map and we can explore the environment using the main method env.step (action). The environment-simulator knows what the current state is. We are looking for the best way from start to finish. We want a trip with the highest expected sum of discounted rewards.


obv, reward, done, _ = env.step(action)
state = obv[0:2]

You can get the action by random selection: action = env.action_space.sample()
The package includes rl_sandbox.py, where you can see basic random browsing, possibly initializing the Q values ​​table, visualization, and so on.
