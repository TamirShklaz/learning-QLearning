import gym

env = gym.make("MountainCar-v0")
env.reset()

# High state -> desired state
print(env.observation_space.high)

import numpy as np

# Low state
print(env.observation_space.low)

# Number of actions
print(env.action_space.n)

# We need to discretise the observations because they are too many
# So we are creating 20 discrete values from a continous data set
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)  # 20x20

#step size between samples
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

#low is reward of every state
#high is reward for goal state
#Make a 20x20x3 table (position, velocity, action)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# done = False
#
# while not done:
#     action = 2
#     new_state, reward, done, _ = env.step(action)
#     print(new_state)
#     env.render()
#
# env.close()
