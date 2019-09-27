import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 2000

# the chance that we try a random action
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# We need to discretise the observations because they are too many
# So we are creating 20 discrete values from a continous data set
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)  # 20x20

# step size between samples
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# low is reward of every state
# high is reward for goal state
# Make a 20x20x3 table (position, velocity, action)
# Q(s,a) = V s -> (p,v)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

render = False
for episode in range(EPISODES):
    done = False
    discrete_state = get_discrete_state(env.reset())

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
    while not done:
        # argmax returns the index of the biggest element
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state][action]
            # Reward is always -1 except the end
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state][action] = new_q
        elif new_state[0] >= env.goal_position:
            print("Found on", episode)
            q_table[discrete_state][action] = 0
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
