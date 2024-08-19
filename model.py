import numpy as np
import envi
import time
import random

action_space = [0,1]
env = envi.environment()
def decay_schedule(
            init_value, min_value,
            decay_ratio, max_steps,
            log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(
        log_start, 0, decay_steps,
        base=log_base, endpoint=True)[::-1]    
    values = (values - values.min()) / \
                        (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values

def Q_learning(env,
    gamma=1,
    init_alpha=1,
    min_alpha=0.1,
    alpha_decay_ratio=0.2,
    init_epsilon=1,
    min_epsilon=.3,
    epsilon_decay_ratio=0.2,
    n_episodes=3000, dtype=np.float64):

    nS = 3**(2*len(env.observable_space) +1)
    nA = len(action_space)
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    pi_track = []
    select_action = lambda state, Q, epsilon: \
                            np.argmax(Q[state]) \
                            if np.random.random() > epsilon \
                            else np.random.randint(len(Q[state]))
    alphas = decay_schedule(
        init_alpha, min_alpha,
        alpha_decay_ratio,
        n_episodes)
    
    epsilons = decay_schedule(
        init_epsilon, min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    for e in range(n_episodes):
        state, done = env.reset()
        #action = select_action(state, Q, epsilons[e])
        while not done:
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done = env.step(action)

            td_target = reward + gamma*\
                        np.max(Q[next_state]) * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + \
                    alphas[e] * td_error
            state = next_state
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q, axis=1)[s]

    return Q, V, pi, Q_track, pi_track

#Q , V , pi , Q_track, pi_track = sarsa(env)
#print(pi_track)