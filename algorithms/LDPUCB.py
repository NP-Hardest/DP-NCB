import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

@njit
def clip(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x
    

@njit
def sample_reward(mean, test_type):
    """
    Sample a reward according to the arm's mean category,
    implemented in pure nopython with only random.random().
    """
    u = np.random.random()  # numba supports this in nopython
    if test_type == 0:
        return 1.0 if u < mean else 0.0
    
    else:
        if mean >= 0.75:
            # Bernoulli(mean)
            return 1.0 if u < mean else 0.0

        elif mean >= 0.5 and mean < 0.75:
            # Beta(4,1) via inverse CDF: U^(1/4)
            # clip just in case of tiny numerical overflows:
            x = u ** 0.25
            return 1.0 if x > 1.0 else (0.0 if x < 0.0 else x)

        elif mean >= 0.25 and mean < 0.5:
            # two‑point uniform {0.4, 1.0}
            return 0.4 if u < 0.5 else 1.0

        elif mean < 0.25:
            # continuous uniform [0,1]
            return u

        else:
            # fallback to Bernoulli
            return 1.0 if u < mean else 0.0



@njit
def LDP_UCB_L_single(means, epsilon, T, test_type):
    k = len(means)
    n = np.ones(k, dtype=np.int64)  # already pulled once
    sums = np.zeros(k, dtype=np.float64)
    mu = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)

    # Initial pull of each arm once (step 1)
    for a in range(k):
        r = sample_reward(means[a], test_type)
        noise = np.random.laplace(loc=0.0, scale=1/epsilon)
        r += noise
        r = clip(r)
        sums[a] += r
        mu[a] = r  # since n[a] = 1
        arms[a] = a

    t = k  # time starts after initialization

    while t < T:
        logt = np.log(t + 1)
        ucb = np.zeros(k)
        forced_pull = -1

        for a in range(k):
            if n[a] <= 4 * logt:
                forced_pull = a
                break  # must pull this arm due to exploration requirement

        if forced_pull != -1:
            A = forced_pull
        else:
            for a in range(k):
                bonus1 = np.sqrt((2 * logt) / n[a])
                bonus2 = np.sqrt((32 * logt)) / (epsilon * np.sqrt(n[a]))
                ucb[a] = mu[a] + bonus1 + bonus2
            A = np.argmax(ucb)

        r = sample_reward(means[A], test_type)
        noise = np.random.laplace(loc=0.0, scale=1/epsilon)
        r += noise
        r = clip(r)

        n[A] += 1
        sums[A] += r
        mu[A] = sums[A] / n[A]
        arms[t] = A
        t += 1

    return arms
def simulate_LDP_UCB_L(means, T_max, epsilon, num_trials, test_type):
    mu_star = np.max(means)
    total_rewards = []

    for _ in tqdm(range(num_trials), desc="LDP-UCB-L Trials"):
        arms = LDP_UCB_L_single(means, epsilon, T_max, test_type)
        rewards = np.array(means)[arms]
        total_rewards.append(rewards)

    total_rewards = np.array(total_rewards)
    expected_means = np.sum(total_rewards, axis=0) / num_trials

    cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-300)))
    inv_t = 1.0 / np.arange(1, T_max + 1)
    geom_mean = np.exp(cumsum_log * inv_t)
    avg_regret = mu_star - geom_mean
    return avg_regret

    # return avg_regret[len(avg_regret)-1]
