import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        if mean == 0.9 or mean == 0.6:
            # Bernoulli(mean)
            return 1.0 if u < mean else 0.0

        elif mean == 0.8:
            # Beta(4,1) via inverse CDF: U^(1/4)
            # clip just in case of tiny numerical overflows:
            x = u ** 0.25
            return 1.0 if x > 1.0 else (0.0 if x < 0.0 else x)

        elif mean == 0.7:
            # two‑point uniform {0.4, 1.0}
            return 0.4 if u < 0.5 else 1.0

        elif mean == 0.5:
            # continuous uniform [0,1]
            return u

        else:
            # fallback to Bernoulli
            return 1.0 if u < mean else 0.0



@njit
def NCB_single(means, c, T, test_type):
    """Run NCB for T steps, return chosen arms[0..T-1]."""
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)

    logT = np.log(T)
    t = 0

    # Phase 1: uniform exploration
    while t < T and np.max(n * mu) <= (c**2) * logT:
        a = np.random.randint(0, k)
        # r = np.random.random() < means[a] 
        r = sample_reward(means[a], test_type)

        n[a] += 1
        sums[a] += r
        mu[a] = sums[a] / n[a]
        arms[t] = a
        t += 1

    # Phase 2: NCB selection
    while t < T:
        bonus = np.empty(k, dtype=np.float64)
        for i in range(k):
            if n[i] == 0:
                bonus[i] = 1e12
            else:
                bonus[i] = 2 * c * np.sqrt((2 * mu[i] * logT) / n[i])
        ncb = mu + bonus
        A = np.argmax(ncb)
        r = sample_reward(means[A], test_type)


        n[A] += 1
        sums[A] += r
        mu[A] = sums[A] / n[A]
        arms[t] = A
        t += 1

    return arms

def simulate_ncb(means, T_max, num_trials, c, test_type):
    """
    Returns an array of length T_max where entry t-1 is the average Nash regret at time t,
    averaged over num_trials independent runs.
    """
    k = len(means)
    mu_star = np.max(means)
    # accumulator for sum of regret curves
    regret_sum = np.zeros(T_max, dtype=np.float64)

    for _ in tqdm(range(num_trials), desc=f"NCB Non-Private Trials"):
        arms = NCB_single(np.array(means), c, T_max, test_type)
        rewards = np.array(means)[arms]                # length T_max
        log_rew = np.log(rewards + 1e-12)               # avoid log(0)
        cum_log = np.cumsum(log_rew)                    # cum sum of logs
        ts = np.arange(1, T_max+1)
        geo_mean = np.exp(cum_log / ts)                 # geometric mean up to each t
        regret_curve = mu_star - geo_mean               # instantaneous Nash regret
        regret_sum += regret_curve

    avg_regret = regret_sum / num_trials
    return avg_regret