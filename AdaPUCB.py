import numpy as np
from tqdm import tqdm
from numba import njit
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
def adap_ucb_single(means, epsilon, T, alpha, test_type):
    """
    Run AdaP-UCB for T steps, return chosen arms[0..T-1].
    
    Parameters:
    - means: True mean rewards for each arm
    - epsilon: Privacy budget (ε)
    - T: Total number of rounds
    - alpha: Algorithm parameter (>3), default=4.0
    
    Returns:
    - Array of chosen arms for each round
    """
    k = len(means)
    n = np.zeros(k, dtype=np.int64)      # Pull counts
    sums = np.zeros(k, dtype=np.float64)  # Cumulative rewards
    arms = np.zeros(T, dtype=np.int64)    # Arm choices
    mu_tilde = np.zeros(k, dtype = np.float64)
    t = 0
    
    # Phase 1: Initialization - play each arm once
    while t < k:
        a = t
        r = sample_reward(means[a], test_type)

        n[a] += 1
        sums[a] += r
        mu_tilde[a] = sums[a] / n[a]
        arms[t] = a
        t += 1
    
    # Phase 2: Adaptive episodes
    while t < T:
        # Start of episode (t_ell = t + 1)
        log_te = np.log(t + 1)  # Natural log of current round
        
        # Compute private index for each arm
        indices = np.zeros(k, dtype=np.float64)
        for i in range(k):
            if n[i] == 0:
                indices[i] = np.inf  # Prioritize unplayed arms
            else:
                exploration = np.sqrt(alpha * log_te / (n[i]))
                privacy_term = (2 * alpha * log_te) / (n[i])
                indices[i] = mu_tilde[i] + exploration + privacy_term
        
        # Select arm with highest index
        A = np.argmax(indices)
        
        # Determine pulls needed to double this arm's count
        pulls_needed = n[A]  # Need to double from current count
        actual_pulls = min(pulls_needed, T - t)
        
        # Pull selected arm 'actual_pulls' times
        for _ in range(actual_pulls):
            r = sample_reward(means[A], test_type)

            n[A] += 1
            sums[A] += r
            mu_hat_A = sums[A] / n[A]
            arms[t] = a
            t += 1
        scale = 2 * alpha * np.log(t) / (epsilon * n[A])
        noise = np.random.laplace(loc=0.0, scale=scale)
        mu_tilde[A] = mu_hat_A + noise
        mu_tilde[A] = clip(mu_tilde[A])
    
    return arms

def simulate_adap_ucb(means, epsilon, T_max, num_trials, alpha, test_type):
    """
    Returns an array where entry t-1 is the average Nash regret at time t,
    averaged over num_trials runs.
    """
    k = len(means)
    mu_star = np.max(means)
    regret_sum = np.zeros(T_max, dtype=np.float64)
    
    for _ in tqdm(range(num_trials), desc = f"AdaP-UCB Trials, epsilon: {epsilon}"):
        arms = adap_ucb_single(means, epsilon, T_max, alpha, test_type)
        rewards = np.array(means)[arms]
        log_rew = np.log(rewards + 1e-12)  # Avoid log(0)
        cum_log = np.cumsum(log_rew)
        ts = np.arange(1, T_max+1)
        geo_mean = np.exp(cum_log / ts)
        regret_curve = mu_star - geo_mean
        regret_sum += regret_curve
    
    avg_regret = regret_sum / num_trials
    return avg_regret
