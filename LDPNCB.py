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
def arm_thresholds(mu, n, epsilon, alpha, logT, c):
    # compute the denom: (μ - (1/ε)*sqrt(8α logT / n))^2 ε^2
    # note: only for n>0 and μ large enough; else set threshold=∞ so we stop immediately
    sqrt_term = np.zeros_like(mu)
    safe_n = np.maximum(n, 1)
    sqrt_term[n>0] = (1/epsilon)*np.sqrt((8*alpha*logT)/safe_n[n>0])
    diff = mu - sqrt_term
    # if diff<=0, we force a tiny positive number so denom->∞ and paper‐term->∞
    diff = np.maximum(diff, 1e-20)
    denom = (diff) * (epsilon**2)

    paper_term = (logT**2) / denom
    return 1600 * (c**2 * logT + paper_term) \
            + np.sqrt(8 * n * alpha * logT) / epsilon    


@njit
def LDP_NCB_single(means, c, T, W, epsilon, alpha, test_type):
    """Run NCB for T steps, return chosen arms[0..T-1]."""
    k = len(means)
    n = np.zeros(k, dtype=np.int64)
    sums = np.zeros(k, dtype=np.float64)
    mu = np.zeros(k, dtype=np.float64)
    arms = np.empty(T, dtype=np.int64)

    logT = np.log(T)
    logW = np.log(W)
    t = 0

    # Phase 1: uniform exploration

    # --- Phase I: uniform exploration until ANY arm breaks the condition ---
    while t <= W:
        lhs = n * mu
        rhs = arm_thresholds(mu, n, epsilon, alpha, logT, c)
        if not np.all(lhs <= rhs):
            break

        # otherwise, keep exploring uniformly
        a = np.random.randint(k)
        r = sample_reward(means[a], test_type)
        scale = 1 / (epsilon)
        noise = np.random.laplace(scale)
        r += noise
        n[a] += 1
        sums[a] += r
        mu[a] = sums[a] / n[a]
        mu[a] = clip(mu[a])
        arms[t] = a
        t += 1


    # Phase 2: NCB selection
    while t < T:
        bonus = np.empty(k, dtype=np.float64)
        for i in range(k):
            if n[i] == 0:
                bonus[i] = 1e12
            else:
                b1 = 2.0 * c * np.sqrt((2.0 * mu[i] * logT) / n[i])
                b2 = (1.0 / epsilon) * np.sqrt((8.0 * alpha * logT) / n[i])
                b3 = (4.0 * c 
                    * np.power(2.0 * alpha, 0.25) 
                    * np.power(logW,     0.75)
                    ) / (np.sqrt(epsilon) * np.power(n[i], 0.75))
                bonus[i] = b1 + b2 + b3
        ncb = mu + bonus
        A = np.argmax(ncb)
        r = sample_reward(means[A], test_type)
        scale = 1 / (epsilon)
        noise = np.random.laplace(loc=0.0, scale=scale)
        r += noise

        n[A] += 1
        sums[A] += r
        mu[A] = sums[A] / n[A]
        mu[A] = clip(mu[A])
        arms[t] = A
        t += 1

    return arms


def simulate_LDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type):
    """
    Returns an array of length T_max where entry t-1 is the average Nash regret at time t,
    averaged over num_trials independent runs.
    """
    W = int(np.sqrt(T_max))+1
    mu_star = np.max(means)
    total_rewards = []

    for _ in tqdm(range(num_trials), desc=f"LDP-NCB Trials"):
        arms = LDP_NCB_single(means, c, T_max, W, epsilon, alpha, test_type)
        rewards = np.array(means)[arms]                # length T_max
        rewards.reshape(1, T_max)
        total_rewards.append(rewards)

    total_rewards = np.array(total_rewards)
    expected_means = np.sum(total_rewards, axis = 0)/num_trials

    # delta= 1e-100
    cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-300)))   # shape (T_max,)
    inv_t = 1.0 / np.arange(1, T_max+1)
    geom_mean = np.exp(cumsum_log * inv_t)           # shape (T_max,)
    avg_regret = mu_star - geom_mean                  # shape (T_max,)
    return avg_regret

    # return avg_regret[len(avg_regret)-1]
