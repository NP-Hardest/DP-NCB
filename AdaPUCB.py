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
    # 1) pick arm by NCB on the *current* (noisy‐then‐denoised) estimates
        bonus = np.empty(k, dtype=float)
        log_te = np.log(t + 1)
        for i in range(k):
            if n[i] == 0:
                bonus[i] = np.inf  # Prioritize unplayed arms
            else:
                exploration = np.sqrt(alpha * log_te / (n[i]))
                privacy_term = (2 * alpha * log_te) / (n[i])
                bonus[i] = mu_tilde[i] + exploration + privacy_term

        A = np.argmax(bonus)
        

        # 2) simulate one “prior” draw to re‐initialize mu_hat[A]
        #    (the pseudo‐code sets mu_hat^0 and then does 2*n_s pulls;
        #     here we’ll roll it all together)
        n_s = n[A]  # backlog count at the start of this episode
        mu_hat_A = 0
        n[A] = 0
        # 3) do the 2*n_s real pulls
        for _ in range(2 * n_s):
            if t >= T:
                break
            r = sample_reward(means[A], test_type)

            n[A] += 1
            # running‐mean update
            mu_hat_A = ((n[A] - 1) * mu_hat_A + r) / n[A]

            arms[t] = A
            t += 1

        # 4) add Laplace noise to make the estimate private
        scale = 1 / (epsilon * (n[A]))
        noise = np.random.laplace(loc=0.0, scale=scale)
        mu_tilde[A] = mu_hat_A + noise
        mu_tilde[A] = clip(mu_tilde[A])

    return arms

def simulate_adap_ucb(means, epsilon, T_max, num_trials, alpha, test_type):
    """
    Returns an array where entry t-1 is the average Nash regret at time t,
    averaged over num_trials runs.
    """
    # mu_star = np.max(means)

    mu_star = 1

    total_rewards = []

    for _ in tqdm(range(num_trials), desc=f"AdaP-UCB Trials"):
        arms = adap_ucb_single(means, epsilon, T_max, alpha, test_type)
        rewards = np.array(means)[arms]                # length T_max
        rewards.reshape(1, T_max)
        total_rewards.append(rewards)

    total_rewards = np.array(total_rewards)
    expected_means = np.sum(total_rewards, axis = 0)/num_trials
    with np.errstate(divide='ignore'):
        cumsum_log = np.cumsum(np.log(expected_means))
    inv_t = 1.0 / np.arange(1, T_max+1)
    geom_mean = np.exp(cumsum_log * inv_t)           # shape (T_max,)
    avg_regret = mu_star - geom_mean                  # shape (T_max,)

    return avg_regret

    # return avg_regret[len(avg_regret)-1]

