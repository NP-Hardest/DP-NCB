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
def GDP_NCB_single(means, c, epsilon, alpha, W, T, test_type):
    """
    Modified NCB with a private, adaptive Phase II.
    
    Inputs:
      - means:    true reward probabilities for each arm (length k)
      - c:        confidence scaling constant
      - epsilon:  privacy budget
      - alpha:    noise‐scale constant (needs alpha > 3)
      - W:        time‐window cap for initial exploration
      - T:        total horizon
    Returns:
      - arms:     array of length T of the pulled arm indices
    """
    k = len(means)
    logT = np.log(T)
    logW = np.log(W)
    
    # Phase I bookkeeping
    N1   = np.zeros(k, dtype=np.int64)       # counts
    mu0  = np.zeros(k, dtype=float)     # phase‐I empirical means

    arms = np.empty(T, dtype=np.int64)
    t = 0

    # --- Phase I: uniform exploration up to window W or until signal is strong ---
    thresh = 1600 * (c**2 * logW + (logW**2) / epsilon)
    while t < T and t < W and np.max(N1 * mu0) <= thresh:
        a = np.random.randint(k)
        # draw reward
        r = sample_reward(means[a], test_type)

        # update Phase I stats
        N1[a] += 1
        # running‐mean update for mu0[a]
        mu0[a] = ((N1[a] - 1) * mu0[a] + r) / N1[a]

        arms[t] = a
        t += 1


    # Phase II bookkeeping
    N2 = np.ones(k, dtype=np.int64)          # start with one pseudo‐pull each
    mu_tilde = np.zeros(k, dtype=float)   # will store the private estimates

    # --- Phase II: private adaptive episodes until horizon T ---
    while t < T:
        # 1) pick arm by NCB on the *current* (noisy‐then‐denoised) estimates
        bonus = np.empty(k, dtype=float)
        
        for i in range(k):
            n_i = N1[i] + N2[i]
            if n_i == 0:
                bonus[i] = np.inf
            else:
                b1 = 2 * c * np.sqrt((2 * mu_tilde[i] * logW) / (n_i)) 
                b2 = (alpha * (logW)**2) / (epsilon * n_i)
                b3 = (4 * np.sqrt(2*alpha) * (logW)**1.5) / (np.sqrt(epsilon) * n_i)
                bonus[i] = b1 + b2 + b3

        scores = mu_tilde + bonus
        A = int(np.argmax(scores))

        # 2) simulate one “prior” draw to re‐initialize mu_hat[A]
        #    (the pseudo‐code sets mu_hat^0 and then does 2*n_s pulls;
        #     here we’ll roll it all together)
        n_s = N2[A]  # backlog count at the start of this episode
        mu_hat_A = mu0[A]
        N2[A] = 0
        # 3) do the 2*n_s real pulls
        for _ in range(2 * n_s):
            if t >= T:
                break
            r = sample_reward(means[A], test_type)

            N2[A] += 1
            # total “effective” count = N1[A] + N2[A] (with N2[A] just incremented)
            tot = N1[A] + N2[A]
            # running‐mean update
            mu_hat_A = ((tot - 1) * mu_hat_A + r) / tot

            arms[t] = A
            t += 1

        # 4) add Laplace noise to make the estimate private
        scale = (logT) / (epsilon * (N1[A] + N2[A]))
        noise = np.random.laplace(loc=0.0, scale=scale)
        mu_tilde[A] = mu_hat_A + noise
        mu_tilde[A] = clip(mu_tilde[A])

    return arms



def simulate_GDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type):
    """
    Returns an array of length T_max where entry t-1 is the average Nash regret at time t,
    averaged over num_trials independent runs.
    """
    W = int(np.sqrt(T_max))+1
    mu_star = np.max(means)
    total_rewards = []

    for _ in tqdm(range(num_trials), desc=f"GDP-NCB Trials"):
        arms = GDP_NCB_single(means, c, epsilon, alpha, W, T_max, test_type)
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
