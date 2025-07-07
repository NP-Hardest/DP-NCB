from mpmath import mp
import numpy as np
# from AdaPUCB import simulate_adap_ucb
# from DPNCB import simulate_DP_NCB
from tqdm import tqdm
from matplotlib import pyplot as plt


def clip(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x


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

def simulate_adap_ucb_expt1(means, epsilon, T_max, num_trials, alpha, test_type):
    """
    Returns an array where entry t-1 is the average Nash regret at time t,
    averaged over num_trials runs.
    """
    mu_star = np.max(means)


    total_rewards = []

    for _ in (range(num_trials)):
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

    # print(avg_regret[len(avg_regret)-1])
    return avg_regret[len(avg_regret)-1]

    
def DP_NCB_single(means, c, epsilon, alpha, W, T, test_type):
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
    thresh = 1 * (c**2 * logW + (logT**2) / epsilon)
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



def simulate_DP_NCB_expt1(means, T_max, epsilon, alpha, num_trials, c, test_type):
    """
    Returns an array of length T_max where entry t-1 is the average Nash regret at time t,
    averaged over num_trials independent runs.
    """
    W = int(np.sqrt(T_max))+1
    mu_star = np.max(means)
    total_rewards = []

    for _ in range(num_trials):
        arms = DP_NCB_single(means, c, epsilon, alpha, W, T_max, test_type)
        rewards = np.array(means)[arms]                # length T_max
        rewards.reshape(1, T_max)
        total_rewards.append(rewards)

    total_rewards = np.array(total_rewards)
    expected_means = np.sum(total_rewards, axis = 0)/num_trials

    # delta= 1e-100
    cumsum_log = np.cumsum(np.log(expected_means))   # shape (T_max,)
    inv_t = 1.0 / np.arange(1, T_max+1)
    geom_mean = np.exp(cumsum_log * inv_t)           # shape (T_max,)
    avg_regret = mu_star - geom_mean                  # shape (T_max,)

    return avg_regret[len(avg_regret)-1]




t_values = np.arange(3, 1003)

num_trials = 50
c = 3.0
alpha = 3.1
test_type = 0
epsilon = 2



def run_expt_1(num_trials, c, alpha, test_type, epsilon):
    reg1 = np.zeros(len(t_values))
    reg2 = np.zeros(len(t_values))
    for _ in range(10):
        print(_)
        regrets = []
        regrets2 = []
        for T in tqdm(t_values, desc=f"Simulating Experiment 1 {_}th time"):
            exponent = -T
            base = 2 * mp.e
            result = mp.power(base, exponent)
            # print(result)
            means = [float(result), 1]
            # print(means)
            avg_regret_adap = simulate_adap_ucb_expt1(means, epsilon, T, num_trials, alpha, test_type)
            avg_regret_dpncb = simulate_DP_NCB_expt1(means, (T), epsilon, alpha, num_trials, c, test_type)

            regrets.append(avg_regret_adap)
            regrets2.append(avg_regret_dpncb)
        reg1 += np.array(regrets)
        reg2 += np.array(regrets2)
    
    reg1 /= 10
    reg2 /= 10


    plt.figure(figsize=(8, 5))
    # plt.xscale('log')
    plt.plot(t_values, reg1, label = "AdaP-UCB")
    plt.plot(t_values, reg2, label = "GDP-NCB")
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T')
    plt.ylabel("Nash Regret")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_1.png")
    plt.show()
