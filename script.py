import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class DP_NCB_Bandit:
    def __init__(self, k, T, epsilon, alpha, W):
        self.k, self.T, self.eps, self.alpha, self.W = k, T, epsilon, alpha, W
        self.c = 3
        self.N1 = np.zeros(k, dtype=int)
        self.mu_hat0 = np.zeros(k)
        self.N2 = np.ones(k, dtype=int)
        self.mu_hat = None
        self.mu_tilde = np.zeros(k)
        self.t = 1

        self.phase1_pulls = np.zeros(k, dtype=int)
        self.phase2_pulls = np.zeros(k, dtype=int)

    def _nc_bound(self, i):
        ni = self.N1[i] + self.N2[i]
        if ni == 0:
            return float('inf')
        mu = max(self.mu_tilde[i], 0)  # clamp to avoid negative sqrt
        bonus1 = 2*self.c*np.sqrt(2 * mu * np.log(self.W) / ni)
        bonus2 = (self.alpha * (np.log(self.W)**2)) / (self.eps * ni)
        bonus3 = 4 * np.sqrt((2*self.alpha/self.eps) * (np.log(self.W)**1.5) / ni)
        return mu + bonus1 + bonus2 + bonus3

    def run(self, env):
        history = []
        threshold = 1600 * (self.c**2 * np.log(self.W) + (np.log(self.T)**2) / self.eps)
        while self.t <= self.W and np.max(self.N1 * self.mu_hat0) <= threshold:
            a = np.random.randint(self.k)
            r = env.pull(a)
            history.append((a, r))
            self.N1[a] += 1
            self.phase1_pulls[a] += 1
            self.mu_hat0[a] += (r - self.mu_hat0[a]) / self.N1[a]
            self.t += 1

        self.mu_hat = self.mu_hat0.copy()
        self.mu_tilde = np.zeros(self.k)

        while self.t <= self.T:
            A = max(range(self.k), key=self._nc_bound)
            ns = self.N2[A]
            self.mu_hat[A] = self.mu_hat0[A]
            self.N2[A] = 0
            for _ in range(2*ns):
                if self.t > self.T:
                    break
                r = env.pull(A)
                history.append((A, r))
                self.N2[A] += 1
                self.phase2_pulls[A] += 1
                total = self.N1[A] + self.N2[A]
                self.mu_hat[A] += (r - self.mu_hat[A]) / total
                self.t += 1
            scale = np.log(self.T) / (self.eps * (self.N1[A] + self.N2[A]))
            noise = np.random.laplace(scale=scale)
            self.mu_tilde[A] = self.mu_hat[A] + noise

        return history

class BernoulliEnv:
    def __init__(self, ps):
        self.ps = ps
    def pull(self, arm):
        return 1.0 if np.random.rand() < self.ps[arm] else 0.0


Horizons = []
regrets = []

t = 100
cnt = 0
diff = 9900

while t <= 1000000:
    print(cnt + 1)
    cnt += 1
    results = []
    print(f"T = {int(t)}")

    for i in tqdm(range(0, 50)):
        k = 5
        epsilon = 0.2
        alpha = 3.1
        T = int(t)   
        W = int(np.sqrt(t))
        env = BernoulliEnv(ps=[0.1, 0.2, 0.3, 0.4, 0.5])
        agent = DP_NCB_Bandit(k, T, epsilon, alpha, W)
        history = agent.run(env)
        rewards = np.array([r for (_, r) in history])
        results.append(rewards)


    results = np.array(results)

    final_rewards = np.sum(results, axis = 0)/50

    eps = 1e-12
    geo_mean = np.exp(np.mean(np.log(final_rewards + eps)))

    mu_star = max(env.ps)
    nash_regret = (mu_star - geo_mean)
    regrets.append(float(nash_regret))
    Horizons.append(t)
    t += diff


plt.plot(Horizons, regrets, marker='o')  # 'o' to show data points
plt.xlabel('T')
plt.ylabel('Nash Regret')
plt.title('Nash Regret vs T')
plt.grid(True)
plt.xlim(100, 1000000)
plt.show()

    # after `history = agent.run(env)`:

    # 1. Count pulls and accumulate rewards
    # pulls = np.zeros(k, dtype=int)
    # rewards = np.zeros(k)
    # for arm, r in history:
    #     pulls[arm] += 1
    #     rewards[arm] += r

    # total_reward = rewards.sum()
    # print("=== Bandit Run Summary ===")
    # print(f"Total rounds: {len(history)}")
    # print(f"Cumulative reward: {total_reward:.2f}")

    # print("\nArm | Pulls | Empirical mean (phase I+II) | Private mean (μ̃)")
    # print("----|-------|----------------------------|-----------------")
    # for i in range(k):
    #     emp_mean = rewards[i] / pulls[i] if pulls[i] > 0 else float('nan')
    #     priv_mean = agent.mu_tilde[i]
    #     print(f"{i:3d} | {pulls[i]:5d} | {emp_mean:24.4f} | {priv_mean:13.4f}")

    # if hasattr(env, 'ps'):
    #     mu_star = max(env.ps)
    #     expected_regret = (mu_star * len(history) - total_reward)/T
    #     print(f"\nEstimated regret: {expected_regret:.2f}")

    # print("Phase I pulls per arm:  ", agent.phase1_pulls)
    # print("Phase II pulls per arm: ", agent.phase2_pulls)
    # print("Total pulls per arm:     ", agent.phase1_pulls + agent.phase2_pulls)

    # # extract rewards from history



