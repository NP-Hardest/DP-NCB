import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from AdaPUCB import simulate_adap_ucb
from NCBNonPrivate import simulate_ncb
from GDPNCB import simulate_GDP_NCB
from Expt_1 import run_expt_1
from LDPNCB import simulate_LDP_NCB

# np.random.seed(42)


path_to_cached = "/Users/nishantpandey/Desktop/DP-NCB/DP-NCB/Cached"
path_to_results = "/Users/nishantpandey/Desktop/DP-NCB/DP-NCB/Results"
# path_to_cached = "/Users/nishant.pandey1/Desktop/scratchy/untitled folder/GDP-NCB/Cached"
# path_to_results = "/Users/nishant.pandey1/Desktop/scratchy/untitled folder/GDP-NCB/Results"

if not os.path.exists(path_to_cached):
    os.makedirs(path_to_cached)

if not os.path.exists(path_to_results):
    os.makedirs(path_to_results)



def plot_type_1(avg_regret, avg_regret_ldpncb, avg_regret_gdpncb, expt):
    ts = np.arange(1, T_max+1)


    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    avg_regret = avg_regret[idxs]
    avg_regret_gdpncb  = avg_regret_gdpncb[idxs]
    avg_regret_ldpncb  = avg_regret_ldpncb[idxs]

        
    mask = (ts >= 1e2) & (ts<= T_max)       # to plot between 1e6 and 6e6

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], avg_regret[mask], label=f"Non-Private NCB")
    plt.plot(ts[mask], avg_regret_gdpncb[mask], label=r'GDP-NCB')
    plt.plot(ts[mask], avg_regret_ldpncb[mask], label=r'LDP-NCB')
    # plt.xscale('log')
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T')
    plt.ylabel("Nash Regret")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_{expt}.png")
    plt.show()

def plot_type_2(regrets):
    ts = np.arange(1, T_max+1)

    num_points = 100000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    avg_regret = regrets[0][idxs]
    avg_regret_02  = regrets[1][idxs]
    avg_regret_05  = regrets[2][idxs]
    avg_regret_1  = regrets[3][idxs]
    avg_regret_2  = regrets[4][idxs]

    mask = (ts >= 1e2) & (ts<= T_max)       # to plot between 1e6 and 6e6

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], avg_regret[mask], label=f"Non-Private NCB")
    plt.plot(ts[mask], avg_regret_02[mask], label=r"GDP-NCB, $\epsilon=0.2$")
    plt.plot(ts[mask], avg_regret_05[mask], label=r'GDP-NCB, $\epsilon=0.5$')
    plt.plot(ts[mask], avg_regret_1[mask], label=r'GDP-NCB, $\epsilon=1$')
    plt.plot(ts[mask], avg_regret_2[mask], label=r'GDP-NCB, $\epsilon=2$')
    # plt.xscale('log')
    plt.xticks([2*10**6, 4*10**6, 6*10**6, 8*10**6, 10**7], [ f"2e6", f"4e6", f"6e6", f"8e6", f"1e7"])
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T')
    plt.ylabel("Nash Regret")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_3.png")
    plt.show()


if __name__ == "__main__":

    np.random.seed(42)

    ##### PARAMETERS
    num_trials = 50
    c = 3.0
    alpha = 3.1
    test_type = 0
    epsilon = 2

    ################# Experiment 1: Comparing Nash Regret for AdaP-UCB and GDP-NCB for 2 bernoulli arms with means (2e)^-T and 1 #################

    # run_expt_1(num_trials, c, alpha, test_type, epsilon)

    ################# Experiment 2: Comparing GDP-NCB and LDP-NCB for Epsilon = 0.2, Bernoulli #################

    means = np.random.uniform(0.005, 1, size=50)
    # means = [0.25, 0.375, 0.5, 0.625, 0.75]
    T_max = 10000000      # choose your total horizon once

    epsilon = 0.2

    # avg_regret = np.load("Cached/regret_NCB_non_private.npy")     #preload
    # avg_regret = simulate_ncb(means, T_max, num_trials, c, test_type)
    # np.save("Cached/regret_NCB_non_private.npy", avg_regret)
    # avg_regret_ldp_ncb = simulate_LDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type)
    # np.save("Cached/regret_LDP_NCB_eps_02.npy", avg_regret_ldp_ncb)
    # avg_regret_dpncb = simulate_GDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type)
    # np.save("Cached/regret_GDP_NCB_eps_02.npy", avg_regret_dpncb)

    # # avg_regret = np.load("Cached/regret_NCB_non_private.npy")
    # # avg_regret_ldp_ncb = np.load("Cached/regret_LDP_NCB_eps_02.npy")
    # # avg_regret_dpncb = np.load("Cached/regret_GDP_NCB_eps_02.npy")


    # plot_type_1(avg_regret, avg_regret_ldp_ncb, avg_regret_dpncb, 2)


    # ################# Experiment 3 : GDP-NCB and LDP-NCB for different epsilon ####################

    # regret_non_private = simulate_ncb(means, T_max, num_trials, c, test_type)

    regret_non_private = np.load("Cached/regret_NCB_non_private.npy")
    regret_gdpncb_02 = np.load("Cached/regret_GDP_NCB_eps_02.npy")
    # regret_gdpncb_02 = simulate_GDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type)
    regret_gdpncb_05 = simulate_GDP_NCB(means, T_max, 0.5, alpha, num_trials, c, test_type)
    regret_gdpncb_1 = simulate_GDP_NCB(means, T_max, 1, alpha, num_trials, c, test_type)
    regret_gdpncb_2 = simulate_GDP_NCB(means, T_max, 2, alpha, num_trials, c, test_type)

    plot_type_2([regret_non_private, regret_gdpncb_02, regret_gdpncb_05, regret_gdpncb_1, regret_gdpncb_2])



    test_type = 1




    ################# Experiment 4 : Epsilon = 2, Multiple Distributions #################

    
    avg_regret = simulate_ncb(means, T_max, num_trials, c, test_type)
    np.save("Cached/regret_NCB_non_private_multi_mixed.npy", avg_regret)
    # avg_regret = np.load("Cached/regret_NCB_non_private.npy")

    avg_regret_ldp_ncb = simulate_LDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type)
    np.save("Cached/regret_LDP_NCB_eps_02_mixed.npy", avg_regret_ldp_ncb)
    avg_regret_gdpncb = simulate_GDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type)
    np.save("Cached/regret_GDP_NCB_eps_02_mixed.npy", avg_regret_gdpncb)


    plot_type_1(avg_regret, avg_regret_ldp_ncb, avg_regret_gdpncb, 4)