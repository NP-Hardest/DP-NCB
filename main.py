import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from AdaPUCB import simulate_adap_ucb
from NCBNonPrivate import simulate_ncb
from DPNCB import simulate_DP_NCB
from Expt_1 import run_expt_1

np.random.seed(42)


path_to_cached = "/Users/nishantpandey/Desktop/Stuff With Code/dpncb/Cached"
path_to_results = "/Users/nishantpandey/Desktop/Stuff With Code/dpncb/Results"
# path_to_cached = "/Users/nishant.pandey1/Desktop/scratchy/untitled folder/DP-NCB/Cached"
# path_to_results = "/Users/nishant.pandey1/Desktop/scratchy/untitled folder/DP-NCB/Results"

if not os.path.exists(path_to_cached):
    os.makedirs(path_to_cached)

if not os.path.exists(path_to_results):
    os.makedirs(path_to_results)



def plot_type_1(avg_regret, avg_regret_adap, avg_regret_dpncb, epsilon, expt):
    ts = np.arange(1, T_max+1)


    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    avg_regret = avg_regret[idxs]
    avg_regret_adap  = avg_regret_adap[idxs]
    avg_regret_dpncb  = avg_regret_dpncb[idxs]

    if(expt == 2):
        bound_plot =    4*(np.sqrt(len(means) * np.log(ts) / ts) + (np.log(ts)**2) * (np.log(len(means))) / (epsilon * ts))
    else:
        bound_plot =    7*(np.sqrt(len(means) * np.log(ts) / ts) + (np.log(ts)**2) * (np.log(len(means))) / (epsilon * ts))
        
    mask = (ts >= 1e6) & (ts<= 1e7)       # to plot between 1e6 and 6e6

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], avg_regret[mask], label=f"Non-Private NCB")
    plt.plot(ts[mask], avg_regret_adap[mask], label=f"AdaP-UCB")
    plt.plot(ts[mask], avg_regret_dpncb[mask], label=r'DP-NCB')
    plt.plot(ts[mask], bound_plot[mask], '--', label=r'Theoretical Bound $O\left(\sqrt{\frac{k\log T}{T}} + \frac{(\log T)^2}{\epsilon T}\right)$')
    # plt.xscale('log')
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T')
    # plt.xticks([2*10**6, 4*10**6, 6*10**6, 8*10**6, 10**7], [ f"2000000", f"4000000", f"6000000", f"8000000", f"10000000"])
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

    mask = (ts >= 1e6) & (ts<= 1e7)       # to plot between 1e6 and 6e6

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], avg_regret[mask], label=f"Non-Private NCB")
    plt.plot(ts[mask], avg_regret_02[mask], label=r"DP-NCB, $\epsilon=0.2$")
    plt.plot(ts[mask], avg_regret_05[mask], label=r'DP-NCB, $\epsilon=0.5$')
    plt.plot(ts[mask], avg_regret_1[mask], label=r'DP-NCB, $\epsilon=1$')
    plt.plot(ts[mask], avg_regret_2[mask], label=r'DP-NCB, $\epsilon=2$')
    # plt.xscale('log')
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T')
    plt.xticks([2*10**6, 4*10**6, 6*10**6, 8*10**6, 10**7], [ f"2000000", f"4000000", f"6000000", f"8000000", f"10000000"])
    plt.ylabel("Nash Regret")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_3.png")
    plt.show()


if __name__ == "__main__":



    ##### PARAMETERS
    num_trials = 50
    c = 3.0
    alpha = 3.1
    test_type = 0
    epsilon = 2
    ################# Experiment 1: Comparing Nash Regret for AdaP-UCB and DP-NCB for 2 bernoulli arms with means (2e)^-T and 1 #################

    run_expt_1(num_trials, c, alpha, test_type, epsilon)



    means = np.random.uniform(0.005, 1, size=50)
    T_max = 10000000      # choose your total horizon once


    avg_regret = simulate_ncb(means, T_max, num_trials, c, test_type)
    np.save("Cached/regret_NCB_non_private.npy", avg_regret)
    # avg_regret = np.load("Cached/regret_NCB_non_private.npy")

    ################# Experiment 2: Comparing various algorithms for Epsilon = 2, Bernoulli #################
    avg_regret_adap = simulate_adap_ucb(means, epsilon, T_max, num_trials, alpha, test_type)
    np.save("Cached/regret_AdaP_UCB_eps_2.npy", avg_regret_adap)
    avg_regret_dpncb = simulate_DP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type)
    np.save("Cached/regret_DP_NCB_eps_2.npy", avg_regret_dpncb)


    plot_type_1(avg_regret, avg_regret_adap, avg_regret_dpncb, epsilon, 2)


    ################# Experiment 3 : DP-NCB for different epsilon ####################

    regret_non_private = np.load("Cached/regret_NCB_non_private.npy")
    regret_dpncb_02 = np.load("Cached/regret_DP_NCB_eps_2.npy")
    regret_dpncb_05 = simulate_DP_NCB(means, T_max, 0.5, alpha, num_trials, c, test_type)
    regret_dpncb_1 = simulate_DP_NCB(means, T_max, 1, alpha, num_trials, c, test_type)
    regret_dpncb_2 = simulate_DP_NCB(means, T_max, 2, alpha, num_trials, c, test_type)

    plot_type_2([regret_non_private, regret_dpncb_02, regret_dpncb_05, regret_dpncb_1, regret_dpncb_2])



    test_type = 1


    # avg_regret = simulate_ncb(means, T_max, num_trials, c, test_type)
    # np.save("Cached/regret_NCB_non_private_multi.npy", avg_regret)

    avg_regret = np.load("Cached/regret_NCB_non_private.npy")


    ################# Experiment 4 : Epsilon = 2, Multiple Distributions #################

    avg_regret_adap = simulate_adap_ucb(means, epsilon, T_max, num_trials, alpha, test_type)
    np.save("Cached/regret_AdaP_UCB_eps_2.npy", avg_regret_adap)
    avg_regret_dpncb = simulate_DP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type)
    np.save("Cached/regret_DP_NCB_eps_2.npy", avg_regret_dpncb)


    plot_type_1(avg_regret, avg_regret_adap, avg_regret_dpncb, epsilon, 4)
