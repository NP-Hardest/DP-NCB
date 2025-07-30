import os
import numpy as np
from Expt_1 import run_expt_1
from algorithms.AdaPUCB import simulate_adap_ucb
from algorithms.NCBNonPrivate import simulate_ncb
from algorithms.GDPNCB import simulate_GDP_NCB
from algorithms.LDPNCB import simulate_LDP_NCB
from algorithms.UCB import simulate_ucb
from algorithms.LDPUCB import simulate_LDP_UCB_L
from helper import plot_expt_a, plot_expt_b, plot_expt_c_d, plot_expt_e_f


# CHANGE THESE DIRECTORIES
path_to_cached = "/Users/nishantpandey/Desktop/Stuff With Code/DP-NCB/DP-NCB/Cached"
path_to_results = "/Users/nishantpandey/Desktop/Stuff With Code/DP-NCB/DP-NCB/Results"

os.makedirs(path_to_cached, exist_ok=True)
os.makedirs(path_to_results, exist_ok=True)

for suffix in ['a', 'b', 'c', 'd', 'e', 'f']:
    subdir = os.path.join(path_to_cached, f"expt_{suffix}")
    os.makedirs(subdir, exist_ok=True)



if __name__ == "__main__":

    np.random.seed(42)

    ##### PARAMETERS
    num_trials = 50
    c = 3.0
    alpha = 3.1
    test_type = 0
    epsilon = 2

    ################# Experiment A: Comparing Nash Regret for AdaP-UCB and GDP-NCB for 2 bernoulli arms with means (2e)^-T and 1 #################

    # nash_regret_adap_ucb, nash_regret_gdp_ncb = run_expt_1(num_trials, c, alpha, test_type, epsilon)
    # # np.save("Cached/expt_a/nash_regret_adap_ucb.npy", nash_regret_adap_ucb)
    # np.save("Cached/expt_a/nash_regret_gdp_ncb.npy", nash_regret_gdp_ncb)



    nash_regret_adap_ucb = np.load("Cached/expt_a/nash_regret_adap_ucb.npy")
    nash_regret_gdp_ncb = np.load("Cached/expt_a/nash_regret_gdp_ncb.npy")

    plot_expt_a([nash_regret_adap_ucb, nash_regret_gdp_ncb])



    ################# Experiment B: Comparing UCB, AdaP-UCB and GDP-NCB for Epsilon = 0.2, Bernoulli #################

    # means = [0.25, 0.375, 0.5, 0.625, 0.75]
    means = np.random.uniform(0.005, 1, size=50)
    T_max = 100000000      # choose your total horizon once

    epsilon = 0.2

    # avg_regret_ucb_nash = simulate_ucb(means, T_max, num_trials, test_type, "Nash")
    # np.save("Cached/expt_a/nash_regret_UCB_non_private.npy", avg_regret_ucb_nash)

    # nash_regret_ncb = simulate_ncb(means, T_max, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_b/nash_regret_NCB.npy", nash_regret_ncb)

    # nash_regret_adap_ucb = simulate_adap_ucb(means, epsilon, T_max, num_trials, alpha, test_type, "Nash")
    # np.save("Cached/expt_b/nash_regret_AdaP-UCB_eps_02.npy", nash_regret_adap_ucb)

    # nash_regret_ldp_ucb = simulate_LDP_UCB_L(means, T_max, epsilon, num_trials, test_type)
    # np.save("Cached/expt_b/nash_regret_LDP_UCB_eps_02.npy", nash_regret_ldp_ucb)

    # nash_regret_gdp_ncb = simulate_GDP_NCB(means, T_max, epsilon, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_b/nash_regret_GDP_NCB_eps_02.npy", nash_regret_gdp_ncb)

    # nash_regret_ldp_ncb = simulate_LDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_b/nash_LDP_NCB_eps_02.npy", nash_regret_ldp_ncb)



    nash_regret_ncb = np.load("Cached/expt_b/nash_regret_NCB.npy")                  #load cached regret
    nash_regret_adap_ucb = np.load("Cached/expt_b/nash_regret_AdaP-UCB_eps_02.npy")
    nash_regret_ldp_ucb = np.load("Cached/expt_b/nash_regret_LDP_UCB_eps_02.npy")
    nash_regret_gdp_ncb = np.load("Cached/expt_b/nash_regret_GDP_NCB_eps_02.npy")
    nash_regret_ldp_ncb = np.load("Cached/expt_b/nash_LDP_NCB_eps_02.npy")

    plot_expt_b([nash_regret_ncb, nash_regret_adap_ucb, nash_regret_ldp_ucb, nash_regret_gdp_ncb, nash_regret_ldp_ncb ], T_max, True)



    ################# Experiment C: Comparing GDP-NCB for different epsilon #################

    T_max = 10000000                

    # nash_regret_ncb = simulate_ncb(means, T_max, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_c/nash_regret_NCB.npy", nash_regret_ncb)

    # nash_regret_gdp_ncb_eps_02 = simulate_GDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_c/nash_regret_GDP_NCB_eps_02.npy", nash_regret_gdp_ncb_eps_02)

    # nash_regret_gdp_ncb_eps_05 = simulate_GDP_NCB(means, T_max, 0.5, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_c/nash_regret_GDP_NCB_eps_05.npy", nash_regret_gdp_ncb_eps_05)

    # nash_regret_gdp_ncb_eps_1 = simulate_GDP_NCB(means, T_max, 1, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_c/nash_regret_GDP_NCB_eps_1.npy", nash_regret_gdp_ncb_eps_1)

    # nash_regret_gdp_ncb_eps_2 = simulate_GDP_NCB(means, T_max, 2, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_c/nash_regret_GDP_NCB_eps_2.npy", nash_regret_gdp_ncb_eps_2)


    nash_regret_ncb = np.load("Cached/expt_c/nash_regret_NCB.npy")                  #load cached regret
    nash_regret_gdp_ncb_eps_02 = np.load("Cached/expt_c/nash_regret_GDP_NCB_eps_02.npy")
    nash_regret_gdp_ncb_eps_05 = np.load("Cached/expt_c/nash_regret_GDP_NCB_eps_05.npy")
    nash_regret_gdp_ncb_eps_1 = np.load("Cached/expt_c/nash_regret_GDP_NCB_eps_1.npy")
    nash_regret_gdp_ncb_eps_2 = np.load("Cached/expt_c/nash_regret_GDP_NCB_eps_2.npy")


    plot_expt_c_d([nash_regret_ncb, nash_regret_gdp_ncb_eps_02, nash_regret_gdp_ncb_eps_05, nash_regret_gdp_ncb_eps_1, nash_regret_gdp_ncb_eps_2], T_max, "GDP")



    ################## Experiment D : Comparing LDP-NCB for different epsilon ####################

    # nash_regret_ncb = simulate_ncb(means, T_max, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_d/nash_regret_NCB.npy", nash_regret_ncb)

    # nash_regret_ldp_ncb_eps_02 = simulate_LDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_d/nash_regret_LDP_NCB_eps_02.npy", nash_regret_ldp_ncb_eps_02)

    # nash_regret_ldp_ncb_eps_05 = simulate_LDP_NCB(means, T_max, 0.5, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_d/nash_regret_LDP_NCB_eps_05.npy", nash_regret_ldp_ncb_eps_05)

    # nash_regret_ldp_ncb_eps_1 = simulate_LDP_NCB(means, T_max, 1, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_d/nash_regret_LDP_NCB_eps_1.npy", nash_regret_ldp_ncb_eps_1)

    # nash_regret_ldp_ncb_eps_2 = simulate_LDP_NCB(means, T_max, 2, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_d/nash_regret_LDP_NCB_eps_2.npy", nash_regret_ldp_ncb_eps_2)
    


    nash_regret_ncb = np.load("Cached/expt_d/nash_regret_NCB.npy")                  #load cached regret
    nash_regret_ldp_ncb_eps_02 = np.load("Cached/expt_d/nash_regret_LDP_NCB_eps_02.npy")
    nash_regret_ldp_ncb_eps_05 = np.load("Cached/expt_d/nash_regret_LDP_NCB_eps_05.npy")
    nash_regret_ldp_ncb_eps_1 = np.load("Cached/expt_d/nash_regret_LDP_NCB_eps_1.npy")
    nash_regret_ldp_ncb_eps_2 = np.load("Cached/expt_d/nash_regret_LDP_NCB_eps_2.npy")

    plot_expt_c_d([nash_regret_ncb, nash_regret_ldp_ncb_eps_02, nash_regret_ldp_ncb_eps_05, nash_regret_ldp_ncb_eps_1, nash_regret_ldp_ncb_eps_2], T_max, "LDP")


    ################# Experiment E : Epsilon = 2, Bernoulli Arms #################

    # nash_regret_ncb = simulate_ncb(means, T_max, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_e/nash_regret_NCB.npy", nash_regret_ncb)

    # nash_regret_gdp_ncb_eps_02 = simulate_GDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_e/nash_regret_GDP_NCB_eps_02.npy", nash_regret_gdp_ncb_eps_02)

    # nash_regret_ldp_ncb_eps_02 = simulate_LDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_e/nash_regret_LDP_NCB_eps_02.npy", nash_regret_ldp_ncb_eps_02)

    nash_regret_ncb = np.load("Cached/expt_d/nash_regret_NCB.npy")
    nash_regret_gdp_ncb_eps_02 = np.load("Cached/expt_c/nash_regret_GDP_NCB_eps_02.npy") 
    nash_regret_ldp_ncb_eps_02 = np.load("Cached/expt_d/nash_regret_LDP_NCB_eps_02.npy")


    plot_expt_e_f([nash_regret_ncb, nash_regret_gdp_ncb_eps_02, nash_regret_ldp_ncb_eps_02 ], T_max, 'e')
    

    ################# Experiment F : Epsilon = 2, Multiple Distributions Arms #################
    test_type = 1
    
    # nash_regret_ncb = simulate_ncb(means, T_max, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_f/nash_regret_NCB.npy", nash_regret_ncb)

    # nash_regret_gdp_ncb_eps_02 = simulate_GDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type, "Nash")
    # np.save("Cached/expt_f/nash_regret_GDP_NCB_eps_02.npy", nash_regret_gdp_ncb_eps_02)

    # nash_regret_ldp_ncb_eps_02 = simulate_LDP_NCB(means, T_max, 0.2, alpha, num_trials, c, test_type)
    # np.save("Cached/expt_f/nash_regret_LDP_NCB_eps_02.npy", nash_regret_ldp_ncb_eps_02)

    nash_regret_ncb = np.load("Cached/expt_f/nash_regret_NCB.npy")
    nash_regret_gdp_ncb_eps_02 = np.load("Cached/expt_f/nash_regret_GDP_NCB_eps_02.npy") 
    nash_regret_ldp_ncb_eps_02 = np.load("Cached/expt_f/nash_regret_LDP_NCB_eps_02.npy")

    plot_expt_e_f([nash_regret_ncb, nash_regret_gdp_ncb_eps_02, nash_regret_ldp_ncb_eps_02 ], T_max, 'f')
    