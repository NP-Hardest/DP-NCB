import matplotlib.pyplot as plt
import numpy as np




def plot_expt_a(regrets):
     
    nash_regret_adap_ucb = regrets[0]
    nash_regret_GDP_NCB = regrets[1]


    t_values = np.arange(3, 1003)

    plt.figure(figsize=(8, 5))
    # plt.xscale('log')
    plt.plot(t_values, nash_regret_adap_ucb, label = "AdaP-UCB")
    plt.plot(t_values, nash_regret_GDP_NCB, label = "GDP-NCB")
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel("Nash Regret", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_a.png")
    plt.show()


def plot_expt_b(regrets, T_max, flag):
    
    ts = np.arange(1, T_max+1)

    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots

    nash_regret_ncb = regrets[0][idxs]
    nash_regret_adap_ucb  = regrets[1][idxs]
    nash_regret_ldp_ucb = regrets[2][idxs]
    nash_regret_gdpncb = regrets[3][idxs]
    nash_regret_ldpncb  = regrets[4][idxs]


        
    mask = (ts >= 1e2) & (ts<= T_max)       # to plot between 1e6 and 6e6

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], nash_regret_ncb[mask], label = f"Non-Private NCB")
    plt.plot(ts[mask], nash_regret_adap_ucb[mask], label = f"AdaP-UCB" )
    plt.plot(ts[mask], nash_regret_ldp_ucb[mask], label = r"LDP-UCB")
    plt.plot(ts[mask], nash_regret_gdpncb[mask], label = r"GDP-NCB")
    plt.plot(ts[mask], nash_regret_ldpncb[mask], label=r'LDP-NCB')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel("Nash Regret" if flag else "Average Regret", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_b.png")
    plt.show()



def plot_expt_c_d(regrets, T_max, name):
    ts = np.arange(1, T_max+1)


    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    
    nash_regret_ncb = regrets[0][idxs]
    nash_regret_dp_ncb_02  = regrets[1][idxs]
    nash_regret_dp_ncb_05 = regrets[2][idxs]
    nash_regret_dp_ncb_1 = regrets[3][idxs]
    nash_regret_dp_ncb_2  = regrets[4][idxs]

        
    mask = (ts >= 1e2) & (ts<= T_max)       # to plot between 1e6 and 6e6

    expt = "c" if name == "GDP" else "d"

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], nash_regret_ncb[mask], label=r'Non-Private NCB')
    plt.plot(ts[mask], nash_regret_dp_ncb_02[mask], label=rf"{name}-NCB, $\epsilon=0.2$")
    plt.plot(ts[mask], nash_regret_dp_ncb_05[mask], label=rf"{name}-NCB, $\epsilon=0.5$")
    plt.plot(ts[mask], nash_regret_dp_ncb_1[mask], label=rf"{name}-NCB, $\epsilon=1$" )
    plt.plot(ts[mask], nash_regret_dp_ncb_2[mask], label=rf"{name}-NCB, $\epsilon=2$" )
    # plt.xscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel("Nash Regret", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_{expt}.png")
    plt.show()


def plot_expt_e_f(regrets, T_max, expt):
    ts = np.arange(1, T_max+1)


    num_points = 10000
    idxs = np.linspace(0, len(ts) - 1, num_points, dtype=int)

    ts = ts[idxs]                    ## getting only 100k points for smooth plots
    nash_regret_ncb = regrets[0][idxs]
    nash_regret_gdp_ncb_02  = regrets[1][idxs]
    nash_regret_ldp_ncb_02 = regrets[2][idxs]


        
    mask = (ts >= 1e2) & (ts<= T_max)       # to plot between 1e6 and 6e6

    plt.figure(figsize=(8, 5))
    plt.plot(ts[mask], nash_regret_ncb[mask], label=r'Non-Private NCB')
    plt.plot(ts[mask], nash_regret_gdp_ncb_02[mask], label=r'GDP-NCB')
    plt.plot(ts[mask], nash_regret_ldp_ncb_02[mask], label=r'LDP-NCB')
    # plt.xscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('T (log scale)' if plt.gca().get_xscale() == 'log' else 'T', fontsize=16)
    plt.ylabel("Nash Regret" , fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Results/expt_{expt}.png")
    plt.show()
