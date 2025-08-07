
# **DP-NCB: Privacy Preserving Fair Bandits**

### Overview

Official Python implementation for the work by Sarkar et al. on Differentially Private Fair Bandits. [[arXiv]](https://arxiv.org/abs/2508.03836)
    

### Repository Structure

```
DP-NCB/
├── algorithms/       # Contains code for all bandit algorithms used
├── Cached/           # Stores regret data from first-time runs
├── Results/          # Final figures
├── Expt-1.py         # Separate script for the first experiment
├── main.py           # Master script: runs Experiments A–F sequentially
├── helper.py         # Common functions for plotting and data handling
└── requirements.txt  # Python dependencies

```


    

#### `Cached/`

Stores serialized regret data generated during the first run of each experiment. On subsequent runs, `main.py` will load these cached files to avoid redundant computation.

#### `Results/`

Holds the final plot images (e.g., `.png` files) for each experiment. These plots compare algorithmic performance under different privacy budgets and non-stationary settings.

### Installation

1.  **Clone the repository**:
    
    ```bash
    git clone https://github.com/NP-Hardest/DP-NCB.git
    cd DP-NCB
    
    ```
    
2.  **Create a conda environment** (optional but recommended):
    
    ```bash
    conda create -n DP-NCB 
    conda activate DP-NCB
    
    ```
    One can also use venv instead of conda.
3.  **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    

### Usage

#### Run All Experiments

Execute the master script to run all experiments (`A` through `F`) in sequence. Use `np.random.seed(42)` for reproducing results from the paper:

```bash
python main.py

```

#### Tweak Parameters

-   Adjust experiment parameters in  `main.py` such as:
    
    -   Privacy budget (ε)
        
    -   Horizon length (T)
        
    -   Number of trials
        
    -   Algorithm-specific settings
        

While running the experiments for the first time, the regret data is stored in `Cached/` directory. For subsequent runs, this data can be directly loaded using `np.load()`:


#### Results & Visualization

-   After running the experiments, plots will be saved in the `Results/` folder.
    
-   Use the functions in `helper.py` to customize plot styles or export formats.
    



#### Contributing

Feel free to fork the repository to reproduce our results and other suggestions for modifications.
    
