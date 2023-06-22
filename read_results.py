import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    df = pd.read_csv('results/final_results/hopper-v4-lsde-mbrl-final.csv', sep=',')

    x = np.array([i for i in df.iloc[:, 0]])
    y = np.array([i for i in df.iloc[:, 1]])
    plt.ylim(0, 3500)
    plt.plot(x, y)
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title('Hopper-v4 LSDE-MBRL')
    plt.savefig('results/final_results/Hopperv4-lsde-mbrl-final.png')

if __name__ == "__main__":
    main()
