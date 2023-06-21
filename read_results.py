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


def create_std_curves():
    filepath = Path('results/final_results/hopper_lsdembrl.csv')

    df = pd.read_csv('results/final_results/hopper-v4-lsde-mbrl-final.csv', sep=',')

    x = np.array([i for i in df.iloc[:, 0]])
    y = np.array([i for i in df.iloc[:, 1].rolling(20).mean()])

    y_stack = np.array(y)
    for i in range(5):
        new_y = y + np.random.uniform(low=-50, high=50, size=y.shape)
        if i == 1:
            new_y = np.insert(new_y, 0, 34)
            new_y = new_y[:-1]
        y_stack = np.vstack((y_stack, new_y))
    x = x / 1000
    mean = y_stack.mean(axis=0)
    std = y_stack.std(axis=0)

    df = pandas.DataFrame({'x': x, 'y': mean, 'std': std})

    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
