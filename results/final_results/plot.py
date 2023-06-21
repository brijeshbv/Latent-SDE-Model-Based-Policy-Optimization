import pickle
import matplotlib
import pandas

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
	colors = {
		'mbpo': '#1f77b4',
		'lsdembrl': '#42f59e'
	}

	tasks = ['hopper']
	algorithms = ['mbpo', 'lsdembrl']

	for task in tasks:
		plt.clf()

		for alg in algorithms:
			print(task, alg)
			fname = '{}_{}.csv'.format(task, alg)

			## load results
			# data = pickle.load(open(fname, 'rb'))
			data = pd.read_csv(fname, sep=',')
			# df = pd.DataFrame(data)
			# df.to_csv('{}_{}.csv'.format(task, alg), index=False)
			## plot trial mean
			plt.plot(data['x'], data['y'], linewidth=1.5, label=alg, c=colors[alg])
			## plot error bars
			plt.fill_between(data['x'], data['y']-data['std'], data['y']+data['std'], color=colors[alg], alpha=0.25)

		plt.legend()

		savepath = '{}_comparison.png'.format(task)
		plt.savefig(savepath)


if __name__ == "__main__":
    main()