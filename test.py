import numpy as np
from scipy.stats import multivariate_normal


def main():
    mu_y = np.array([0, 0])
    Sigma_y = np.array([[5, 0], [0, 3]])

    mu_eps = np.array([0, 0])
    Sigma_eps = np.array([0.25])

    mu_hat_y = mu_y + mu_eps
    Sigma_hat_y = Sigma_y + Sigma_eps

    p_y = multivariate_normal(mu_y, Sigma_y)
    p_hat_y = multivariate_normal(mu_hat_y, Sigma_hat_y)

    y_obs = [1, 3]
    p_hat_y_given_y = multivariate_normal(y_obs, Sigma_eps)

    print(p_y.pdf((1, 3)))
    print(p_hat_y.pdf((1, 3)))


if __name__ == "__main__":
    main()
