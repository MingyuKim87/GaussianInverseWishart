import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy

# data
from data_loader import *

# plot
import seaborn as sns
from plot import *


def empirical_variance(data, x_bar):
    '''
        Args:
            data : [n, dim]
            x_bar : [dim]
        Returns:

    '''
    # data count
    n = data.shape[0]
    
    # diff [n,dim]
    diff = data - x_bar 

    # copy tensor and transpose [dim, n]
    diff_T = (data - x_bar).detach().clone().T

    # matmul [dim, dim]
    S = (1 / n) * torch.matmul(diff_T, diff)

    # cov
    cov = np.cov(data.T)

    return S


def posterior_gaussian_inverse_wishart(data, mu_0, lambda_0, Phi_0, nu_0):
    '''
        Args:
            data : data points [n, dim]
            mu_0 : prior mu_n [dim]
            lambda_0 : prior lambda (scalar)
            W_0 : prior W [dim x dim]
            nu_0 : prior nu_n [scalar]

        Returns:
            mu_n : prior mu_n [dim]
            lambda_n : prior lambda (scalar)
            W_n : prior W [dim x dim]
            nu_n : prior nu_n [scalar]
    '''
    # dimension
    dim = data.shape[-1]

    # Exceptional Treatment
    assert dim == mu_0.shape[-1], "Check out dimensionality"

    # information from data
    n_count = data.shape[0]
    x_bar = data.mean(dim=0)
    x_var = empirical_variance(data, x_bar)

    # posterior
    mu_n = ((lambda_0 * mu_0) + (n_count * x_bar)) / (lambda_0 + n_count)
    lambda_n = lambda_0 + n_count
    nu_n = nu_0 + n_count

    # posterior covariance
    x_var_0 = (lambda_0 * n_count / (lambda_0 + n_count)) * empirical_variance(data, mu_0)
    Phi_n = Phi_0 + x_var + x_var_0 

    return mu_n, lambda_n, Phi_n, nu_n

def posterior_sampling(mu_n_np, lambda_n_np, Phi_n_np, nu_n_np):

    # # type casting
    # mu_n_np = mu_n.numpy()
    # lambda_n_np = lambda_n.numpy()
    # Phi_n_np = Phi_n.numpy()
    # nu_n = nu_n.numpy()

    # sampling wishart dist.
    sigma = scipy.stats.invwishart.rvs(df=nu_n_np, scale=Phi_n_np)

    # sampling gaussian dist.
    mu = scipy.stats.multivariate_normal.rvs(mean=mu_n_np, cov=(1/lambda_n_np) * sigma)

    return mu, sigma

if __name__ == "__main__":
    # file path
    global_file_path = "./Data/GlobalInformation.csv"
    outlier_file_path = "./Data/Outliers.csv"

    # import data
    x,y = load_global_stat(global_file_path, outlier_file_path)

    # standardize
    x,y, x_params, y_params = standardize(x,y)

    # parameters
    n_count = y.shape[0]
    dim = y.shape[-1]

    # hyper parameters
    mu_0 = torch.zeros(dim)
    lambda_0 = 1
    nu_0 = 1
    Phi_0 = torch.eye(dim)

    # data preprocessing
    mu_n, lambda_n, Phi_n, nu_n = posterior_gaussian_inverse_wishart(y, mu_0, lambda_0, Phi_0, nu_0)

    # type casting
    mu_n_np, Phi_n_np = mu_n.numpy(), Phi_n.numpy()

    # mu
    mus = []
    sigmas = []
    sampled_data = []

    for i in range(1000):
        # parameter sampling
        mu, sigma = posterior_sampling(mu_n_np, lambda_n, Phi_n_np, nu_n)

        # data sampling
        data_point = scipy.stats.multivariate_normal.rvs(mean=mu, cov=sigma)

        # type cast
        data_point = data_point.astype(np.float32)

        # append
        mus.append(mu)
        sigmas.append(sigmas)
        sampled_data.append(data_point)

    # type casting
    mus = np.stack(mus, axis=0), 
    #sigmas = np.stack(sigmas, axis=0)
    sampled_data = np.stack(sampled_data, axis=0)

    # file save
    np.save("sampled_data", sampled_data)

    # data frame
    y_df = make_df(y)
    sample_df = make_df(sampled_data)
    
    # Make a plot
        # original data
    plot_1 = sns.pairplot(y_df, plot_kws={'alpha':0.1})
    plot_1.savefig("./scatter_plot_1.png")

        # sample data
    plot_2 = sns.pairplot(sample_df, plot_kws={'alpha':0.1})
    plot_2.savefig("./scatter_plot_2.png")
    










    
    



