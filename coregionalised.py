import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import gpflow
from scipy.cluster.vq import kmeans2
from sklearn.model_selection import train_test_split
import string
import random
tf.logging.set_verbosity(0)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

np.random.seed(123)
tf.set_random_seed(123)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if __name__ == "__main__":
    # Load data
    data_name = "1week"
    aurn = pd.read_csv('demos/coregional_data/aurn_{}.csv'.format(data_name))
    cams = pd.read_csv('demos/coregional_data/cams_{}.csv'.format(data_name)) # Get full CAMS data.
    cams = cams[['date', 'lat', 'lon', 'val']]
    print(cams.date.drop_duplicates())
    mind = aurn.Date.drop_duplicates().tolist()[0]

    aurn = aurn[['Date', 'Latitude', 'Longitude', 'pm25_value']]
    aurn.columns = ['date', 'lat', 'lon', 'val']

    n_sparse = 4000
    if n_sparse:
        zpoints = kmeans2(cams[['date', 'lat', 'lon']].values, n_sparse, minit='points')[0]
        zpoints = np.vstack((zpoints, aurn[['date', 'lat', 'lon']].values))

    aurn['indicator'] = 0
    cams['indicator'] = 1

    # Proportion of CAMS data to be subsetted
    subset_denom = None
    if subset_denom:
        cams = cams.sample(n = int(cams.shape[0]/subset_denom))
    
    all_data = pd.concat([aurn, cams])

    # Check data dimensions
    # assert all_data.shape[0] == aurn.shape[0] + cams.shape[0], "Rows lost in concatenation"
    # assert all_data.shape[1] == aurn.shape[1] == cams.shape[1], "Column count mismatch in data"
    all_data = all_data.head(n=int(all_data.shape[0]/2))

    print('{} observations loaded.'.format(all_data.shape[0]))
    print(all_data.head())

    # Transform Data
    all_data.val = np.log(all_data.val)

    # Split Data""
    X_aug = all_data[['date', 'lat', 'lon', 'indicator']].values
    y_aug = all_data[['val', 'indicator']].values

    X_train, X_test, y_train, y_test = train_test_split(X_aug,
                                                        y_aug,
                                                        test_size=0.4,
                                                        random_state=123)


           # Fit GP
    output_dim = 2

    # Dimension of X, excluding the indicator column
    base_dims = X_train.shape[1] - 1

    # Reference point of the index column
    coreg_dim = X_train.shape[1] - 1

    # Rank of w
    rank = 1

    # Base Kernel
    k1 = gpflow.kernels.RBF(input_dim=2, active_dims=[0, 1], ARD=True)
    k3 = gpflow.kernels.RBF(input_dim = 1, active_dims =[2])
    # Coregeionalised kernel
    k2 = gpflow.kernels.Coregion(1, output_dim=output_dim, rank=rank, active_dims=[int(coreg_dim)])

    # Initialise W
    k2.W = np.random.randn(output_dim, rank)
    # Combine
    kern = k1 * k3 * k2

    # Define Likelihoods
    liks = gpflow.likelihoods.SwitchedLikelihood(
        [gpflow.likelihoods.Gaussian(),
        gpflow.likelihoods.Gaussian()])

    # Variational GP
    if n_sparse:
        m = gpflow.models.SVGP(X_train, y_train, kern = kern, likelihood = liks, Z = zpoints.copy(), num_latent = 1)
    else:
        m = gpflow.models.VGP(X_train,
                            y_train,
                            kern=kern,
                            likelihood=liks,
                            num_latent=1)

    gpflow.train.ScipyOptimizer().minimize(m, maxiter=1000)

    """# Visualise the B Matrix"""
    B = k2.W.value @ k2.W.value.T + np.diag(k2.kappa.value)
    print('-'*80)
    print('B =', B)
    print('-'*80)
    # plt.imshow(B)

    """## Predictions"""

    mu, var = m.predict_f(X_test)
    print('mu shape: {}'.format(mu.shape))

    results = pd.DataFrame(X_test)
    results.columns = ['date', 'lat', 'lon', 'indicator']
    results['mu'] = np.exp(mu)
    results['var'] = var
    results['truth'] = np.exp(y_test[:, 0])
    results['sq_error'] = np.square(results['mu'] - results['truth'])
    print(results.head())

    print("RMSE on {} held out data points: {}".format(
        X_test.shape[0], np.sqrt(np.mean(results.sq_error))))

    fname = 'demos/corregionalised_gp_results_{}_sparse{}.csv'.format(data_name, n_sparse)
    results.to_csv(fname, index=False)

    saver= gpflow.saver.Saver()
    try:
        saver.save('coreg_model_{}_sparse{}.gpflow'.format(data_name, n_sparse), m)
    except ValueError:
        tempname = id_generator()
        print("Filename coreg_model.gpflow already exists. \nSaving model as {}.gpflow".format(tempname))
        saver.save('{}_{}.gpflow'.format(tempname, data_name), m)
    
  # results['sq_error'].groupby(results.indicator).describe()
