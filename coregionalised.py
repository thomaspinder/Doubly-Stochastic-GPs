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
from itertools import product
tf.logging.set_verbosity(0)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

np.random.seed(123)
tf.set_random_seed(123)

def tester(X, gp, y=None):
    mu, var = gp.predict_f(X)
    results = pd.DataFrame(X_test)
    results.columns = ['date', 'lat', 'lon', 'indicator']
    results['mu'] = np.exp(mu)
    results['var'] = var
    if y:
        results['truth'] = np.exp(y_test[:, 0])
        results['sq_error'] = np.square(results['mu'] - results['truth'])
    return results

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if __name__ == "__main__":
    # Load data
    data_name = "1week"
    aurn = pd.read_csv('demos/coregional_data/aurn_{}.csv'.format(data_name))
    cams = pd.read_csv('demos/coregional_data/cams_{}.csv'.format(data_name)) # Get full CAMS data.
    cams = cams[['date', 'lat', 'lon', 'val']]
    mind = aurn.Date.drop_duplicates().tolist()[0]

    aurn = aurn[['Date', 'Latitude', 'Longitude', 'pm25_value']]
    aurn.columns = ['date', 'lat', 'lon', 'val']

    n_sparse = 2000
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
                                                        random_state=123,
                                                        shuffle=True)


    # Fit GP
    output_dim = 2

    # Dimension of X, excluding the indicator column
    base_dims = X_train.shape[1] - 1

    # Reference point of the index column
    coreg_dim = X_train.shape[1] - 1

    # Rank of w
    rank = 1

    # Base Kernel
    k1 = gpflow.kernels.RBF(input_dim=3, active_dims=[0, 1, 2], ARD=True)
    # k3 = gpflow.kernels.RBF(input_dim = 1, active_dims =[2])
    # Coregeionalised kernel
    k2 = gpflow.kernels.Coregion(1, output_dim=output_dim, rank=rank, active_dims=[int(coreg_dim)])

    # Initialise W
    k2.W = np.random.randn(output_dim, rank)
    # Combine
    kern = k1 * k2 # k3

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

    gpflow.train.ScipyOptimizer().minimize(m, maxiter=100)
    
    gp_params = m.as_pandas_table()
    gp_params.to_csv('demos/coreg_{}_{}_gp_params.csv'.format(n_sparse, data_name))

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

    fname = 'demos/corregionalised_nonsep_gp_results_{}_sparse{}.csv'.format(data_name, n_sparse)
    results.to_csv(fname, index=False)

    saver = gpflow.saver.Saver()
    try:
        saver.save('models/coreg_model_{}_sparse{}.gpflow'.format(data_name, n_sparse), m)
    except ValueError:
        tempname = id_generator()
        print("Filename coreg_model.gpflow already exists. \nSaving model as {}.gpflow".format(tempname))
        saver.save('models/{}_{}.gpflow'.format(tempname, data_name), m)

    ##################################
    # Make tests on a linear grid
    ##################################
    # Generate test data
    date_lims = np.arange(cams.date.min(), cams.date.max())
    lats = np.round(np.linspace(cams.lat.min(), cams.lat.max(), num = 50)[:, None], 1)
    lons = np.round(np.linspace(cams.lon.min(), cams.lon.max(), num = 50)[:, None], 1) # To make out of prediction samples: np.arange(cams.date.max() + 1, cams.date.max() + 7)

    # Get all combinations of lat/lon
    coord_set = list(product(lats, lons))
    coords = np.vstack([np.hstack((coord_set[i][0], coord_set[i][1])) for i in range(len(coord_set))])
    
    # Build a dates column
    dates = np.repeat(date_lims, repeats=coords.shape[0])[:, None]
    indicator = np.vstack((np.zeros_like(dates), np.ones_like(dates)))
    coords_full = np.tile(coords, (date_lims.shape[0], 1))
    test_data = np.hstack((np.tile(np.hstack((dates, coords_full)), (2, 1)), indicator))

    mu, var = m.predict_f(test_data)

    results = pd.DataFrame(test_data)
    results.columns = ['date', 'lat', 'lon', 'indicator']
    results['mu'] = np.exp(mu)
    results['var'] = var
    # results['truth'] = np.exp(y_test[:, 0])
    # results['sq_error'] = np.square(results['mu'] - results['truth'])
    print(results.head())

    fname = 'demos/corregionalised_gp_nonsep_results_{}_sparse{}_linspace.csv'.format(data_name, n_sparse)
    results.to_csv(fname, index=False)
    
  # results['sq_error'].groupby(results.indicator).describe()

