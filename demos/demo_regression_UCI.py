import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(0)

import time

import matplotlib.pyplot as plt

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.mean_functions import Constant
from gpflow.models.sgpr import SGPR, GPRFITC
from gpflow.models.svgp import SVGP
from gpflow.models.gpr import GPR
from gpflow.training import AdamOptimizer, ScipyOptimizer, NatGradOptimizer
from gpflow.actions import Action, Loop

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from doubly_stochastic_dgp.dgp import DGP
from datasets import Datasets
datasets = Datasets(data_path='/data/')


data = datasets.all_datasets['kin8nm'].get_data()
X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
print('N: {}, D: {}, Ns: {}'.format(X.shape[0], X.shape[1], Xs.shape[0]))


def make_single_layer_models(X, Y, Z):
    D = X.shape[1]
    m_sgpr = SGPR(X, Y, RBF(D), Z.copy())
    m_svgp = SVGP(X, Y, RBF(D), Gaussian(), Z.copy())
    m_fitc = GPRFITC(X, Y, RBF(D), Z.copy())
    for m in m_sgpr, m_svgp, m_fitc:
        m.likelihood.variance = 0.01
    return [m_sgpr, m_svgp, m_fitc], ['{} {}'.format(n, len(Z)) for n in ['SGPR', 'SVGP', 'FITC']]

Z_100 = kmeans2(X, 100, minit='points')[0]
models_single_layer, names_single_layer = make_single_layer_models(X, Y, Z_100)


# ## DGP models
# 
# We'll include a DGP with a single layer here for comparision. We've used a largish minibatch size of $\text{min}(1000, N)$, but it works fine for smaller batches too
# 
# In the paper we used 1 sample. Here we'll go up to 5 in celebration of the new implementation (which is much more efficient)

# In[5]:


def make_dgp_models(X, Y, Z):
    models, names = [], []
    for L in range(1, 4):
        D = X.shape[1]

        # the layer shapes are defined by the kernel dims, so here all hidden layers are D dimensional 
        kernels = []
        for l in range(L):
            kernels.append(RBF(D))

        # between layer noise (doesn't actually make much difference but we include it anyway)
        for kernel in kernels[:-1]:
            kernel += White(D, variance=1e-5) 

        mb = 1000 if X.shape[0] > 1000 else None 
        model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=5, minibatch_size=mb)

        # start the inner layers almost deterministically 
        for layer in model.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-5

        models.append(model)
        names.append('DGP{} {}'.format(L, len(Z)))
    
    return models, names

models_dgp, names_dgp = make_dgp_models(X, Y, Z_100)


# ## Prediction
# 
# We'll calculate test rmse and likelihood in batches (so the larger datasets don't cause memory problems)
# 
# For the DGP models we need to take an average over the samples for the rmse. The `predict_density` function already does this internally
# 

# In[6]:


def batch_assess(model, assess_model, X, Y):
    n_batches = max(int(X.shape[0]/1000.), 1)
    lik, sq_diff = [], []
    for X_batch, Y_batch in zip(np.array_split(X, n_batches), np.array_split(Y, n_batches)):
        l, sq = assess_model(model, X_batch, Y_batch)
        lik.append(l)
        sq_diff.append(sq)
    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    return np.average(lik), np.average(sq_diff)**0.5

def assess_single_layer(model, X_batch, Y_batch):
    m, v = model.predict_y(X_batch)
    lik = np.sum(norm.logpdf(Y_batch*Y_std, loc=m*Y_std, scale=Y_std*v**0.5),  1)
    sq_diff = Y_std**2*((m - Y_batch)**2)
    return lik, sq_diff 

S = 100
def assess_sampled(model, X_batch, Y_batch):
    m, v = model.predict_y(X_batch, S)
    S_lik = np.sum(norm.logpdf(Y_batch*Y_std, loc=m*Y_std, scale=Y_std*v**0.5), 2)
    lik = logsumexp(S_lik, 0, b=1/float(S))
    
    mean = np.average(m, 0)
    sq_diff = Y_std**2*((mean - Y_batch)**2)
    return lik, sq_diff


# ## Training 
# 
# We'll optimize single layer models and using LFBGS and the dgp models with Adam. It will be interesting to compare the result of `m_svgp` compared to `m_dgp1`: if there is a difference it will be down to the optimizer. 
# 
# We'll show here also the reuslt of using a small and large number of iterations.

# In[7]:


iterations_few = 100
iterations_many = 5000
s = '{:<16}  lik: {:.4f}, rmse: {:.4f}'


# In[8]:


for iterations in [iterations_few, iterations_many]:
    print('after {} iterations'.format(iterations))
    for m, name in zip(models_single_layer, names_single_layer):
        ScipyOptimizer().minimize(m, maxiter=iterations)
        lik, rmse = batch_assess(m, assess_single_layer, Xs, Ys)
        print(s.format(name, lik, rmse))


# Now for the DGP models. First we use Adam for all parameters (as in the Doubly Stochastic VI for DGPs paper)

# In[9]:


for iterations in [iterations_few, iterations_many]:
    print('after {} iterations'.format(iterations))
    for m, name in zip(models_dgp, names_dgp):
        AdamOptimizer(0.01).minimize(m, maxiter=iterations)
        lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
        print(s.format(name, lik, rmse))


# We can also use natural gradients for the final layer, which can help considerably. 

# In[10]:


for iterations in [iterations_few, iterations_many]:
    print('after {} iterations'.format(iterations))
    for m, name in zip(models_dgp, names_dgp):
        ng_vars = [[m.layers[-1].q_mu, m.layers[-1].q_sqrt]]
        for v in ng_vars[0]:
            v.set_trainable(False)    
        ng_action = NatGradOptimizer(gamma=0.1).make_optimize_action(m, var_list=ng_vars)
        adam_action = AdamOptimizer(0.01).make_optimize_action(m)

        Loop([ng_action, adam_action], stop=iterations)()

        lik, rmse = batch_assess(m, assess_sampled, Xs, Ys)
        print(s.format(name, lik, rmse))


# Note that even after 100 iterations we get a good result, which is not the case using ordinary gradients.
