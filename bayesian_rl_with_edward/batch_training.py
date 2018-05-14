
# coding: utf-8

# # Batch Training
# 
# Running algorithms which require the full data set for each update
# can be expensive when the data is large. In order to scale inferences,
# we can do _batch training_. This trains the model using
# only a subsample of data at a time.
# 
# In this tutorial, we extend the
# [supervised learning tutorial](http://edwardlib.org/tutorials/supervised-regression), 
# where the task is to infer hidden structure from
# labeled examples $\{(x_n, y_n)\}$.
# A webpage version is available at
# http://edwardlib.org/tutorials/batch-training.

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from edward.models import Normal

def visualise(X_data, y_data, w, b, n_samples=10):
  w_samples = w.sample(n_samples)[:, 0].eval()
  b_samples = b.sample(n_samples).eval()
  plt.scatter(X_data[:, 0], y_data)
  plt.ylim([-10, 10])
  inputs = np.linspace(-8, 8, num=400)
  for ns in range(n_samples):
    output = inputs * w_samples[ns] + b_samples[ns]
    plt.plot(inputs, output)
  plt.grid()
  plt.show() 



# ## Data
# 
# Simulate $N$ training examples and a fixed number of test examples.
# Each example is a pair of inputs $\mathbf{x}_n\in\mathbb{R}^{10}$ and
# outputs $y_n\in\mathbb{R}$. They have a linear dependence with
# normally distributed noise.
# 
# We also define a helper function to select the next batch of data
# points from the full set of examples. It keeps track of the current
# batch index and returns the next batch using the function 
# ``next()``. We will generate batches from `data` during inference.

# In[2]:


def build_toy_dataset(N, w):
  D = len(w)
  x = np.random.normal(0.0, 2.0, size=(N, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.05, size=N)
  return x, y


# In[3]:


def generator(arrays, batch_size):
  """Generate batches, one with respect to each array's first axis."""
  starts = [0] * len(arrays)  # pointers to where we are in iteration
  while True:
    batches = []
    for i, array in enumerate(arrays):
      start = starts[i]
      stop = start + batch_size
      diff = stop - array.shape[0]
      if diff <= 0:
        batch = array[start:stop]
        starts[i] += batch_size
      else:
        batch = np.concatenate((array[start:], array[:diff]))
        starts[i] = diff
      batches.append(batch)
    yield batches


# In[4]:


ed.set_seed(42)

N = 10000  # size of training data
M = 128    # batch size during training
D = 10     # number of features

w_true = np.ones(D) * 5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(235, w_true)

data = generator([X_train, y_train], M)


# ## Model
# 
# Posit the model as Bayesian linear regression (Murphy, 2012).
# For a set of $N$ data points $(\mathbf{X},\mathbf{y})=\{(\mathbf{x}_n, y_n)\}$,
# the model posits the following distributions:
# 
# \begin{align*}
#   p(\mathbf{w})
#   &=
#   \text{Normal}(\mathbf{w} \mid \mathbf{0}, \sigma_w^2\mathbf{I}),
#   \\[1.5ex]
#   p(b)
#   &=
#   \text{Normal}(b \mid 0, \sigma_b^2),
#   \\
#   p(\mathbf{y} \mid \mathbf{w}, b, \mathbf{X})
#   &=
#   \prod_{n=1}^N
#   \text{Normal}(y_n \mid \mathbf{x}_n^\top\mathbf{w} + b, \sigma_y^2).
# \end{align*}
# 
# The latent variables are the linear model's weights $\mathbf{w}$ and
# intercept $b$, also known as the bias.
# Assume $\sigma_w^2,\sigma_b^2$ are known prior variances and $\sigma_y^2$ is a
# known likelihood variance. The mean of the likelihood is given by a
# linear transformation of the inputs $\mathbf{x}_n$.
# 
# Let's build the model in Edward, fixing $\sigma_w,\sigma_b,\sigma_y=1$. 

# In[5]:


X = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=1.0)


# Here, we define a placeholder `X`. During inference, we pass in
# the value for this placeholder according to batches of data.
# To enable training with batches of varying size, 
# we don't fix the number of rows for `X` and `y`. (Alternatively,
# we could fix it to be the batch size if training and testing 
# with a fixed size.)

# ## Inference
# 
# We now turn to inferring the posterior using variational inference.
# Define the variational model to be a fully factorized normal across
# the weights.

# In[6]:


qw = Normal(loc=tf.get_variable("qw/loc", [D]),
            scale=tf.nn.softplus(tf.get_variable("qw/scale", [D])))
qb = Normal(loc=tf.get_variable("qb/loc", [1]),
            scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))


# Run variational inference with the Kullback-Leibler divergence.
# We use $5$ latent variable samples for computing
# black box stochastic gradients in the algorithm.
# (For more details, see the
# [$\text{KL}(q\|p)$ tutorial](http://edwardlib.org/tutorials/klqp).)
# 
# For batch training, we will iterate over the number of batches and
# feed them to the respective placeholder. We set the number of
# iterations to be equal to the number of batches times the number of
# epochs (full passes over the data set).

# In[7]:


n_batch = int(N / M)
print(n_batch)
n_epoch = 5

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})
inference.initialize(n_iter=n_batch * n_epoch, n_samples=5, scale={y: N / M})
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  X_batch, y_batch = next(data)
  info_dict = inference.update({X: X_batch, y_ph: y_batch})
  inference.print_progress(info_dict)
visualise(X_batch, y_batch, w, b, n_samples=10)
visualise(X_batch, y_batch, qw, qb, n_samples=10)


# When initializing inference, note we scale $y$ by $N/M$, so it is as if the
# algorithm had seen $N/M$ as many data points per iteration.
# Algorithmically, this will scale all computation regarding $y$ by
# $N/M$ such as scaling the log-likelihood in a variational method's
# objective. (Statistically, this avoids inference being dominated by the prior.)
# 
# The loop construction makes training very flexible. For example, we
# can also try running many updates for each batch.

# In[8]:


n_batch = int(N / M)
n_epoch = 1

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})
inference.initialize(
    n_iter=n_batch * n_epoch * 10, n_samples=5, scale={y: N / M})
tf.global_variables_initializer().run()

for _ in range(inference.n_iter // 10):
  X_batch, y_batch = next(data)
  for _ in range(10):
    #visualise(X_batch, y_batch, qw, qb, n_samples=10)
    info_dict = inference.update({X: X_batch, y_ph: y_batch})

  inference.print_progress(info_dict)

visualise(X_batch, y_batch, w, b, n_samples=10)
visualise(X_batch, y_batch, qw, qb, n_samples=10)



# In general, make sure that the total number of training iterations is 
# specified correctly when initializing `inference`. Otherwise an incorrect
# number of training iterations can have unintended consequences; for example,
# `ed.KLqp` uses an internal counter to appropriately decay its optimizer's 
# learning rate step size.
# 
# Note also that the reported `loss` value as we run the
# algorithm corresponds to the computed objective given the current
# batch and not the total data set. We can instead have it report
# the loss over the total data set by summing `info_dict['loss']`
# for each epoch.

# ## Criticism
# 
# A standard evaluation for regression is to compare prediction accuracy on
# held-out "testing" data. We do this by first forming the posterior predictive
# distribution.

# In[9]:


y_post = ed.copy(y, {w: qw, b: qb})
# This is equivalent to
# y_post = Normal(loc=ed.dot(X, qw) + qb, scale=tf.ones(N))


# With this we can evaluate various quantities using predictions from
# the model (posterior predictive).

# In[10]:


print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))


# ## Footnotes
# 
# Only certain algorithms support batch training such as
# `MAP`, `KLqp`, and `SGLD`. Also, above we
# illustrated batch training for models with only global latent variables,
# which are variables are shared across all data points.
# For more complex strategies, see the
# [inference data subsampling API](http://edwardlib.org/api/inference-data-subsampling).
