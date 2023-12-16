# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: 'Python 3.10.10 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quick start guide

# %% [markdown]
# This notebook illustrates many library functionalities on a simple example.
# The data we study is a kaggle dataset corresponding to aerodynamic and acoustic tests
# of two and three-dimensional airfoils (see the
# [data card](https://www.kaggle.com/datasets/fedesoriano/airfoil-selfnoise-dataset) for more details).
#
# The notebook is structured in three parts
#  * Data exploration
#  * Fitting a Gaussian Process model using _gpflow_
#  * Visualising and interacting with the model using _inferlylens_
#
# To start with, we import the required libraries

# %%
import gpflow
import numpy as np
import pandas as pd

import inferlylens as infly
import inferlylens.data.plots as dataplot
import inferlylens.models.plots as modelplot

# %% [markdown]
# We now load the data, and define what will be the inputs and outputs for this dataset.
# Note that the input and output names must correspond to columns of the dataframe.

# %%
df = pd.read_parquet("../datasets/airfoil.parquet")

# we do some basic feature engineering
df["Log frequency"] = np.log(df["Frequency"])

input_names = [
    "Log frequency",
    "Angle of attack",
    "Chord length",
    "Free-stream velocity",
    "Displacement thickness",
]
output_names = ["Sound pressure"]

df["Frequency"] = np.log(df["Frequency"])

print("data shape:", df.shape)
print(df.head())

# %% [markdown]
# ## Data exploration
#
# First, let's investigate the input distribution and the input/output relationship.

# %%
fig = dataplot.pairsplot(df, input_names, opacity=0.2, title="Input distribution", width=900, height=900)
fig.show()

# %%
fig = dataplot.gridplot(df, input_names, output_names)
fig.update_layout(title="Input vs output", width=900, height=300)
fig.show()

# %% [markdown]
# ## Fitting a GP model
#
# In this section, we use _inferlylens_ to help with data preparation (train/test split and data transform)
# and _gpflow_ to fit a Gaussian process regression model.

# %%
dataset = infly.data.Dataset(df, input_names, output_names)
trainset, testset = dataset.split([1300, df.shape[0] - 1300], ["train", "test"])

print(dataset.df.head())
print("trainset shape:", trainset.df.shape)
print("testset shape:", testset.df.shape)

## Separate training input and outputs
X = trainset.df[input_names]
Y = trainset.df[output_names]


# %% [markdown]
# When fitting a GP model, it is common practice to rescale the input and output data to improve the
# robustness of the training.

# %%
# map the training inputs to the unit cube
input_transform = infly.data.transforms.unit_cube_rescaling(X)  # this is a tensorflow Bijector
Xt = input_transform.forward(X)

# map training output to zero mean and unit variance
output_transform = infly.data.transforms.standardise_rescaling(Y)
Yt = output_transform.forward(Y)

# initialise the GP model. Note that this is a standard GPflow model
kernel = gpflow.kernels.Matern52(lengthscales=0.1 * np.ones(len(input_names)), variance=1.0)
gpr = gpflow.models.GPR((Xt, Yt), kernel)
gpr.likelihood.variance.assign(1e-4)

# train the GP model
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(gpr.training_loss, gpr.trainable_variables, options={"maxiter": 2000})
print(opt_logs)

# get model summary
gpflow.utilities.print_summary(gpr)

# %% [markdown]
# ## Using _inferlylens_ to interact with the GP model
#
# The main disadvantage of rescaling the input and outputs is that it is then necessary to map input and
# the model predictions back into the original space to obtain meaningful values. To make this straightfoward,
# _inferlylens_ offers a thin wrapper around gpflow models that take care of all input/output remappings.

# %%
gpmodel = infly.models.GPmodel(gpr, input_transform, output_transform)

# the three methods associated with the GPmodel class are:
pred, var = gpmodel.predict_y(Xnew=testset.df[input_names])
quantiles = gpmodel.predict_quantiles(Xnew=testset.df[input_names], levels=np.array([0.05, 0.95]))
log_lik = gpmodel.predict_log_density(data=testset)

# %% [markdown]
# _inferlylens_ comes with some default plots to get some insights into the model

# %%
fig = modelplot.plot_lengthscales(gpmodel.gpflow.kernel, input_names, range_r=[0.0, 1.5])
fig.update_layout(title="Lengthscales")
fig.show()

# %%
fig = modelplot.plot_actual_vs_predicted(gpmodel, testset)
fig.update_layout(title="Actual vs predicted", width=800, height=800)
fig.show()

# %%
ref_pt = np.mean(np.asarray(X), axis=0, keepdims=True)
input_range = input_transform.inverse(np.array([[0, 1]] * 5).T)

fig = modelplot.plot_slices(gpmodel, input_names, output_names, reference_point=ref_pt, xlim=input_range)
fig.update_layout(title="Prediction slices", width=1500, height=400)
fig.show()

# %% [markdown]
# Finally, we can compute some modelling performance metrics:

# %%
q2 = infly.models.q2(gpmodel, testset)
nlpd = infly.models.nlpd(gpmodel, testset)
rmse = infly.models.rmse(gpmodel, testset)

print(pd.DataFrame({"q2": q2, "nlpd": nlpd, "rmse": rmse}))
