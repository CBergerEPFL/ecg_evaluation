#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:43:49 2021

@author: adriano
"""

#%%
import os
import sys
import numpy as np
import time
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import pdb

sys.path.append(os.path.join(os.getcwd(), ".."))
from shared_utils import colorednoise as cn


CWD = os.getcwd()

plt.close("all")

#%%
def _load_data(file_name):
    # TO BE DONE
    t_obs = []
    X_obs = []
    return t_obs, X_obs


def _gen_data(
    system="lorenz_stochastic",
    observables=[0],
    length=10000,
    x0=None,
    p=[10.0, 8.0 / 3.0, 28.0],
    step=0.001,
    sample=0.03,
    discard=1000,
    epsilon_sde=0.0,
    epsilon_add=0.0,
    beta_add=0.0,
):

    if system == "lorenz_stochastic":
        t_obs, X_obs, t_gen, X_gen = _gen_lorenz_stochastic(
            length=length,
            x0=x0,
            sigma=p[0],
            beta=p[1],
            rho=p[2],
            step=step,
            sample=sample,
            discard=discard,
            epsilon_sde=epsilon_sde,
            observables=observables,
        )
    elif system == "rossler_stochastic":
        t_obs, X_obs, t_gen, X_gen = _gen_rossler_stochastic(
            length=length,
            x0=x0,
            a=p[0],
            b=p[1],
            c=p[2],
            step=step,
            sample=sample,
            discard=discard,
            epsilon_sde=epsilon_sde,
            observables=observables,
        )
    elif system == "noise":
        Nx = len(observables)
        t_obs = np.arange(0, length * sample, sample)
        X_obs = np.zeros((t_obs.shape[0], Nx))
        t_gen = np.arange(0, length * step, step)
        X_gen = np.zeros((t_gen.shape[0], Nx))

    Nt, Nx = X_obs.shape
    for ii in np.arange(Nx):
        X_obs[:, ii] = X_obs[:, ii] + epsilon_add * cn.powerlaw_psd_gaussian(
            beta_add, Nt
        )
    # self.X_obs = self.X_obs + \
    # 		epsilon_add * np.random.randn(self.X_obs.shape[0],
    # 								     self.X_obs.shape[1])

    return t_obs, X_obs, t_gen, X_gen


def _gen_lorenz_stochastic(
    length=10000,
    x0=None,
    sigma=10.0,
    beta=8.0 / 3.0,
    rho=28.0,
    step=0.001,
    sample=0.03,
    discard=1000,
    epsilon_sde=0.0,
    observables=[0],
):

    if not x0:
        x0 = (0.0, -0.01, 9.0) + 0.25 * (-1 + 2 * np.random.random(3))

    sample = int(sample / step)
    t = np.linspace(
        0, (sample * (length + discard)) * step, sample * (length + discard)
    )
    Nt_simulation = len(t)
    X = np.zeros((Nt_simulation, 3))
    X[0, :] = x0

    for tt in range(Nt_simulation - 1):
        r = np.random.randn(3)
        X[tt + 1, 0] = (
            X[tt, 0]
            + (sigma * (X[tt, 1] - X[tt, 0])) * step
            + epsilon_sde * np.sqrt(step) * r[0]
        )
        X[tt + 1, 1] = (
            X[tt, 1]
            + (X[tt, 0] * (rho - X[tt, 2]) - X[tt, 1]) * step
            + epsilon_sde * np.sqrt(step) * r[1]
        )
        X[tt + 1, 2] = (
            X[tt, 2]
            + (X[tt, 0] * X[tt, 1] - beta * X[tt, 2]) * step
            + epsilon_sde * np.sqrt(step) * r[2]
        )
    t_gen = t[discard * sample :]
    t_gen = t_gen - t_gen[0]
    X_gen = X[discard * sample :, :]
    t_obs = t[discard * sample :: sample]
    t_obs = t_obs - t_obs[0]
    X_obs = X[discard * sample :: sample, observables]
    return t_obs, X_obs, t_gen, X_gen


def _gen_rossler_stochastic(
    length=10000,
    x0=None,
    a=0.2,
    b=0.2,
    c=5.7,
    step=0.001,
    sample=0.03,
    discard=1000,
    epsilon_sde=0.0,
    observables=[0],
):

    if not x0:
        x0 = (-9.0, 0.0, 0.0) + 0.25 * (-1 + 2 * np.random.random(3))

    sample = int(sample / step)
    t = np.linspace(
        0, (sample * (length + discard)) * step, sample * (length + discard)
    )

    Nt_simulation = len(t)
    X = np.zeros((Nt_simulation, 3))
    X[0, :] = x0

    for tt in range(Nt_simulation - 1):
        r = np.random.randn(1)
        X[tt + 1, 0] = (
            X[tt, 0] - (X[tt, 1] + X[tt, 2]) * step + epsilon_sde * np.sqrt(step) * r[0]
        )
        X[tt + 1, 1] = X[tt, 1] + (X[tt, 0] + a * X[tt, 1]) * step
        X[tt + 1, 2] = X[tt, 2] + (b + X[tt, 2] * (X[tt, 0] - c)) * step
        # if np.isnan(X[tt + 1, 0]):
        #     pdb.set_trace()
        # if np.isnan(X[tt + 1, 1]):
        #     pdb.set_trace()
        # if np.isnan(X[tt + 1, 2]):
        #     pdb.set_trace()
    t_gen = t[discard * sample :]
    t_gen = t_gen - t_gen[0]
    X_gen = X[discard * sample :, :]
    t_obs = t[discard * sample :: sample]
    t_obs = t_obs - t_obs[0]
    X_obs = X[discard * sample :: sample, observables]

    return t_obs, X_obs, t_gen, X_gen


## parametrs

## Main runner line
# system      = 'lorenz_stochastic'
# p           = (10.0, 8.0/3.0, 28.0)
# system      = 'rossler_stochastic'
# p           = (0.2, 0.2, 5.7)
# observables = [0,1,2]
# length      = 10000
# x0          = None
# step        = 0.001
# sample      = 0.01
# discard     = 1000
# epsilon_sde = 0.0
# epsilon_add = 0.0
# beta_add    = 0.0

# np.random.seed(42)

# t_obs, X_obs, t_gen, X_gen = _gen_data(
# 			system=system,
# 			observables=observables,
# 			length=length,
# 			x0=x0,
# 			p=p, step=step,
# 			sample=sample,
# 			discard=discard,
# 			epsilon_sde=epsilon_sde,
# 			epsilon_add=epsilon_add,
# 			beta_add=beta_add
# 	)

# plt.figure()
# ax1 = plt.subplot(311)
# ax1.plot(t_obs,X_obs[:,0],'.k')
# ax1.plot(t_gen,X_gen[:,0],'.r')
# ax2 = plt.subplot(312, sharex=ax1)
# ax2.plot(t_obs,X_obs[:,1],'.k')
# ax2.plot(t_gen,X_gen[:,1],'.r')
# ax3 = plt.subplot(313, sharex=ax1)
# ax3.plot(t_obs,X_obs[:,2],'.k')
# ax3.plot(t_gen,X_gen[:,2],'.r')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(X_obs[:,0],X_obs[:,1],X_obs[:,2],'.k')
# plt.plot(X_gen[:,0],X_gen[:,1],X_gen[:,2],'.r')
# plt.show()
