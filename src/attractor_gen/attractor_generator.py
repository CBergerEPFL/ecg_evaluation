import os
import sys
import time
from math import cos, fabs, sin, sqrt

import datashader as ds
import desolver as de
import desolver.backend as D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit
from scipy.interpolate import interp1d


dict_parameters = {
    "chua": {
        "a": [15.6, 15.6],
        "b": [25, 51],
        "mu0": [-1.143, -1.143],
        "mu1": [-0.714, -0.714],
    },
    "duffing": {"a": [0.1, 0.1], "b": [0.1, 0.65]},
    "lorenz": {"sigma": [10, 10], "beta": [8 / 3, 8 / 3], "rho": [28, 100]},
    "rikitake": {"a": [2, 7], "b": [3, 3], "c": [5, 5], "d": [0.75, 0.75]},
    "rossler": {"a": [0.2, 0.2], "b": [0.2, 0.2], "c": [4, 18]},
    "becker": {
        "beta1": [1.0, 1.0],
        "beta2": [0.84, 0.84],
        "rho": [0.048, 0.3],
        "kappaprime": [0.8525, 0.922],  # 0.922 #0.9
        "nu": [4.2164 * 10**-5, 4.2164 * 10**-5],  # 4.2164*10**-5"
    },
}


class Attractor_Generator(object):
    def __init__(self, params=None):
        self.params = params
        self.available_attractors_2d = {
            "bedhead": bedhead,
            "clifford": clifford,
            "de_jong": de_jong,
            "fractal_dream": fractal_dream,
            "gumowski_mira": gumowski_mira,
            "hopalong1": hopalong1,
            "hopalong2": hopalong2,
            "svensson": svensson,
            "symmetric_icon": symmetric_icon,
        }
        self.available_attractors_3d = {
            "chua": chua,
            "duffing": duffing,
            "duffing_map": duffing_map,
            "lorenz": lorenz,
            "lotka_volterra": lotka_volterra,
            "nose_hover": nose_hover,
            "rikitake": rikitake,
            "rossler": rossler,
            "wang": wang,
            "becker": becker,
        }

    def print_available_attractors(self):
        print("Available 2D attractors: ", self.available_attractors_2d.keys())
        print("Available 3D attractors: ", self.available_attractors_3d.keys())

    def attractors_2d(self):
        return list(self.available_attractors_2d.keys())

    def attractors_3d(self):
        return list(self.available_attractors_3d.keys())

    def compute_attractor(
        self, attractor, x0=0, y0=0, z0=0, dt=0.01, num_steps=1000, var_params=False
    ):
        self.dt = dt
        self.attractor = attractor
        if attractor in self.available_attractors_2d.keys():
            self.attractor_dimension = "2d"
            attr_fn = self.available_attractors_2d[attractor]
            self.coordinates, self.t = compute_attractor_2d(
                attr_fn,
                x0,
                y0,
                dt,
                num_steps,
            )
        elif attractor in self.available_attractors_3d.keys():
            self.attractor_dimension = "3d"
            attr_fn = self.available_attractors_3d[attractor]
            self.coordinates, self.t = compute_attractor_3d(
                attr_fn, x0, y0, z0, dt, num_steps, var_params
            )
        elif attractor == "random_noise3d":
            self.attractor_dimension = "3d"
            self.coordinates = np.random.normal(
                scale=1 / np.sqrt(3), size=(num_steps + 1, 3)
            )

            # self.xs = np.random.normal(scale = 1/np.sqrt(3), size = num_steps+1)
            # self.ys = np.random.normal(scale = 1/np.sqrt(3), size = num_steps+1)
            # self.zs = np.random.normal(scale = 1/np.sqrt(3), size = num_steps+1)
            self.t = np.arange(num_steps + 1)
        else:
            print("Unknown attractor. Available attractors are: ")
            self.available_attractor()
            sys.exit(2)

    def generate_data_file(self, path, filename, format="csv"):

        if self.attractor_dimension == "2d":
            dict_data = {
                "t": self.t,
                "x": self.coordinates[:, 0],
                "y": self.coordinates[:, 1],
            }
        elif self.attractor_dimension == "3d":
            dict_data = {
                "t": self.t,
                "x": self.coordinates[:, 0],
                "y": self.coordinates[:, 1],
                "z": self.coordinates[:, 2],
            }
        else:
            print("Unknown attractor. Available attractors are: ")
            self.available_attractor()
            sys.exit(2)

        df = pd.DataFrame(dict_data)
        df = df.set_index(["t"], drop=True)
        if not os.path.exists(path):
            os.makedirs(path)

        if format.lower() == "csv":
            df.to_csv(os.path.join(path, filename + ".csv"))
        elif format.lower() == "parquet":
            df.to_parquet(os.path.join(path, filename + ".parquet"))
        else:
            print(format.lower(), "not recognized.")
            print("Available formats are: `csv` and `parquet`")
            print("Using default `csv`")
            df.to_csv(os.path.join(path, filename + ".csv"))


def return_attractors_parameters(attractor):
    dict_attractor = dict_parameters[attractor.__name__]
    dict_return = {}
    for key in dict_attractor.keys():
        limits = dict_attractor[key]
        dict_return[key] = np.random.uniform(limits[0], limits[1])

    return dict_return


# Compute either 2d or 3d attractors
# ============================================================================


@jit(nopython=True)
def compute_attractor_2d(attractor, x0, y0, dt, num_steps):
    # Step through "time", calculating the partial derivatives
    # at the current point and using them to estimate the next point
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    t = np.empty(num_steps + 1)
    xs[0] = x0
    ys[0] = y0
    t[0] = 0
    for i in range(num_steps):
        x_dot, y_dot = attractor(x=xs[i], y=ys[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        t[i + 1] = (i + 1) * dt

        coordinates = np.array(xs)
    return coordinates, t


def compute_attractor_3d(attractor, x0, y0, z0, dt, num_steps, var_params):
    # Step through "time", calculating the partial derivatives
    # at the current point and using them to estimate the next point
    # Need one more for the initial values

    t0 = 0.0
    ind_tfin = num_steps * dt - t0
    y_init = D.array([x0, y0, z0])
    print(y_init)
    print(ind_tfin)
    rhs = attractor
    if var_params:
        dict_parameters = return_attractors_parameters(attractor)
        a = de.OdeSystem(
            rhs,
            y0=y_init,
            dense_output=True,
            t=(t0, ind_tfin),
            rtol=1e-9,
            atol=1e-9,
            constants=dict_parameters,
        )

    else:
        a = de.OdeSystem(
            rhs, y0=y_init, dense_output=True, t=(t0, ind_tfin), rtol=1e-9, atol=1e-9
        )

    a.method = "RK45CK"
    a.integrate()
    t_ode = a.t
    dt = 0.01
    t = np.arange(t_ode[0], t_ode[-1], dt)
    interp = interp1d(t_ode, a.y, kind="cubic", axis=0)
    coordinates = interp(t)
    print(coordinates)
    return coordinates, t


# ============================================================================


# Two-dimensional attractors (private)
# ==============================================================================


@jit(nopython=True)
def bedhead(x, y, dt, a=-0.81, b=-0.92):
    """
    Bedhead attractor.

    Parameters
    ==========
    a, b - are bedhead system parameters.
    Default values are:
            - x0 = 1., y0 = 1., a = -0.81, b = -0.92

    Other useful combinations are:
    1) x0 = 1., y0 = 1., a = -0.64, b = 0.76
    2) x0 = 1., y0 = 1., a =  0.06, b = 0.98
    3) x0 = 1., y0 = 1., a = -0.67, b = 0.83
    """
    x_out = sin(x * y / b) * y + cos(a * x - y)
    y_out = x + sin(y) / b
    return x_out, y_out


@jit(nopython=True)
def clifford(x, y, dt, a=-1.3, b=-1.3, c=-1.8, d=-1.9):
    """
    Clifford attractor.

    Parameters
    ==========
    a, b, c, d - are clifford system parameters.
    Default values are:
            - x0 = 0, y0 = 0, a = -1.3, b = -1.3, c = -1.8, d = -1.9

    Other useful combinations are:
    1)  x0 = 0, y0 = 0, a = -1.4 , b =  1.6 , c =  1.0 , d =  0.7
    2)  x0 = 0, y0 = 0, a =  1.7 , b =  1.7 , c =  0.6 , d =  1.2
    3)  x0 = 0, y0 = 0, a =  1.7 , b =  0.7 , c =  1.4 , d =  2.0
    4)  x0 = 0, y0 = 0, a = -1.7 , b =  1.8 , c = -1.9 , d = -0.4
    5)  x0 = 0, y0 = 0, a =  1.1 , b = -1.32, c = -1.03, d =  1.54
    6)  x0 = 0, y0 = 0, a =  0.77, b =  1.99, c = -1.31, d = -1.45
    7)  x0 = 0, y0 = 0, a = -1.9 , b = -1.9 , c = -1.9 , d = -1.0
    8)  x0 = 0, y0 = 0, a =  0.75, b =  1.34, c = -1.93, d =  1.0
    9)  x0 = 0, y0 = 0, a = -1.32, b = -1.65, c =  0.74, d =  1.81
    10) x0 = 0, y0 = 0, a =  -1.6, b =  1.6 , c =  0.7 , d = -1.0
    11) x0 = 0, y0 = 0, a =  -1.7, b =  1.5 , c = -0.5 , d =  0.7
    """
    x_out = sin(a * y) + c * cos(a * x)
    y_out = sin(b * x) + d * cos(b * y)
    return x_out, y_out


@jit(nopython=True)
def de_jong(x, y, dt, a=-1.244, b=-1.251, c=-1.815, d=-1.908):
    """
    De_jong attractor.

    Parameters
    ==========
    a, b, c, d - are de_jong system parameters.
    Default values are:
            - x0 = 0, y0 = 0, a = -1.244, b = -1.251, c = -1.815, d = -1.908

    Other useful combinations are:
    1) x0 = 0, y0 = 0, a =  1.7  , b =  1.7  , c =  0.6  , d =  1.2
    2) x0 = 0, y0 = 0, a =  1.4  , b = -2.3  , c =  2.4  , d = -2.1
    3) x0 = 0, y0 = 0, a = -2.7  , b = -0.09 , c = -0.86 , d = -2.2
    4) x0 = 0, y0 = 0, a = -0.827, b = -1.637, c =  1.659, d = -0.943
    5) x0 = 0, y0 = 0, a = -2.24 , b =  0.43 , c = -0.65 , d = -2.43
    6) x0 = 0, y0 = 0, a =  2.01 , b = -2.53 , c =  1.61 , d = -0.33
    7) x0 = 0, y0 = 0, a =  1.4  , b =  1.56 , c =  1.4  , d = -6.56
    """
    x_out = sin(a * y) - cos(b * x)
    y_out = sin(c * x) - cos(d * y)
    return x_out, y_out


@jit(nopython=True)
def fractal_dream(x, y, dt, a=-0.966918, b=2.879879, c=0.765145, d=0.744728):
    """
    Fractal_dream attractor.
    a, b, c, d - are fractal_dream system parameters.
    Default values are:
            - x0 = 0.1, y0 = 0.1, a = -0.966918, b = 2.879879, c = 0.765145, d = 0.744728

    Other useful combinations are:
    1) x0 = 0.1, y0 = 0.1, a = -2.8276, 1.2813, 1.9655, 0.5 97
    2) x0 = 0.1, y0 = 0.1, a =  -1.1554, -2.3419, -1.9799, 2.1828
    3) x0 = 0.1, y0 = 0.1, a = -1.9956, -1.4528, -2.6206, 0.8517
    """
    x_out = sin(y * b) + c * sin(x * b)
    y_out = sin(x * a) + d * sin(y * a)
    return x_out, y_out


@jit(nopython=True)
def gumowski_mira(x, y, dt, a=0.0, b=0.5, mu=-0.75):
    """
    Gumowski_Mira attractor.

    Parameters
    ==========
    a, b, mu - are gumowski_mira system parameters.
    Default values are:
            - x0 = 0.1, y0 = 0.1, a = 0.0, b = 0.5, mu = -0.75

    Other useful combinations are:
    1)  x0 = 0  , y0 = 1  , a = 0.008, b = 0.05, mu = -0.496
    2)  x0 = 0.1, y0 = 0.1, a = 0.0  , b = 0.5 , mu = -0.7509
    3)  x0 = 0  , y0 = 1  , a = 0.0  , b = 0.5 , mu = -0.22
    4)  x0 = 0  , y0 = 1  , a = 0.008, b = 0.05, mu = -0.9
    5)  x0 = 0  , y0 = 1  , a = 0.008, b = 0.05, mu = -0.45
    6)  x0 = 0.1, y0 = 0.1, a = 0.008, b = 0.05, mu =  0.16
    7)  x0 = 0  , y0 = 0.5, a = 0.008, b = 0.05, mu = -0.7
    8)  x0 = 0.5, y0 = 0  , a = 0.0  , b = 0.05, mu = -0.2
    9)  x0 = 0.5, y0 = 0.5, a = 0.0  , b = 0.05, mu = -0.22
    10) x0 = 0  , y0 = 0.5, a = 0.0  , b = 0.05, mu = -0.31
    11) x0 = 0  , y0 = 0.5, a = 0.0  , b = 0.05, mu = -0.55
    12) x0 = 0.5, y0 = 0.5, a = 0.0  , b = 0.05, mu = -0.23
    13) x0 = 0.5, y0 = 0.5, a = 0.009, b = 0.05, mu =  0.32
    14) x0 = 0.1, y0 = 0.1, a = 0.0  , b = 0.5 , mu = -0.65
    15) x0 = 0.0, y0 = 0.5, a = 0.0  , b = 0   , mu = -0.578
    16) x0 = 0.0, y0 = 0.5, a = 0.0  , b = 0   , mu = -0.604
    17) x0 = 0.0, y0 = 0.5, a = 0.0  , b = 0   , mu =  0.228
    18) x0 = 0.0, y0 = 0.5, a = 0.0  , b = 0   , mu = -0.002
    19) x0 = 0.0, y0 = 0.5, a = 0.0  , b = 0   , mu = -0.623
    """

    def G(x, mu):
        return mu * x + 2 * (1 - mu) * x**2 / (1.0 + x**2)

    x_out = y + a * (1 - b * y**2) * y + G(x, mu)
    y_out = -x + G(x_out, mu)
    return x_out, y_out


@jit(nopython=True)
def hopalong1(x, y, dt, a=2.0, b=1.0, c=0.0):
    """
    Hopalong1 attractor.

    Parameters
    ==========
    a, b, c - are hopalong1 system parameters.
    Default values are:
            - x0 = 0, y0 = 0, a = 2, b = 1, c = 0

    Other useful combinations are:
    1) x0 = 0, y0 = 0, a = -11.0, b = 0.05, c = 0.5
    2) x0 = 0, y0 = 0, a =  2.0 , b = 0.05, c = 2.0
    3) x0 = 0, y0 = 0, a =  1.1 , b = 0.5 , c = 1.0
    """
    x_out = y - sqrt(fabs(b * x - c)) * np.sign(x)
    y_out = a - x
    return x_out, y_out


@jit(nopython=True)
def hopalong2(x, y, dt, a=7.17, b=8.44, c=2.56):
    """
    Hopalong2 attractor.

    Parameters
    ==========
    a, b, c - are hopalong2 system parameters.
    Default values are:
            - x0 = 0, y0 = 0, a = 7.17, b = 8.44, c = 2.56

    Other useful combinations are:
    1) x0 = 0, y0 = 0, a = 7.8, b = 0.13, c = 8.15
    2) x0 = 0, y0 = 0, a = 9.7, b = 1.6 , c = 7.9
    3) x0 = 0, y0 = 0, a = 1.1, b = 0.5 , c = 1.0
    """
    x_out = y - 1.0 - sqrt(fabs(b * x - 1.0 - c)) * np.sign(x - 1.0)
    y_out = a - x - 1.0
    return x_out, y_out


@jit(nopython=True)
def svensson(x, y, dt, a=1.5, b=-1.8, c=1.6, d=0.9):
    """
    Svensson attractor.

    Parameters
    ==========
    a, b, c, d - are svensson system parameters.
    Default values are:
            - x0 = 0, y0 = 0, a = 1.5, b = -1.8, c = 1.6, d = 0.9

    Other useful combinations are:
    1) x0 = 0, y0 = 0, a = -1.78, b =  1.29, c = -0.09, d = -1.18
    2) x0 = 0, y0 = 0, a = -0.91, b = -1.29, c = -1.97, d = -1.56
    3) x0 = 0, y0 = 0, a =  1.4 , b =  1.56, c =  1.4 , d = -6.56
    """
    x_out = d * sin(a * x) - sin(b * y)
    y_out = c * cos(a * x) + cos(b * y)
    return x_out, y_out


@jit(nopython=True)
def symmetric_icon(x, y, dt, a=1.8, b=0.0, g=1.0, om=0.1, l=-1.93, d=5):
    """
    Symmetric_Icon attractor.

    Parameters
    ==========
    a, b, g, om, l, d - are symmetric_icon system parameters.
    Default values are:
            - x0 = 0.01, y0 = 0.01, a = 1.8, b = 0.0,
              g = 1.0, om = 0.1, l = -1.93, d = 5

    Other useful combinations are:
    1)  x0 = 0.01, y0 = 0.01, a = 5.0, b = -1.0,
            g = 1.0, om = 0.188, l = -2.5, d = 5
    2)  x0 = 0.01, y0 = 0.01, a = -1.0, b = 0.1,
            g = -0.82, om = 0.12, l = 1.56, d = 3
    3)  x0 = 0.01, y0 = 0.01, a = 1.806, b = 0.0,
            g = 1.0, om = 0.0, l = -1.806, d = 5
    4)  x0 = 0.01, y0 = 0.01, a = 10.0, b = -12.0,
            g = 1.0, om = 0.0, l = -2.195, d = 3
    5)  x0 = 0.01, y0 = 0.01, a = -2.5, b = 0.0,
            g = 0.9, om = 0.0, l = 2.5, d = 3
    6)  x0 = 0.01, y0 = 0.01, a = 3.0, b = -16.79,
            g = 1.0, om = 0.0, l = -2.05, d = 9
    7)  x0 = 0.01, y0 = 0.01, a = 5.0, b = 1.5,
            g = 1.0, om = 0.0, l = -2.7, d = 6
    8)  x0 = 0.01, y0 = 0.01, a = 1.0, b = -0.1,
            g = 0.167, om = 0.0, l = -2.08, d = 7
    9)  x0 = 0.01, y0 = 0.01, a = 2.32, b = 0.0,
            g = 0.75, om = 0.0, l = -2.32, d = 5
    10) x0 = 0.01, y0 = 0.01, a = -2.0, b = 0.0,
            g = -0.5, om = 0.0, l = 2.6, d = 5
    11) x0 = 0.01, y0 = 0.01, a = 2.0, b = 0.2,
            g = 0.1, om = 0.0, l = -2.34 , d = 5
    12) x0 = 0.01, y0 = 0.01, a = 2.0, b = 0.0,
            g = 1.0, om = 0.1, l = -1.86, d = 4
    13) x0 = 0.01, y0 = 0.01, a = -1.0, b = 0.1,
            g = -0.82, om = 0.0, l = 1.56, d = 3
    14) x0 = 0.01, y0 = 0.01, a = -1.0, b = 0.03,
            g = -0.8, om = 0.0, l = 1.455, d = 3
    15) x0 = 0.01, y0 = 0.01, a = -2.5, b = -0.1,
            g = 0.9, om = -0.15, l = 2.39, d = 16
    """
    zzbar = x * x + y * y
    p = a * zzbar + l
    zreal, zimag = x, y

    for i in range(1, d - 1):
        za, zb = zreal * x - zimag * y, zimag * x + zreal * y
        zreal, zimag = za, zb

    zn = x * zreal - y * zimag
    p += b * zn

    x_out = p * x + g * zreal - om * y
    y_out = p * y - g * zimag + om * x
    return x_out, y_out


# ==============================================================================


# Three-dimensional attractors
# ==============================================================================


def chua(t, state, a=15.6, b=28, mu0=-1.143, mu1=-0.714):
    """
    Chua attractor.
    Chua circuit. This is a simple electronic circuit that
    exhibits classic chaotic behavior. This means roughly
    that it is a "nonperiodic oscillator".

    Parameters
    ==========
    a, b, mu0, mu1 - are chua system parameters.
    Default values are:
            - x0 = 0, y0 = 0, z0=0, a = 15.6, b = 28, mu0 = -1.143, mu1 = -0.714
    """
    x, y, z = state

    ht = mu1 * x + 0.5 * (mu0 - mu1) * (fabs(x + 1) - fabs(x - 1))
    xdot = a * (y - x - ht)
    ydot = x - y + z
    zdot = -b * y

    return D.array([xdot, ydot, zdot])


def duffing(t, state, a=0.1, b=0.1, omega=1.2):
    """
    Duffing attractor.

    Parameters
    ==========
    a, b - are duffing system parameters.
    Default values are:
            - x0 = 0, y0 = 0, z0 = 0, a = 0.1 and b = 0.1 TO verify value b
    """
    x, y, z = state
    xdot = y
    ydot = -a * y - x**3 + b * cos(omega * z)
    zdot = 1
    return D.array([xdot, ydot, zdot])


def duffing_map(t, state, a=2.75, b=0.2):
    """
    Duffing_map attractor.
    It is a discrete-time dynamical system (2nd order).

    Parameters
    ==========
    a, b - are duffing_map system parameters.
    Default values are:
            - x0 = 0, y0 = 0, z0 = 0, a = 2.75 and b = 0.2
    """
    x, y, z = state
    xdot = y
    ydot = a * y - y**3 - b * x
    zdot = 1
    return D.array([xdot, ydot, zdot])


def lorenz(t, state, sigma=10.0, beta=8 / 3, rho=28.0):
    """
    Lorenz attractor.
    Lorenz attractor is ordinary differential equation (ODE) of 3rd
    order system. In 1963, E. Lorenz developed a simplified mathematical
    model for atmospheric convection.

    Parameters
    ==========
    sigma, beta, rho - are lorenz system parameters.
    Default values are:
            - x0 = 0, y0 = 1, z0 = 1.05, sigma = 10, beta = 8/3, rho = 28
    """
    x, y, z = state
    xdot = rho * y - sigma * x
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return D.array([xdot, ydot, zdot])


def lotka_volterra(t, state):
    """
    Lotka_volterra attractor.
    Lotka_volterra system does not have any system parameters.
    The Lotka–Volterra equations, also known as the predator–prey
    equations, are a pair of first-order nonlinear differential
    equations, frequently used to describe the dynamics of biological
    systems in which two species interact, one as a predator and
    the other as prey.

    Chaotic Lotka-Volterra model require a careful tuning of parameters
    and are even less likely to exhibit chaos as the number of species
    increases. Possible initial values:
            - x0 = 0.6, y0 = 0.2, z0 = 0.01
    """
    x, y, z = state
    xdot = x * (1 - x - 9 * y)
    ydot = -y * (1 - 6 * x - y + 9 * z)
    zdot = z * (1 - 3 * x - z)
    return D.array([xdot, ydot, zdot])


def nose_hover(t, state):
    """
    Nose_hover attractor.
    Nose–Hoover system does not have any system parameters.
    The Nose–Hoover thermostat is a deterministic algorithm for
    constant-temperature molecular dynamics simulations. It was
    originally developed by Nose and was improved further by Hoover.

    Nose–Hoover oscillator is ordinary differential equation (ODE)
    of 3rd order system. Nose–Hoover system has only five terms and
    two quadratic nonlinearities. Possible initial values:
            - x0 = 0, y0 = 0, z0 = 0
    """
    x, y, z = state
    r = np.random.randn(3)
    xdot = y
    ydot = y * z - x
    zdot = 1 - y * y
    return D.array([xdot, ydot, zdot])


def rikitake(t, state, a=2, b=3, c=5, d=0.75):
    """
    Rikitake attractor.
    Rikitake system is ordinary differential equation (ODE) of
    3rd order system, that attempts to explain the reversal of
    the Earth’s magnetic field.

    Parameters
    ==========
    a, and mu - are rikitake system parameters.
    Default values are:
            - x0 = 0, y0 = 0, z0 = 0, a = 5, mu = 2

    Another useful combinations is:
            - x0 = 0, y0 = 0, z0 = 0, a = 1, mu = 1
    """
    x, y, z = state
    xdot = -a * x + y * (z + c)
    ydot = -b * y + x * (z - c)
    zdot = d * z - x * y
    return D.array([xdot, ydot, zdot])


def rossler(t, state, a=0.36, b=0.4, c=4.5):
    """
    Rossler attractor.

    Parameters
    ==========
    a, b and c - are rossler system parameters.
    Default values are:
            - x0 = 0, y0 = 0, z0 = 0, a = 0.2, b = 0.2 and c = 5.7.

    Other useful combinations are:
    1) x0 = 0, y0 = 0, z0 = 0, a = 0.1, b = 0.1 and c = 14 (another useful parameters)
    2) x0 = 0, y0 = 0, z0 = 0, a = 0.5, b = 1.0 and c = 3 (J. C. Sprott)

    Notes
    =====
    - Varying a:
    b = 0.2 and c = 5.7 are fixed. Change a:

    a <= 0      : Converges to the centrally located fixed point
    a = 0.1     : Unit cycle of period 1
    a = 0.2     : Standard parameter value selected by Rössler, chaotic
    a = 0.3     : Chaotic attractor, significantly more Möbius strip-like
                              (folding over itself).
    a = 0.35    : Similar to .3, but increasingly chaotic
    a = 0.38    : Similar to .35, but increasingly chaotic

    - Varying b:
    a = 0.2 and c = 5.7 are fixed. Change b:

    If b approaches 0 the attractor approaches infinity, but if b would
    be more than a and c, system becomes not a chaotic.

    - Varying c:
    a = b = 0.1 are fixed. Change c:

    c = 4       : period-1 orbit,
    c = 6       : period-2 orbit,
    c = 8.5     : period-4 orbit,
    c = 8.7     : period-8 orbit,
    c = 9       : sparse chaotic attractor,
    c = 12      : period-3 orbit,
    c = 12.6    : period-6 orbit,
    c = 13      : sparse chaotic attractor,
    c = 18      : filled-in chaotic attractor.
    """
    x, y, z = state
    xdot = -(y + z)
    ydot = x + a * y
    zdot = b + z * (x - c)
    return D.array([xdot, ydot, zdot])


def wang(t, state):
    """
    Wang attractor.
    Wang system (improved Lorenz model) as classic chaotic attractor.
    Possible initial condition:
            - x0 = 0, y0 = 0, z0 = 0,
    """
    x, y, z = state
    xdot = x - y * z
    ydot = x - y + x * z
    zdot = -3 * z + x * y
    return D.array([xdot, ydot, zdot])


def becker(
    t, state, beta1=1.0, beta2=0.84, rho=0.048, kappaprime=0.8525, nu=4.2164 * 10**-5
):

    kappacr1 = beta1 - 1.0
    kappacr2 = (
        kappacr1
        + rho * (2 * beta1 + (beta2 - 1) * (2 + rho))
        + np.sqrt(
            4 * rho * rho * (kappacr1 + beta2)
            + (kappacr1 + rho * rho * (beta2 - 1))
            * (kappacr1 + rho * rho * (beta2 - 1))
        )
    ) / (2 + 2 * rho)
    kappa = kappaprime * kappacr2

    x, y, z = state
    # ydot = kappa*(1 - np.exp(x))
    ydot = kappa * (1 - np.exp(x))
    zdot = -rho * (beta2 * x + z) * np.exp(x)
    xdot = (((beta1 - 1) * x + y - z) * np.exp(x) + ydot - zdot) / (1 + nu * np.exp(x))
    return D.array([xdot, ydot, zdot])


# ==============================================================================


# Calculator of attractor properties
# ==============================================================================


def check_min_max(data):
    """
    Calculate minimum and maximum for data coordinates.

    Parameters
    ----------
    - data [numpy.ndarray]: data matrix
    """
    min_coord = np.min(data, axis=0)
    max_coord = np.max(data, axis=0)
    return min_coord, max_coord


def check_moments(data, axis=0) -> dict:
    """
    Calculate stochastic parameters: mean, variance, skewness, kurtosis etc.

    Parameters
    ----------
    - data [numpy.ndarray]: data matrix
    """
    dict_moments = {
        "mean": np.mean(data, axis=axis),
        "variance": np.var(data, axis=axis),
        "skewness": skew(data, axis=axis),
        "kurtosis": kurtosis(data, axis=axis),
        "median": np.median(data, axis=axis),
    }
    return dict_moments


def check_probability(data, kde_points=1000):
    """
    Check probability for each chaotic coordinates.

    Parameters
    ----------
    - data [numpy.ndarray]: data matrix
    """
    p_axi = np.zeros([3, kde_points])
    d_kde = np.zeros([3, kde_points])
    for ii in range(3):
        p_axi[ii] = np.linspace(data[ii, :].min(), data[ii, :].max(), kde_points)
        d_kde[ii] = gaussian_kde(data[ii, :]).evaluate(p_axi[ii, :])
        d_kde[ii] /= d_kde[ii].max()
    return d_kde


def calculate_spectrum(data, fft_points=4096):
    """
    Calculate FFT (in dB) for input 3D coordinates.
    You can set number of FFT points into the object instance.

    Parameters
    ----------
    - data [numpy.ndarray]: data matrix
    """

    spectrum = fft(data, fft_points, axis=0)
    spectrum = np.abs(fftshift(spectrum, axes=0))
    # spectrum = np.abs(spectrum)
    spectrum /= np.max(spectrum)
    spec_log = 20 * np.log10(spectrum + np.finfo(np.float32).eps)
    return spec_log


def calculate_correlation(self):
    """
    Calculate auto correlation function for chaotic coordinates.

    Parameters
    ----------
    - data [numpy.ndarray]: data matrix
    """
    nn, mm = 3, len(data)
    auto_corr = np.zeros([mm, nn])
    for ii in range(nn):
        auto_corr[:, ii] = np.correlate(data[:, ii], data[:, ii], "same")
    return auto_corr


# ==============================================================================
