import numpy as np


def ackley_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x ** 2)
    t2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(t1 / ndim)) - np.exp(t2 / ndim) + 20 + np.e


def bent_cigar_func(x):
    x = np.array(x).ravel()
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def discus_func(x):
    x = np.array(x).ravel()
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)


def elliptic_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    idx = np.arange(0, ndim)
    return np.sum(10 ** (6.0 * idx / (ndim - 1)) * x ** 2)


def griewank_func(x):
    x = np.array(x).ravel()
    idx = np.arange(1, len(x) + 1)
    t1 = np.sum(x ** 2) / 4000
    t2 = np.prod(np.cos(x / np.sqrt(idx)))
    return t1 - t2 + 1


def grie_rosen_cec_func(x):
    """This is based on the CEC version which unrolls the griewank and rosenbrock functions for better performance"""
    z = np.array(x).ravel()
    z += 1.0  # This centers the optimal solution of rosenbrock to 0

    tmp1 = (z[:-1] * z[:-1] - z[1:]) ** 2
    tmp2 = (z[:-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f = np.sum(temp ** 2 / 4000.0 - np.cos(temp) + 1.0)
    # Last calculation
    tmp1 = (z[-1] * z[-1] - z[0]) ** 2
    tmp2 = (z[-1] - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    f += (temp ** 2) / 4000.0 - np.cos(temp) + 1.0

    return f


def happy_cat_func(x):
    z = np.array(x).ravel()
    ndim = len(z)
    t1 = np.sum(z)
    t2 = np.sum(z ** 2)
    return np.abs(t2 - ndim) ** 0.25 + (0.5 * t2 + t1) / ndim + 0.5


def happy_cat_shifted_func(x):
    return happy_cat_func(x - 1.0)


def hgbat_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    t1 = np.sum(x)
    t2 = np.sum(x ** 2)
    return np.abs(t2 ** 2 - t1 ** 2) ** 0.5 + (0.5 * t2 + t1) / ndim + 0.5


def hgbat_shifted_func(x):
    return hgbat_func(x - 1.0)


def katsuura_func(x):
    x = np.array(x).ravel()
    ndim = len(x)
    result = 1.0
    for idx in range(0, ndim):
        temp = np.sum([np.abs(2 ** j * x[idx] - np.round(2 ** j * x[idx])) / 2 ** j for j in range(1, 33)])
        result *= (1 + (idx + 1) * temp) ** (10.0 / ndim ** 1.2)
    return (result - 1) * 10 / ndim ** 2


def levy_func(x):
    x = np.array(x).ravel()
    w = 1. + x / 4
    t1 = np.sin(np.pi * w[0]) ** 2 + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    t2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    return t1 + t2


def lunacek_bi_rastrigin_func(x, miu0=2.5, d=1):
    x = np.array(x).ravel()
    ndim = len(x)
    s = 1.0 - 1.0 / (2 * np.sqrt(ndim + 20) - 8.2)
    miu1 = -np.sqrt((miu0 ** 2 - d) / s)
    delta_x_miu0 = x - miu0
    term1 = np.sum(delta_x_miu0 ** 2)
    term2 = np.sum((x - miu1) ** 2) * s + d * ndim
    result = min(term1, term2) + 10 * (ndim - np.sum(np.cos(2 * np.pi * delta_x_miu0)))
    return result


def lunacek_bi_rastrigin_shifted_func(x, miu0=2.5, d=1):
    x = np.array(x).ravel()
    return lunacek_bi_rastrigin_func(x + miu0, miu0, d)


def non_continuous_rastrigin_func(x):
    x = np.array(x).ravel()
    y = rounder(x, np.abs(x))
    shifted_y = np.roll(y, -1)
    results = rastrigin_func(np.column_stack((y, shifted_y)))
    return np.sum(results)

def step_rastrigin_func(x):
    ndim = len(x)
    f = 0.0
    for i in range(ndim):
        if (np.fabs(y[i] - f_shift[i]) > 0.5):
            y[i] = f_shift[i] + np.floor(2 * (y[i] - f_shift[i]) + 0.5) / 2

    z = sr_func(x, ndim, f_shift, f_rotate, 5.12 / 100.0, s_flag, r_flag)

    for i in range(ndim):
        f += (z[i] * z[i] - 10.0 * np.cos(2.0 * PI * z[i]) + 10.0)
    return f

def rastrigin_func(x):
    x = np.array(x).ravel()
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def rosenbrock_func(x):
    x = np.array(x).ravel()
    term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
    term2 = (x[:-1] - 1) ** 2
    return np.sum(term1 + term2)


def rosenbrock_shifted_func(x):
    """
    This version shifts the optimum to the origin as CEC2021 version.
    """
    x = np.array(x).ravel()
    return rosenbrock_func(x + 1.0)


def rotated_expanded_schaffer_func(x):
    x = np.asarray(x).ravel()
    x_pairs = np.column_stack((x, np.roll(x, -1)))
    sum_sq = x_pairs[:, 0] ** 2 + x_pairs[:, 1] ** 2
    # Calculate the Schaffer function for all pairs simultaneously
    schaffer_values = (0.5 + (np.sin(np.sqrt(sum_sq)) ** 2 - 0.5) /
                       (1 + 0.001 * sum_sq) ** 2)
    return np.sum(schaffer_values)


def schaffer_func(x):
    x = np.array(x).ravel()
    return 0.5 + (np.sin(np.sqrt(np.sum(x ** 2))) ** 2 - 0.5) / (1 + 0.001 * np.sum(x ** 2)) ** 2


def expanded_schaffer_f6_func(x):
    """
    This is a direct conversion of the CEC2021 C-Code for the Expanded Schaffer F6 Function
    """
    z = np.array(x).ravel()

    temp1 = np.sin(np.sqrt(z[:-1] ** 2 + z[1:] ** 2))
    temp1 = temp1 ** 2
    temp2 = 1.0 + 0.001 * (z[:-1] ** 2 + z[1:] ** 2)
    f = np.sum(0.5 + (temp1 - 0.5) / (temp2 ** 2))

    temp1_last = np.sin(np.sqrt(z[-1] ** 2 + z[0] ** 2))
    temp1_last = temp1_last ** 2
    temp2_last = 1.0 + 0.001 * (z[-1] ** 2 + z[0] ** 2)
    f += 0.5 + (temp1_last - 0.5) / (temp2_last ** 2)

    return f


def schaffer_f7_func(x):
    x = np.array(x).ravel()
    t = x[:-1] ** 2 + x[1:] ** 2
    result = np.sum(np.sqrt(t) * (np.sin(50. * t ** 0.2) + 1))
    ndim = len(x)
    return (result / (ndim - 1)) ** 2


def modified_schwefel_func(x):
    """
        This is a direct conversion of the CEC2021 C-Code for the Modified Schwefel F11 Function
    """
    z = np.array(x).ravel() + 4.209687462275036e+002
    nx = len(z)

    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2
    fx = np.zeros(nx)
    fx[mask1] -= (500.0 + np.fmod(np.abs(z[mask1]), 500)) * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z[mask1]), 500))) - (
            (z[mask1] - 500.0) / 100.) ** 2 / nx
    fx[mask2] -= (-500.0 + np.fmod(np.abs(z[mask2]), 500)) * np.sin(np.sqrt(500.0 - np.fmod(np.abs(z[mask2]), 500))) - (
            (z[mask2] + 500.0) / 100.) ** 2 / nx
    fx[mask3] -= z[mask3] * np.sin(np.sqrt(np.abs(z[mask3])))

    return np.sum(fx) + 4.189828872724338e+002 * nx


def weierstrass_func(x, a=0.5, b=3., k_max=20):
    """ Conversion from CEC2005 C-Code """
    x = np.array(x).ravel()
    k = np.arange(0, k_max + 1)
    cos_term = np.cos(2 * np.pi * b ** k * (x[:, np.newaxis] + 0.5))
    term_sum = np.sum(a ** k * cos_term, axis=1)
    return np.sum(term_sum)


def weierstrass_norm_func(x, a=0.5, b=3., k_max=20):
    """
    This function matches CEC2005 description of F11 except for addition of the bias and follows the C implementation
    """
    return weierstrass_func(x, a, b, k_max) - weierstrass_func(np.zeros(len(x)), a, b, k_max)


def zakharov_func(x):
    z = np.array(x).ravel()
    idx = np.arange(1, len(z) + 1)
    temp = 0.5 * np.sum(idx * z)
    return np.sum(z ** 2) + temp ** 2 + temp ** 4
