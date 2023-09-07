import numpy as np

from hdoptbench.functions.operators import bent_cigar_func, ackley_func, discus_func, elliptic_func, griewank_func, \
    grie_rosen_cec_func, happy_cat_func, happy_cat_shifted_func, hgbat_func, hgbat_shifted_func, katsuura_func, \
    levy_func, lunacek_bi_rastrigin_func, lunacek_bi_rastrigin_shifted_func

EPSILON = 1e-14


def test_ackley_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(ackley_func(x) - 0) <= EPSILON

def test_bent_cigar_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(bent_cigar_func(x) - 0) <= EPSILON

def test_discus_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(discus_func(x) - 0) <= EPSILON


def test_elliptic_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(elliptic_func(x) - 0) <= EPSILON

def test_griewank_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(griewank_func(x) - 0) <= EPSILON

def test_griewank_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(griewank_func(x) - 0) <= EPSILON

def test_grie_rosen_cec_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(grie_rosen_cec_func(x) - 0) <= EPSILON

def test_happy_cat_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim) - 1
    assert abs(happy_cat_func(x) - 0) <= EPSILON

def test_happy_cat_shifted_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(happy_cat_shifted_func(x) - 0) <= EPSILON

def test_hgbat_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim) - 1
    assert abs(hgbat_func(x) - 0) <= EPSILON

def test_hgbat_shifted_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(hgbat_shifted_func(x) - 0) <= EPSILON

def test_katsuura_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(katsuura_func(x) - 0) <= EPSILON

def test_levy_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(levy_func(x) - 0) <= EPSILON

def test_lunacek_bi_rastrigin_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim) + 2.5
    assert abs(lunacek_bi_rastrigin_func(x) - 0) <= EPSILON

def test_lunacek_bi_rastrigin_shifted_func_optimum_result():
    ndim = 2
    x = np.zeros(ndim)
    assert abs(lunacek_bi_rastrigin_shifted_func(x) - 0) <= EPSILON

