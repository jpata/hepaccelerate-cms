import numpy as np
from scipy.special import kolmogorov
import math

def likelihood(data_i, s_i, b_i, mu):
    b_i[b_i < 0] = 0.0
    s_i[s_i < 0] = 0.0

    sel = ((s_i>0) & (b_i>0))
    ret = data_i[sel] * np.log(mu*s_i[sel] + b_i[sel]) - (mu*s_i[sel] + b_i[sel])
    return np.sum(ret)

def sig_q0_asimov(s_i, b_i):
    mu = 1
    data_i = mu * s_i + b_i
    l0 = likelihood(data_i, s_i, b_i, 0)
    l1 = likelihood(data_i, s_i, b_i, mu)
    q0 = -2 * (l0 - l1)
    Z = np.sqrt(q0)
    return Z

def sig_naive(s_i, b_i):
    s = np.sum(s_i)
    b = np.sum(b_i)
    return s/np.sqrt(s+b)


"""
Tests for similarity between two one-dimensional histograms, ignoring normalization.
Adapted from ROOT https://root.cern.ch/doc/master/classTH1.html#aeadcf087afe6ba203bcde124cfabbee4

Args:
    bins1 (np.array): bin contents of the first histogram
    bins2 (np.array): bin contents of the second histogram
"""
def kolmogorov_smirnov(bins1, bins2, variances1=None, variances2=None):
    assert(bins1.shape == bins2.shape)
    if not (variances1 is None):
        assert(bins1.shape == variances1.shape)
    if not (variances2 is None):
        assert(bins2.shape == variances2.shape)

    sum1 = np.sum(bins1)
    sum2 = np.sum(bins2)

    bins1_norm = bins1 / sum1
    bins2_norm = bins2 / sum2

    bins1_cdf = np.cumsum(bins1_norm)
    bins2_cdf = np.cumsum(bins2_norm)

    esum1 = None
    esum2 = None
    if not (variances1 is None):
        esum1 = sum1*sum1 / np.sum(variances1)
    if not (variances2 is None):
        esum2 = sum2*sum2 / np.sum(variances2)

    dfmax = np.max(np.abs(bins1_cdf - bins2_cdf))

    if esum1 and esum2:
        z = dfmax * math.sqrt(esum1 * esum2 / (esum1 + esum2))
    elif esum1:
        z = dfmax * math.sqrt(esum1)
    elif esum2:
        z = dfmax * math.sqrt(esum2)
    else:
        z = dfmax

    p = kolmogorov(z)
    return p
