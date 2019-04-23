import numpy as np

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