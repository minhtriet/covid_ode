import unittest
import numpy as np
from scipy.integrate import odeint
from torchdiffeq import torchdiffeq
import torch

class TestODE(unittest.TestCase):

    def all_odes(y, t):
        alpha = 0.57
        beta = 0.11
        delta = 0.11
        gamma = 0.456
        epsilon = 0.171
        theta = 0.371
        zeta = 0.125
        eta = 0.125
        mu = 0.017
        nu = 0.027
        tau = 0.01
        lambda_ = 0.034
        rho = 0.034
        kappa = 0.017
        xi = 0.017
        sigma = 0.017
        S, I, D, A, R, T, H, E = y
        tmp= [S * (alpha * I + beta * D + gamma * A + delta * R),  # S
                S * (alpha * I + beta * D + gamma * A + delta * R) - ( epsilon + zeta + lambda_) * I,  # I
                epsilon * I - (eta + rho) * D,  # D
                zeta * I - (theta + mu + kappa) * A,  # A
                eta * D + theta * A - (nu + xi) * R,  # R
                mu * A + nu * R - (sigma + tau) * T,  # dTdt
                lambda_ * I + rho * D + kappa * A + xi * R + sigma * T,  # dH
                tau * T]  # E
        return tmp

    def all_odes_torch(t, y):
        alpha = 0.57
        beta = 0.11
        delta = 0.11
        gamma = 0.456
        epsilon = 0.171
        theta = 0.371
        zeta = 0.125
        eta = 0.125
        mu = 0.017
        nu = 0.027
        tau = 0.01
        lambda_ = 0.034
        rho = 0.034
        kappa = 0.017
        xi = 0.017
        sigma = 0.017
        S, I, D, A, R, T, H, E = y
        tmp= [S * (alpha * I + beta * D + gamma * A + delta * R),  # S
                S * (alpha * I + beta * D + gamma * A + delta * R) - ( epsilon + zeta + lambda_) * I,  # I
                epsilon * I - (eta + rho) * D,  # D
                zeta * I - (theta + mu + kappa) * A,  # A
                eta * D + theta * A - (nu + xi) * R,  # R
                mu * A + nu * R - (sigma + tau) * T,  # dTdt
                lambda_ * I + rho * D + kappa * A + xi * R + sigma * T,  # dH
                tau * T]  # E
        return torch.tensor(tmp, requires_grad=True)


    def sir_ode(y, t):
        N = y.sum()
        beta = 0.11
        gamma = .456
        return [y[0] * y[1] * beta / N,  # S
                          beta * y[1] * y[0] / N - gamma * y[1],  # I
                          gamma * y[1]  # R
                          ]  # , dtype=torch.float64, requires_grad=True)


    def test_scipy_int_and_torchdiffeq_int(self):
        y0  = [5.0000e-01, 2.4431e-03, 5.0000e-01, 5.0000e-01, 5.0000e-01, 5.0000e-01, 1.3178e-03, 2.0718e-04]
        # a, b = odeint(TestODE.all_odes, y0=y0, t=np.linspace(0, 5, 6), full_output=True)

        # a_torch = torchdiffeq.odeint(TestODE.all_odes_torch, y0=torch.Tensor(y0), t=torch.linspace(0, 5, 6), method='rk4')
        # assert a_torch.requires_grad

        a =  odeint(TestODE.sir_ode, y0=[0.5,2,3], t=[0,1,2])
        # tensor([[0.5000, 2.0000, 3.0000],
        #         [0.5163, 1.2808, 3.7355],
        #         [0.5270, 0.8204, 4.2066]], grad_fn=<CopySlices>)
        pass