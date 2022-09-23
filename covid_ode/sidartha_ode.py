import torch
import torch.nn as nn

from torchdiffeq.torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SIDARTHEOde(nn.Module):
    def __init__(self, len_data,
                 alpha=0.57, beta=0.11, delta=0.11, gamma=0.456, epsilon=0.171, theta=0.371, zeta=0.125, eta=0.125,
                 mu=0.017, nu=0.027, tau=0.01, lambda_=0.034, rho=0.034, kappa=0.017, xi=0.017, sigma=0.017):
        """
        α=0.570, β=δ=0.011, γ=0.456, ε=0.171, θ=0.371, ζ=η=0.125, μ=0.017, ν=0.027, τ=0.01,
        λ = ρ = 0.034 and κ = ξ = σ = 0.017

        """
        super().__init__()
        torch.manual_seed(32)
        torch.set_default_dtype(torch.float64)
        self.len_data = len_data
        # observed: infected, death, recover
        # unobserved are set as torch.Parameters
        self.y0 = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))  # SDART
        self.pos_dict = dict(zip(['S', 'D', 'A', 'R', 'T', 'I', 'E', 'H'], range(8)))
        # S susceptible
        # I infected
        # D diagnosed
        # A ailing
        # R recognized
        # T threatened
        # H healed
        # E extinct

        # transmission rate
        self.alpha = nn.Parameter(torch.tensor([alpha])).to(device)
        self.beta = nn.Parameter(torch.tensor([beta])).to(device)
        self.gamma = nn.Parameter(torch.tensor([gamma])).to(device)
        self.delta = nn.Parameter(torch.tensor([delta])).to(device)
        # detection rate
        self.epsilon = nn.Parameter(torch.tensor([epsilon])).to(device)
        self.theta = nn.Parameter(torch.tensor([theta])).to(device)
        # develop symptom rate
        self.zeta = nn.Parameter(torch.tensor([zeta])).to(device)  # ζ
        self.eta = nn.Parameter(torch.tensor([eta])).to(device)  # η
        # develop life-threatening symptoms rate
        self.mu = nn.Parameter(torch.tensor([mu])).to(device)  # for undetected
        self.nu = nn.Parameter(torch.tensor([nu])).to(device)  # ν for detected
        # mortality rate
        self.tau = nn.Parameter(torch.tensor([tau])).to(device)
        # recovery rate
        self.lambda_ = nn.Parameter(torch.tensor([lambda_])).to(device)
        self.kappa = nn.Parameter(torch.tensor([kappa])).to(device)
        self.xi = nn.Parameter(torch.tensor([xi])).to(device)  # ξ
        self.rho = nn.Parameter(torch.tensor([rho])).to(device)
        self.sigma = nn.Parameter(torch.tensor([sigma])).to(device)
        # self.double()

    def f(self, t, y):
        return torch.cat([y[self.pos_dict['S']] * (
                self.alpha * y[self.pos_dict['I']] + self.beta * y[self.pos_dict['D']] + self.gamma * y[
            self.pos_dict['A']] + self.delta * y[self.pos_dict['R']]),  # S
                          self.epsilon * y[self.pos_dict['I']] - (self.eta + self.rho) * y[self.pos_dict['D']],  # D
                          self.zeta * y[self.pos_dict['I']] - (self.theta + self.mu + self.kappa) * y[
                              self.pos_dict['A']],  # A
                          self.eta * y[self.pos_dict['D']] + self.theta * y[self.pos_dict['A']] - (self.nu + self.xi) *
                          y[self.pos_dict['R']],  # R
                          self.mu * y[self.pos_dict['A']] + self.nu * y[self.pos_dict['R']] - (self.sigma + self.tau) *
                          y[self.pos_dict['T']],  # T
                          y[self.pos_dict['S']] * (self.alpha * y[self.pos_dict['I']] + self.beta * y[
                              self.pos_dict['D']] + self.gamma * y[
                                                       self.pos_dict['A']] + self.delta * y[self.pos_dict['R']]) - (
                                  self.epsilon + self.zeta + self.lambda_) * y[self.pos_dict['I']],  # I
                          self.lambda_ * y[self.pos_dict['I']] + self.rho * y[self.pos_dict['D']] + self.kappa * y[
                              self.pos_dict['A']] + self.xi * y[self.pos_dict['R']] + self.sigma * y[
                              self.pos_dict['T']],  # dH
                          self.tau * y[self.pos_dict['T']]  # E
                          ])

    def forward(self, I0, E0, H0):
        time_range = torch.linspace(0, self.len_data, self.len_data + 1)
        return odeint(self.f, t=time_range,
                      y0=torch.cat((self.y0, torch.tensor([I0, E0, H0]))),
                      method='rk4')
