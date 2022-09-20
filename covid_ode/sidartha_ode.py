import torch
import torch.nn as nn
import warnings

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

        # observed: infected, death, recover
        self.I, self.E, self.H = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        # unobserved are set as torch.Parameters
        self.S = nn.Parameter(torch.tensor([0.5]).float().to(device))
        self.D = nn.Parameter(torch.tensor([0.5]).float().to(device)) 
        self.A = nn.Parameter(torch.tensor([0.5]).float().to(device))
        self.R = nn.Parameter(torch.tensor([0.5]).float().to(device))
        self.T = nn.Parameter(torch.tensor([0.5]).float().to(device))
        self.y0 = torch.tensor([self.S, self.I, self.D, self.A, self.R, self.T, self.H, self.E])
        # S susceptible
        # I infected
        # D diagnosed
        # A ailing
        # R recognized
        # T threatened
        # H healed
        # E extinct
        
        # transmission rate
        self.alpha = nn.Parameter(torch.FloatTensor([alpha])).to(device)
        self.beta = nn.Parameter(torch.FloatTensor([beta])).to(device)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma])).to(device)
        self.delta = nn.Parameter(torch.FloatTensor([delta])).to(device)
        # detection rate
        self.epsilon = nn.Parameter(torch.FloatTensor([epsilon])).to(device)
        self.theta = nn.Parameter(torch.FloatTensor([theta])).to(device)
        # develop symptom rate
        self.zeta = nn.Parameter(torch.FloatTensor([zeta])).to(device)   # ζ
        self.eta = nn.Parameter(torch.FloatTensor([eta])).to(device)   # η
        # develop life-threatening symptoms rate
        self.mu = nn.Parameter(torch.FloatTensor([mu])).to(device)  # for undetected
        self.nu = nn.Parameter(torch.FloatTensor([nu])).to(device)  # ν for detected
        # mortality rate
        self.tau = nn.Parameter(torch.FloatTensor([tau])).to(device)
        # recovery rate
        self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_])).to(device)
        self.kappa = nn.Parameter(torch.FloatTensor([kappa])).to(device)
        self.xi = nn.Parameter(torch.FloatTensor([xi])).to(device)  # ξ
        self.rho = nn.Parameter(torch.FloatTensor([rho])).to(device)
        self.sigma = nn.Parameter(torch.FloatTensor([sigma])).to(device)

    # The following d*d* function will be discretized and apply RK45 method
    # return something pointwise here

    def f(self, t, y):
        S, I, D, A, R, T, H, E = y
        return torch.tensor([S*(self.alpha*I + self.beta*D + self.gamma*A + self.delta*R),    # S
                             S*(self.alpha*I + self.beta*D + self.gamma*A + self.delta*R) - (self.epsilon + self.zeta + self.lambda_)*I,  # I
                             self.epsilon*I - (self.eta + self.rho)*D,   # D
                             self.zeta*I - (self.theta + self.mu + self.kappa)*A,  # A
                             self.eta*D + self.theta*A - (self.nu + self.xi)*R,   # R
                             self.mu*A + self.nu*R - (self.sigma + self.tau)*T,     # dTdt
                             self.lambda_*I + self.rho*D + self.kappa*A + self.xi*R + self.sigma*T,    # dH
                             self.tau*T    # E
                            ])


    # def dSdt(self, t, S):
    #     return -torch.take(S, t)*(self.alpha*torch.take(self.I, t) + self.beta*self.D + self.gamma*self.A + self.delta*self.R)
    #
    # def dIdt(self, t, I):
    #     return self.S*(self.alpha*I + self.beta*self.D + self.gamma*self.A + self.delta*self.R) - \
    #            (self.epsilon + self.zeta + self.lambda_)*I
    #
    # def dDdt(self, t, D):
    #     return self.epsilon*self.I - (self.eta + self.rho)*D
    #
    # def dAdt(self, t, A):
    #     return self.zeta*self.I - (self.theta + self.mu + self.kappa)*A
    #
    # def dRdt(self, t, R):
    #     return self.eta*self.D + self.theta*self.A - (self.nu + self.xi)*R
    #
    # def dTdt(self, t, T):
    #     return self.mu*self.A + self.nu*self.R - (self.sigma + self.tau)*T
    #
    # def dHdt(self, t, H):
    #     return self.lambda_*self.I + self.rho*self.D + self.kappa*self.A + self.xi*self.R * self.sigma*self.T
    #
    # def dEdt(self, t, E):
    #     return self.tau*self.T

    def forward(self, input_steps):
        time_range = torch.linspace(0, input_steps, 1000)
        return odeint(self.f, t=time_range, y0=self.y0, method='rk4')

