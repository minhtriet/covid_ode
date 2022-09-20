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
        torch.set_default_dtype(torch.float64)
        # observed: infected, death, recover
        self.len_data = len_data
        # just for the init, every forward round it get changed
        self.I, self.E, self.H = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        # unobserved are set as torch.Parameters
        self.S = nn.Parameter(torch.tensor([0.5]).to(device))
        self.D = nn.Parameter(torch.tensor([0.5]).to(device))
        self.A = nn.Parameter(torch.tensor([0.5]).to(device))
        self.R = nn.Parameter(torch.tensor([0.5]).to(device))
        self.T = nn.Parameter(torch.tensor([0.5]).to(device))
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
        self.alpha = nn.Parameter(torch.tensor([alpha])).to(device)
        self.beta = nn.Parameter(torch.tensor([beta])).to(device)
        self.gamma = nn.Parameter(torch.tensor([gamma])).to(device)
        self.delta = nn.Parameter(torch.tensor([delta])).to(device)
        # detection rate
        self.epsilon = nn.Parameter(torch.tensor([epsilon])).to(device)
        self.theta = nn.Parameter(torch.tensor([theta])).to(device)
        # develop symptom rate
        self.zeta = nn.Parameter(torch.tensor([zeta])).to(device)   # ζ
        self.eta = nn.Parameter(torch.tensor([eta])).to(device)   # η
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
        #self.double()

    def f(self, t, y):
        S, I, D, A, R, T, H, E = y
        return torch.cat([S * (self.alpha * I + self.beta * D + self.gamma * A + self.delta * R),  # S
                             S*(self.alpha*I + self.beta*D + self.gamma*A + self.delta*R) - (self.epsilon + self.zeta + self.lambda_)*I,  # I
                             self.epsilon*I - (self.eta + self.rho)*D,   # D
                             self.zeta*I - (self.theta + self.mu + self.kappa)*A,  # A
                             self.eta*D + self.theta*A - (self.nu + self.xi)*R,   # R
                             self.mu*A + self.nu*R - (self.sigma + self.tau)*T,     # dTdt
                             self.lambda_*I + self.rho*D + self.kappa*A + self.xi*R + self.sigma*T,    # dH
                             self.tau*T    # E
                            ])
        # return torch.tensor([S*(self.alpha*I + self.beta*D + self.gamma*A + self.delta*R),    # S
        #                      S*(self.alpha*I + self.beta*D + self.gamma*A + self.delta*R) - (self.epsilon + self.zeta + self.lambda_)*I,  # I
        #                      self.epsilon*I - (self.eta + self.rho)*D,   # D
        #                      self.zeta*I - (self.theta + self.mu + self.kappa)*A,  # A
        #                      self.eta*D + self.theta*A - (self.nu + self.xi)*R,   # R
        #                      self.mu*A + self.nu*R - (self.sigma + self.tau)*T,     # dTdt
        #                      self.lambda_*I + self.rho*D + self.kappa*A + self.xi*R + self.sigma*T,    # dH
        #                      self.tau*T    # E
        #                     ], dtype=torch.float64, requires_grad=True)

    def forward(self, I0, E0, H0):
        self.y0[1] = I0 if I0 else self.y0[1]
        self.y0[7] = E0 if E0 else self.y0[7]
        self.y0[6] = H0 if H0 else self.y0[6]
        time_range = torch.linspace(0, self.len_data, self.len_data+1)
        return odeint(self.f, t=time_range, y0=self.y0, method='rk4')#.double()

