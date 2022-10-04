import torch
import torch.nn as nn

from torchdiffeq.torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SIDARTHEOde(nn.Module):
    def __init__(self, len_data,
                 α=0.57, β=0.11, δ=0.11, γ=0.456, ε=0.171, θ=0.371, ζ=0.125, η=0.125,
                 μ=0.017, ν=0.027, τ=0.01, λ=0.034, ρ=0.034, κ=0.017, ξ=0.017, σ=0.017):
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
        self.y0 = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))  # SIART
        self.pos_dict = dict(zip(['S', 'I', 'A', 'R', 'T', 'D', 'E', 'H'], range(8)))
        # S susceptible
        # I infected
        # D diagnosed
        # A ailing
        # R recognized
        # T threatened
        # H healed
        # E extinct

        # transmission rate
        self.α = nn.Parameter(torch.tensor([α])).to(device)
        self.β = nn.Parameter(torch.tensor([β])).to(device)
        self.γ = nn.Parameter(torch.tensor([γ])).to(device)
        self.δ = nn.Parameter(torch.tensor([δ])).to(device)
        # detection rate
        self.ε = nn.Parameter(torch.tensor([ε])).to(device)
        self.θ = nn.Parameter(torch.tensor([θ])).to(device)
        # develop symptom rate
        self.ζ = nn.Parameter(torch.tensor([ζ])).to(device)
        self.η = nn.Parameter(torch.tensor([η])).to(device)
        # develop life-threatening symptoms rate
        self.μ = nn.Parameter(torch.tensor([μ])).to(device)  # for undetected
        self.ν = nn.Parameter(torch.tensor([ν])).to(device)  # for detected
        # mortality rate
        self.τ = nn.Parameter(torch.tensor([τ])).to(device)
        # recovery rate
        self.λ = nn.Parameter(torch.tensor([λ])).to(device)
        self.κ = nn.Parameter(torch.tensor([κ])).to(device)
        self.ξ = nn.Parameter(torch.tensor([ξ])).to(device)
        self.ρ = nn.Parameter(torch.tensor([ρ])).to(device)
        self.σ = nn.Parameter(torch.tensor([σ])).to(device)

    def f(self, t, y):
        return torch.cat([
            -y[self.pos_dict['S']] * (self.α * y[self.pos_dict['I']] + self.β * y[self.pos_dict['D']] + self.γ * y[
                self.pos_dict['A']] + self.δ * y[self.pos_dict['R']]),  # S
            y[self.pos_dict['S']] * (self.α * y[self.pos_dict['I']] + self.β * y[
                self.pos_dict['D']] + self.γ * y[ self.pos_dict['A']] + self.δ * y[self.pos_dict['R']]) -
                (self.ε + self.ζ + self.λ) * y[self.pos_dict['I']],  # I
            self.ζ * y[self.pos_dict['I']] - (self.θ + self.μ + self.κ) * y[self.pos_dict['A']],  # A
            self.η * y[self.pos_dict['D']] + self.θ * y[self.pos_dict['A']] - (self.ν + self.ξ) * y[self.pos_dict['R']],  # R
            self.μ * y[self.pos_dict['A']] + self.ν * y[self.pos_dict['R']] - (self.σ + self.τ) * y[self.pos_dict['T']],  # T
            self.ε * y[self.pos_dict['I']] - (self.η + self.ρ) * y[self.pos_dict['D']],  # D
            self.τ * y[self.pos_dict['T']],  # E
            self.λ * y[self.pos_dict['I']] + self.ρ * y[self.pos_dict['D']] + self.κ * y[
                self.pos_dict['A']] + self.ξ * y[self.pos_dict['R']] + self.σ * y[self.pos_dict['T']],  # H
        ])

    def forward(self, D0, E0, H0):
        time_range = torch.linspace(0, self.len_data, self.len_data + 1)
        return odeint(self.f, t=time_range,
                      y0=torch.cat((self.y0, torch.tensor([D0, E0, H0]))),
                      method='rk4')
