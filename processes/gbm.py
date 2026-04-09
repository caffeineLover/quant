from __future__ import annotations
import math
from base import StochasticProcess
from rng.RandNumGen import MersenneTwisterEngine
from rng.Distributions import NormalDistribution



##  Geometric Brownian Motion:   dS = mu * S dt + sigma * S dW
##
##  Exact discretization:
##
##        S_{t+dt} = S_t * exp( (mu - 0.5*sigma^2) dt + sigma*sqrt(dt)*Z)
##
##        where Z ~ N(0,1).
##
class GeometricBrownianMotion(StochasticProcess):
    def __init__(
        self,
        engine: MersenneTwisterEngine,
        mu: float,
        sigma: float,
    ) -> None:
        if sigma < 0.0:
            raise ValueError("sigma must be nonnegative")

        self.engine = engine
        self.mu = mu
        self.sigma = sigma
        self.normal = NormalDistribution(engine, mu=0.0, sigma=1.0)

    def next_value(self, current_value: float, dt: float) -> float:
        if current_value <= 0.0:
            raise ValueError("current_value must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        z = self.normal.sample()
        drift = (self.mu - 0.5 * self.sigma * self.sigma) * dt
        diffusion = self.sigma * math.sqrt(dt) * z

        return current_value * math.exp(drift + diffusion)

    def __repr__(self) -> str:
        return f"GeometricBrownianMotion(mu={self.mu}, sigma={self.sigma})"