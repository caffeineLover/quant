from __future__ import annotations
import math
from processes.base import StochasticProcess
from rng.RandNumGen import MersenneTwisterEngine
from rng.Distributions import NormalDistribution



##  Arithmetic Brownian Motion:  dS = mu dt + sigma dW
##
##  Exact one-step form:         S_{t+dt} = S_t + mu*dt + sigma*sqrt(dt)*Z
##
##    where Z ~ N(0,1).
##
class ArithmeticBrownianMotion(StochasticProcess):

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
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        z = self.normal.sample()
        return current_value + self.mu * dt + self.sigma * math.sqrt(dt) * z



    def __repr__(self) -> str:
        return f"ArithmeticBrownianMotion(mu={self.mu}, sigma={self.sigma})"