from __future__ import annotations
import math
from processes.base import StochasticProcess
from rng.RandNumGen import MersenneTwisterEngine
from rng.Distributions import NormalDistribution





##  Heston stochastic volatility model:
##
##        dS = mu * S dt + sqrt(v) * S dW_1
##        dv = kappa * (theta - v) dt + xi * sqrt(v) dW_2
##
##    with corr(dW_1, dW_2) = rho.
##
##    This implementation uses a simple Euler-style discretization with full truncation on
##    the variance process:
##
##        v_eff = max(v, 0)
##
##    Then:
##        S_{t+dt} = S_t * exp((mu - 0.5*v_eff)dt + sqrt(v_eff*dt)*Z1)
##        v_{t+dt} = v_t + kappa(theta - v_eff)dt + xi*sqrt(v_eff*dt)*Z2
##
##  This is a practical starting point for research code, not great discretization quality.
##
class HestonProcess(StochasticProcess):

    def __init__(
        self,
        engine: MersenneTwisterEngine,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ) -> None:
        if kappa < 0.0:
            raise ValueError("kappa must be nonnegative")
        if theta < 0.0:
            raise ValueError("theta must be nonnegative")
        if xi < 0.0:
            raise ValueError("xi must be nonnegative")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must lie in [-1, 1]")

        self.engine = engine
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.normal = NormalDistribution(engine, mu=0.0, sigma=1.0)



    ## Advance one step.
    ##
    ## Parameters:
    ##    * current_value is a tuple (s,v) where:
    ##        - s = asset price
    ##        - v = instantaneous variance
    ##    * dt is the time step.
    ##
    ##  Returns:
    ##    * tuple[float, float] is Next (s, v)
    ##
    def next_value(self, current_value: tuple[float, float], dt: float) -> tuple[float, float]:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        s, v = current_value

        if s <= 0.0:
            raise ValueError("asset price must be positive")

        v_eff = max(v, 0.0)

        z1 = self.normal.sample()
        z_indep = self.normal.sample()
        z2 = self.rho * z1 + math.sqrt(1.0 - self.rho * self.rho) * z_indep

        s_next = s * math.exp(
            (self.mu - 0.5 * v_eff) * dt + math.sqrt(v_eff * dt) * z1
        )

        v_next = (
            v
            + self.kappa * (self.theta - v_eff) * dt
            + self.xi * math.sqrt(v_eff * dt) * z2
        )

        v_next = max(v_next, 0.0)

        return s_next, v_next



    def generate_path(
        self,
        initial_value: tuple[float, float],
        dt: float,
        steps: int,
    ) -> list[tuple[float, float]]:
        return super().generate_path(initial_value, dt, steps)



    def generate_price_path(
        self,
        s0: float,
        v0: float,
        dt: float,
        steps: int,
    ) -> list[float]:
        """
        Convenience method: generate only the asset-price path.
        """
        path = self.generate_path((s0, v0), dt, steps)
        return [s for s, _ in path]



    # Convenience method: generate only the variance path.
    def generate_variance_path(
        self,
        v0: float,
        s0: float,
        dt: float,
        steps: int,
    ) -> list[float]:
        path = self.generate_path((s0, v0), dt, steps)
        return [v for _, v in path]



    def __repr__(self) -> str:
        return (
            "HestonProcess("
            f"mu={self.mu}, kappa={self.kappa}, theta={self.theta}, "
            f"xi={self.xi}, rho={self.rho})"
        )