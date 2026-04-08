from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import Optional
from RandNumGen import MersenneTwisterEngine



###  Abstract base class for probability distributions.
###
class Distribution(ABC):

    def __init__(self, engine: MersenneTwisterEngine) -> None:
        self.engine = engine

    # Return one draw from the distribution.
    @abstractmethod
    def sample(self):
        raise NotImplementedError

    # Return a list of n draws from the distribution.
    def sample_n(self, n: int) -> list:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return [self.sample() for _ in range(n)]



## Uniform(a, b)
##
class UniformDistribution(Distribution):

    def __init__(
        self,
        engine: MersenneTwisterEngine,
        a: float = 0.0,
        b: float = 1.0,
    ) -> None:
        super().__init__(engine)

        if b <= a:
            raise ValueError("b must be greater than a")

        self.a = a
        self.b = b

    def sample(self) -> float:
        return self.engine.uniform(self.a, self.b)

    def __repr__(self) -> str:
        return f"UniformDistribution(a={self.a}, b={self.b})"





##  Normal(mu, sigma^2) using the Marsaglia polar method.
##
##     * Similar to Box-Muller but avoids expensive trig functions.
##
class NormalDistribution(Distribution):

    def __init__(
        self,
        engine: MersenneTwisterEngine,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(engine)

        if sigma <= 0.0:
            raise ValueError("sigma must be positive")

        self.mu = mu
        self.sigma = sigma
        self._spare: Optional[float] = None

    def sample(self) -> float:
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return self.mu + self.sigma * z

        while True:
            u = 2.0 * self.engine.random() - 1.0
            v = 2.0 * self.engine.random() - 1.0

            s = u * u + v * v

            if s == 0.0 or s >= 1.0:
                continue

            factor = math.sqrt(-2.0 * math.log(s) / s)
            z0 = u * factor
            z1 = v * factor

            self._spare = z1
            return self.mu + self.sigma * z0

    def __repr__(self) -> str:
        return f"NormalDistribution(mu={self.mu}, sigma={self.sigma})"





## Exponential distribution with rate lambda_.
##
##    Density:
##       f(x) = lambda * exp(-lambda x),   x >= 0
##
class ExponentialDistribution(Distribution):

    def __init__(
        self,
        engine: MersenneTwisterEngine,
        lambda_: float = 1.0,
    ) -> None:
        super().__init__(engine)

        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive")

        self.lambda_ = lambda_

    def sample(self) -> float:
        u = self.engine.random()

        # avoid log(0)
        while u <= 0.0:
            u = self.engine.random()

        return -math.log(u) / self.lambda_

    def __repr__(self) -> str:
        return f"ExponentialDistribution(lambda_={self.lambda_})"





## Poisson(lambda) using Knuth's algorithm.
##
##    Good for modest lambda.  For large lambda (around 30), a different method may be preferable.
##
class PoissonDistribution(Distribution):
    def __init__(
        self,
        engine: MersenneTwisterEngine,
        lambda_: float,
    ) -> None:
        super().__init__(engine)

        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be positive")

        self.lambda_ = lambda_

    def sample(self) -> int:
        limit = math.exp(-self.lambda_)
        k = 0
        p = 1.0

        while p > limit:
            k += 1
            p *= self.engine.random()

        return k - 1

    def __repr__(self) -> str:
        return f"PoissonDistribution(lambda_={self.lambda_})"