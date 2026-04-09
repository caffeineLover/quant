from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List





##  Abstract base class for stochastic processes.
##
##     A process should know how to evolve one step forward and how to generate a full path.
##
class StochasticProcess(ABC):

    # Advance the process by one time step of length dt.
    @abstractmethod
    def next_value(self, current_value: Any, dt: float) -> Any:
        raise NotImplementedError



    # Generate a path of length steps + 1, including the initial value.
    def generate_path(self, initial_value: Any, dt: float, steps: int) -> List[Any]:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        path = [initial_value]
        x = initial_value

        for _ in range(steps):
            x = self.next_value(x, dt)
            path.append(x)

        return path