from __future__ import annotations
import random
from typing import Optional, Any, Sequence, TypeVar, Dict


"""
    Random number engine based on Python's built-in Mersenne Twister.

    * Provide uniform random numbers in [0, 1)
    * Handle seeding for reproducibility
    * Uses random.Random internally (MT19937)
    * Designed for extensibility (e.g., add more distributions)
    * Provides convenience methods for common distributions
    * Allows saving/restoring state for reproducibility
"""

T = TypeVar("T")



class MersenneTwisterEngine:

    def __init__(self, seed: Optional[int] = None) -> None:
        # If no seed is provided, a seed is used from system entropy.
        self._seed = seed
        self._rng = random.Random(seed)
        self._draw_count = 0



    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def draw_count(self) -> int:
        return self._draw_count
    


    # Return a uniform random float in [0, 1).
    def random(self) -> float:
        self._draw_count += 1
        return self._rng.random()



    # Return a uniform random float in [a, b).
    def uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        if b <= a:
            raise ValueError("b must be greater than a")

        return a + (b - a) * self.random()



    # Integer in [a, b], inclusive.
    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError("b must be >= a")

        return self._rng.randint(a, b)



    # Random element from a non-empty sequence.
    def choice(self, seq):
        if not seq:
            raise ValueError("sequence must not be empty")

        return self._rng.choice(seq)


    # Reset the RNG seed.
    def reseed(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng.seed(seed)
        self.draw_count = 0



    def get_state(self):
        """
        Get internal RNG state (for reproducibility).
        """
        return self._rng.getstate()

    def set_state(self, state) -> None:
        """
        Restore internal RNG state.
        """
        self._rng.setstate(state)



    # Lightweight metadata for logging experiments (not full state).
    def snapshot(self) -> Dict[str, Any]:
        return {
            "engine": "MersenneTwister",
            "seed": self._seed,
            "draw_count": self._draw_count,
        }



    # String representation of the class for debugging.
    def __repr__(self) -> str:
        return (
            f"MersenneTwisterEngine(seed={self._seed}, "
            f"draws={self._draw_count})"
        )