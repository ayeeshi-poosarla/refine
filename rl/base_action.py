"""BaseAction — abstract interface for all RL rubric-editing actions."""

from abc import ABC, abstractmethod

from rl.state import RubricState


class BaseAction(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable action identifier."""
        ...

    @abstractmethod
    def apply(self, state: RubricState) -> RubricState:
        """Return a new RubricState with the action applied.

        Must never mutate the input state — call state.copy() first.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
