from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional
from game_state import GameState

class Agent(ABC):
    @abstractmethod
    def get_action(self, state: GameState) -> Any:
        """
        Determines the next action based on the current game state.
        Returns None if no action is needed.
        """
        pass

class Environment(ABC):
    @abstractmethod
    def reset(self) -> GameState:
        """
        Resets the environment to a starting state and returns the initial observation.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[GameState, float, bool, Dict]:
        """
        Executes the action in the environment.
        Returns:
            state (GameState): The new state of the environment.
            reward (float): The reward received from the action.
            done (bool): Whether the episode (game) has ended.
            info (Dict): Diagnostic information.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment (e.g., shows the game window with overlays).
        """
        pass

