"""ZPC Victor Orch-OR Game AI v2.4.0

Reinforcement-learning game AI integrating quantum decision systems with
thought-metaphysics inspired state modelling.

Architecture
------------
- QuantumGameState  : tracks the game world as a superposition of micro-states
- ReinforcementAgent: selects actions via Orch-OR collapse + Q-value estimation
- VictorGameAI      : top-level controller tying both components together
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from backend.core.victor_core import VictorCore


@dataclass
class MicroState:
    """A single micro-state in the quantum game lattice."""

    state_id: int
    features: dict[str, float]
    reward: float = 0.0
    visits: int = 0


@dataclass
class GameTransition:
    from_state: int
    action: str
    to_state: int
    reward: float
    timestamp: float = field(default_factory=time.time)


class QuantumGameState:
    """
    Models a game world as a lattice of micro-states held in superposition.

    Each observation collapses to the highest-amplitude micro-state,
    which is then passed to the reinforcement agent.
    """

    def __init__(self, num_states: int = 8) -> None:
        self.num_states = num_states
        self._states: list[MicroState] = [
            MicroState(state_id=i, features={"index": float(i)})
            for i in range(num_states)
        ]
        self._amplitudes: list[float] = [1.0 / num_states] * num_states
        self._step = 0

    def observe(self) -> MicroState:
        """Return the most probable micro-state (highest amplitude²)."""
        best_idx = max(range(self.num_states), key=lambda i: self._amplitudes[i] ** 2)
        return self._states[best_idx]

    def update_amplitudes(self, reward_map: dict[int, float]) -> None:
        """Bias amplitudes toward states that received positive reward."""
        for i, state in enumerate(self._states):
            r = reward_map.get(state.state_id, 0.0)
            self._amplitudes[i] = max(1e-6, self._amplitudes[i] * (1.0 + 0.1 * r))
        # Renormalise
        total = math.sqrt(sum(a ** 2 for a in self._amplitudes)) or 1.0
        self._amplitudes = [a / total for a in self._amplitudes]
        self._step += 1

    def get_state_vector(self) -> list[float]:
        return list(self._amplitudes)


class ReinforcementAgent:
    """
    Q-learning agent whose action selection is mediated by VictorCore (Orch-OR).

    Q-values are tracked per (state_id, action) pair and updated via
    the Bellman equation after each transition.
    """

    GAMMA = 0.95         # Discount factor
    ALPHA = 0.1          # Learning rate
    EPSILON_INIT = 0.3   # Initial exploration rate
    EPSILON_MIN = 0.05

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self._q: dict[tuple[int, str], float] = {}
        self._epsilon = self.EPSILON_INIT
        self._step = 0
        self._core = VictorCore()

    def _q_value(self, state_id: int, action: str) -> float:
        return self._q.get((state_id, action), 0.0)

    def select_action(self, state: MicroState) -> str:
        """Select an action using Orch-OR collapse biased by Q-values."""
        q_vals = [self._q_value(state.state_id, a) for a in self.actions]
        # Shift so minimum is 0, then use as amplitudes
        min_q = min(q_vals)
        amplitudes = [max(1e-6, q - min_q + 0.1) for q in q_vals]
        result = self._core.decide(self.actions, amplitudes)
        return result.selected if result else self.actions[0]

    def update(
        self,
        state_id: int,
        action: str,
        reward: float,
        next_state_id: int,
        done: bool = False,
    ) -> float:
        """Apply Bellman update and return the new Q-value."""
        best_next = (
            max(self._q_value(next_state_id, a) for a in self.actions)
            if not done
            else 0.0
        )
        current = self._q_value(state_id, action)
        target = reward + self.GAMMA * best_next
        new_val = current + self.ALPHA * (target - current)
        self._q[(state_id, action)] = round(new_val, 6)

        # Decay exploration rate
        self._step += 1
        self._epsilon = max(
            self.EPSILON_MIN,
            self.EPSILON_INIT * math.exp(-0.001 * self._step),
        )
        return new_val

    def get_policy(self) -> dict[str, str]:
        """Return the greedy policy: best action for each known state."""
        known_states: set[int] = {k[0] for k in self._q}
        return {
            str(s): max(self.actions, key=lambda a: self._q_value(s, a))
            for s in sorted(known_states)
        }


class VictorGameAI:
    """
    Top-level Quantum Game AI controller.

    Integrates QuantumGameState and ReinforcementAgent into a single
    step/train interface callable from an external game loop.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        num_states: int = 8,
    ) -> None:
        self.actions = actions or ["move_left", "move_right", "jump", "wait"]
        self._game_state = QuantumGameState(num_states=num_states)
        self._agent = ReinforcementAgent(self.actions)
        self._history: list[GameTransition] = []
        self._total_reward = 0.0

    def step(self, reward: float = 0.0, done: bool = False) -> str:
        """
        Advance the game by one step:
        1. Observe current quantum state
        2. Select action via Orch-OR
        3. Apply reward and update Q-values
        4. Update state amplitudes
        Returns the selected action label.
        """
        current = self._game_state.observe()
        action = self._agent.select_action(current)

        # Simulate next state as a simple index rotation
        next_idx = (current.state_id + 1) % self._game_state.num_states
        self._agent.update(current.state_id, action, reward, next_idx, done)

        reward_map = {current.state_id: reward}
        self._game_state.update_amplitudes(reward_map)

        self._history.append(
            GameTransition(
                from_state=current.state_id,
                action=action,
                to_state=next_idx,
                reward=reward,
            )
        )
        self._total_reward += reward
        return action

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_steps": len(self._history),
            "total_reward": round(self._total_reward, 4),
            "epsilon": round(self._agent._epsilon, 4),
            "policy": self._agent.get_policy(),
            "state_vector": self._game_state.get_state_vector(),
        }
