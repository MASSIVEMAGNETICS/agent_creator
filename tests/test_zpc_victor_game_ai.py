import pytest

from backend.core.zpc_victor_orch_or_game_ai_v2_4_0 import (
    MicroState,
    QuantumGameState,
    ReinforcementAgent,
    VictorGameAI,
)


# ------------------------------------------------------------------ QuantumGameState

class TestQuantumGameState:
    def test_initial_amplitudes_uniform(self):
        qgs = QuantumGameState(num_states=4)
        amps = qgs.get_state_vector()
        assert len(amps) == 4
        # All equal initially
        assert all(abs(a - amps[0]) < 1e-6 for a in amps)

    def test_observe_returns_micro_state(self):
        qgs = QuantumGameState(num_states=4)
        obs = qgs.observe()
        assert isinstance(obs, MicroState)

    def test_observe_state_id_in_range(self):
        qgs = QuantumGameState(num_states=8)
        obs = qgs.observe()
        assert 0 <= obs.state_id < 8

    def test_update_amplitudes_changes_vector(self):
        qgs = QuantumGameState(num_states=4)
        before = list(qgs.get_state_vector())
        qgs.update_amplitudes({0: 1.0})
        after = qgs.get_state_vector()
        assert before != after

    def test_amplitudes_remain_positive_after_update(self):
        qgs = QuantumGameState(num_states=4)
        qgs.update_amplitudes({0: -10.0, 1: -10.0, 2: -10.0, 3: -10.0})
        for a in qgs.get_state_vector():
            assert a > 0.0


# ------------------------------------------------------------------ ReinforcementAgent

class TestReinforcementAgent:
    def test_select_action_returns_valid_action(self):
        agent = ReinforcementAgent(["left", "right", "jump"])
        state = MicroState(state_id=0, features={"index": 0.0})
        action = agent.select_action(state)
        assert action in ["left", "right", "jump"]

    def test_update_returns_float(self):
        agent = ReinforcementAgent(["left", "right"])
        new_q = agent.update(0, "left", reward=1.0, next_state_id=1)
        assert isinstance(new_q, float)

    def test_q_value_updated(self):
        agent = ReinforcementAgent(["A", "B"])
        agent.update(0, "A", reward=1.0, next_state_id=1)
        assert agent._q_value(0, "A") != 0.0

    def test_epsilon_decays_over_steps(self):
        agent = ReinforcementAgent(["A"])
        initial = agent._epsilon
        for _ in range(100):
            agent.update(0, "A", 0.0, 0)
        assert agent._epsilon <= initial

    def test_get_policy_returns_dict(self):
        agent = ReinforcementAgent(["X", "Y"])
        agent.update(0, "X", reward=1.0, next_state_id=1)
        policy = agent.get_policy()
        assert isinstance(policy, dict)
        assert "0" in policy


# ------------------------------------------------------------------ VictorGameAI

class TestVictorGameAI:
    @pytest.fixture
    def ai(self) -> VictorGameAI:
        return VictorGameAI(actions=["move_left", "move_right", "jump", "wait"])

    def test_step_returns_valid_action(self, ai):
        action = ai.step(reward=0.0)
        assert action in ["move_left", "move_right", "jump", "wait"]

    def test_step_increments_history(self, ai):
        ai.step()
        ai.step()
        stats = ai.get_stats()
        assert stats["total_steps"] == 2

    def test_total_reward_accumulates(self, ai):
        ai.step(reward=1.0)
        ai.step(reward=2.0)
        assert ai.get_stats()["total_reward"] == pytest.approx(3.0)

    def test_multiple_steps_no_error(self, ai):
        for i in range(20):
            ai.step(reward=float(i % 3))

    def test_stats_keys_present(self, ai):
        ai.step()
        stats = ai.get_stats()
        for key in ("total_steps", "total_reward", "epsilon", "policy", "state_vector"):
            assert key in stats

    def test_custom_actions(self):
        ai = VictorGameAI(actions=["fire", "shield"])
        action = ai.step()
        assert action in ["fire", "shield"]

    def test_default_actions_exist(self):
        ai = VictorGameAI()
        assert len(ai.actions) > 0

    def test_done_flag_ends_episode(self, ai):
        for _ in range(5):
            ai.step(reward=1.0)
        action = ai.step(reward=0.0, done=True)
        assert action in ai.actions
