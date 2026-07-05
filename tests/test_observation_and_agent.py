import numpy as np
import pytest
import torch

from agents.cnn_dqn_agent import CNNDQNAgent
from environment import observation as obs_module
from environment.escape_room_env import EscapeRoomEnv
from environment.level_generator import LevelGenerator
from environment.observation import (
    NUM_CHANNELS,
    OBS_SIZE,
    build_observation,
)


@pytest.fixture
def small_env():
    level_data = LevelGenerator().generate(
        rows=6, cols=8, wall_density=0.15, num_puzzles=2, seed=5
    )
    return EscapeRoomEnv(level_data=level_data, max_steps=200)


class TestObservation:
    def test_shape_and_dtype(self, small_env):
        observation = build_observation(small_env)
        assert observation.shape == (NUM_CHANNELS, OBS_SIZE, OBS_SIZE)
        assert observation.dtype == np.float32

    def test_padding_marked_as_wall(self, small_env):
        observation = build_observation(small_env)
        walls = observation[obs_module.CHANNEL_WALLS]

        # rows 6..11 and cols 8..11 are outside the 6x8 level
        assert walls[small_env.rows:, :].min() == 1.0
        assert walls[:, small_env.cols:].min() == 1.0

    def test_one_hot_positions(self, small_env):
        observation = build_observation(small_env)

        agent_map = observation[obs_module.CHANNEL_AGENT]
        assert agent_map.sum() == 1.0
        assert agent_map[small_env.agent_position] == 1.0

        key_map = observation[obs_module.CHANNEL_KEY]
        assert key_map[small_env.key_position] == 1.0

        assert observation[obs_module.CHANNEL_DOOR_CLOSED][
            small_env.door_position
        ] == 1.0
        assert observation[obs_module.CHANNEL_DOOR_OPEN].sum() == 0.0

        assert observation[obs_module.CHANNEL_EXIT][small_env.exit_position] == 1.0

        puzzle_map = observation[obs_module.CHANNEL_PUZZLES]
        assert puzzle_map.sum() == len(small_env.puzzles)

    def test_scalar_channels_track_state(self, small_env):
        observation = build_observation(small_env)
        assert observation[obs_module.CHANNEL_HAS_KEY].max() == 0.0
        assert observation[obs_module.CHANNEL_STEPS_REMAINING_FRACTION].min() == 1.0

        small_env.has_key = True
        small_env.door_open = True
        small_env.puzzles[0].solved = True
        small_env.steps = 100

        observation = build_observation(small_env)
        assert observation[obs_module.CHANNEL_HAS_KEY].min() == 1.0
        assert observation[obs_module.CHANNEL_DOOR_IS_OPEN].min() == 1.0
        assert observation[obs_module.CHANNEL_KEY].sum() == 0.0
        assert observation[obs_module.CHANNEL_DOOR_OPEN][
            small_env.door_position
        ] == 1.0
        assert observation[obs_module.CHANNEL_DOOR_CLOSED].sum() == 0.0
        assert observation[obs_module.CHANNEL_PUZZLES].sum() == len(
            small_env.puzzles
        ) - 1
        assert observation[
            obs_module.CHANNEL_PUZZLES_SOLVED_FRACTION
        ].max() == pytest.approx(0.5)
        assert observation[
            obs_module.CHANNEL_STEPS_REMAINING_FRACTION
        ].max() == pytest.approx(0.5)

    @pytest.mark.parametrize("rows,cols", [(6, 6), (9, 7), (12, 12)])
    def test_all_sizes_produce_fixed_shape(self, rows, cols):
        level_data = LevelGenerator().generate(
            rows=rows, cols=cols, wall_density=0.15, num_puzzles=1, seed=2
        )
        env = EscapeRoomEnv(level_data=level_data)
        observation = build_observation(env)
        assert observation.shape == (NUM_CHANNELS, OBS_SIZE, OBS_SIZE)


class TestCNNDQNAgent:
    @pytest.fixture
    def agent(self):
        return CNNDQNAgent(
            action_space_size=7,
            batch_size=8,
            device=torch.device("cpu"),
        )

    def test_choose_action_valid(self, agent, small_env):
        observation = build_observation(small_env)
        action = agent.choose_action(observation, training=False)
        assert 0 <= action < 7

    def test_train_step_decreases_nothing_before_batch(self, agent, small_env):
        assert agent.train_step() is None

    def test_train_step_returns_loss(self, agent, small_env):
        observation = build_observation(small_env)

        for _ in range(10):
            action = np.random.randint(0, 7)
            _, reward, done, _ = small_env.step(action)
            next_observation = build_observation(small_env)
            agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation

        loss = agent.train_step()
        assert loss is not None
        assert np.isfinite(loss)

    def test_double_dqn_uses_policy_argmax(self, agent):
        """The chosen next action must come from the policy network."""
        observations = np.random.rand(8, NUM_CHANNELS, OBS_SIZE, OBS_SIZE).astype(
            np.float32
        )
        tensor = torch.tensor(observations)
        with torch.no_grad():
            policy_actions = agent.policy_network(tensor).argmax(dim=1)
            target_actions = agent.target_network(tensor).argmax(dim=1)
        # target starts as a copy of policy, so they must agree initially
        assert torch.equal(policy_actions, target_actions)

    def test_save_load_roundtrip(self, agent, tmp_path, small_env):
        observation = build_observation(small_env)
        action_before = agent.choose_action(observation, training=False)

        path = tmp_path / "cnn_dqn_test.pth"
        agent.save(str(path))

        restored = CNNDQNAgent(action_space_size=7, device=torch.device("cpu"))
        restored.load(str(path))

        action_after = restored.choose_action(observation, training=False)
        assert action_before == action_after
        assert restored.epsilon == agent.epsilon
