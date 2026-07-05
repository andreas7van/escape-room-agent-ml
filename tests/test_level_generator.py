import pytest

from environment.escape_room_env import EscapeRoomEnv
from environment.level_generator import (
    MAX_SIZE,
    MIN_SIZE,
    TEST_SEEDS,
    TRAIN_SEEDS,
    LevelGenerator,
    generate_from_seed,
)


@pytest.fixture
def generator():
    return LevelGenerator()


class TestSolvability:
    @pytest.mark.parametrize("seed", range(0, 30))
    def test_generated_levels_pass_env_validation(self, generator, seed):
        level_data = generator.generate(
            rows=8, cols=8, wall_density=0.2, num_puzzles=2, seed=seed
        )

        # set_level_data raises ValueError if BFS validation fails
        env = EscapeRoomEnv(level_data=level_data)
        assert env.rows == 8
        assert env.cols == 8

    @pytest.mark.parametrize("seed", range(0, 10))
    def test_exit_only_reachable_through_door(self, generator, seed):
        level_data = generator.generate(
            rows=10, cols=10, wall_density=0.15, num_puzzles=3, seed=seed
        )
        env = EscapeRoomEnv(level_data=level_data)

        reachable_closed = env.reachable_positions_from_start(door_passable=False)
        reachable_open = env.reachable_positions_from_start(door_passable=True)

        assert env.exit_position not in reachable_closed
        assert env.exit_position in reachable_open

    @pytest.mark.parametrize("seed", range(0, 10))
    def test_scripted_agent_can_escape(self, generator, seed):
        """Walk BFS shortest paths key -> puzzles -> door -> exit and check
        the episode actually terminates with the escape reward."""
        level_data = generator.generate(
            rows=7, cols=9, wall_density=0.15, num_puzzles=2, seed=seed
        )
        env = EscapeRoomEnv(level_data=level_data, max_steps=500)
        env.reset()

        def bfs_path(start, goal, door_passable):
            from collections import deque

            queue = deque([(start, [start])])
            visited = {start}
            while queue:
                position, path = queue.popleft()
                if position == goal:
                    return path
                for neighbor in env.get_neighbors(position, door_passable):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            return None

        def walk_to(goal, door_passable=False):
            path = bfs_path(env.agent_position, goal, door_passable)
            assert path is not None, f"No path to {goal}"
            for cell in path[1:]:
                row, col = env.agent_position
                delta = (cell[0] - row, cell[1] - col)
                action = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}[delta]
                env.step(action)
                assert env.agent_position == cell

        walk_to(env.key_position)
        _, reward, _, _ = env.step(4)
        assert env.has_key

        for puzzle in env.puzzles:
            walk_to(puzzle.position)
            env.step(5)
            assert puzzle.solved

        # stand next to the door, open it, then walk through to the exit
        door_neighbors = [
            cell
            for cell in env.get_neighbors(env.door_position, door_passable=True)
            if cell != env.exit_position
        ]
        reachable = env.reachable_positions_from_start(door_passable=False)
        adjacent = [cell for cell in door_neighbors if cell in reachable]
        assert adjacent
        walk_to(adjacent[0])
        env.step(6)
        assert env.door_open

        walk_to(env.exit_position, door_passable=True)
        assert env.done


class TestDeterminism:
    def test_same_seed_same_level(self, generator):
        level_a = generator.generate(9, 11, wall_density=0.18, num_puzzles=3, seed=42)
        level_b = generator.generate(9, 11, wall_density=0.18, num_puzzles=3, seed=42)
        assert level_a == level_b

    def test_different_seeds_differ(self, generator):
        levels = [
            generator.generate(8, 8, wall_density=0.15, num_puzzles=2, seed=seed)
            for seed in range(5)
        ]
        unique = {str(level) for level in levels}
        assert len(unique) == len(levels)

    def test_generate_from_seed_deterministic(self):
        level_a = generate_from_seed(1234)
        level_b = generate_from_seed(1234)
        assert level_a == level_b


class TestSizeAndStructure:
    @pytest.mark.parametrize("rows,cols", [(6, 6), (6, 12), (12, 6), (12, 12)])
    def test_size_bounds_respected(self, generator, rows, cols):
        level_data = generator.generate(
            rows, cols, wall_density=0.15, num_puzzles=1, seed=7
        )
        assert level_data["rows"] == rows
        assert level_data["cols"] == cols

        positions = (
            [tuple(level_data["start_position"])]
            + [tuple(level_data["key_position"])]
            + [tuple(level_data["door_position"])]
            + [tuple(level_data["exit_position"])]
            + [tuple(wall) for wall in level_data["walls"]]
            + [tuple(p["position"]) for p in level_data["puzzles"]]
        )
        for row, col in positions:
            assert 0 <= row < rows
            assert 0 <= col < cols

    def test_rejects_out_of_range_sizes(self, generator):
        with pytest.raises(ValueError):
            generator.generate(MIN_SIZE - 1, 8, seed=0)
        with pytest.raises(ValueError):
            generator.generate(8, MAX_SIZE + 1, seed=0)

    @pytest.mark.parametrize("num_puzzles", [1, 2, 3])
    def test_puzzle_count(self, generator, num_puzzles):
        level_data = generator.generate(
            10, 10, wall_density=0.15, num_puzzles=num_puzzles, seed=3
        )
        assert len(level_data["puzzles"]) == num_puzzles

    def test_special_positions_are_distinct_and_free(self, generator):
        level_data = generator.generate(
            8, 8, wall_density=0.25, num_puzzles=3, seed=11
        )

        specials = [
            tuple(level_data["start_position"]),
            tuple(level_data["key_position"]),
            tuple(level_data["door_position"]),
            tuple(level_data["exit_position"]),
        ] + [tuple(p["position"]) for p in level_data["puzzles"]]

        assert len(specials) == len(set(specials))

        walls = {tuple(wall) for wall in level_data["walls"]}
        for position in specials:
            assert position not in walls

    def test_generate_from_seed_sizes_in_range(self):
        for seed in list(TEST_SEEDS)[:15]:
            level_data = generate_from_seed(seed)
            assert MIN_SIZE <= level_data["rows"] <= MAX_SIZE
            assert MIN_SIZE <= level_data["cols"] <= MAX_SIZE


class TestSeedSplit:
    def test_train_and_test_seeds_disjoint(self):
        assert set(TRAIN_SEEDS).isdisjoint(set(TEST_SEEDS))

    def test_expected_ranges(self):
        assert len(TRAIN_SEEDS) == 1000
        assert len(TEST_SEEDS) == 100


class TestExistingLevelsStillLoad:
    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_json_levels_unchanged(self, level):
        env = EscapeRoomEnv(level_path=f"config/levels/level_{level}.json")
        assert env.rows == 6
        assert env.cols == 6
        assert len(env.puzzles) == 3
