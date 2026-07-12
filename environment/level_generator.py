import random

from environment.escape_room_env import EscapeRoomEnv

# Seed split: evaluation must always run on levels the agent has NEVER seen
# during training, so training, validation and testing draw from disjoint
# seed ranges. Validation levels are used during training for model
# selection; the test range is reserved for the final evaluation only.
TRAIN_SEEDS = range(0, 1000)
TEST_SEEDS = range(1000, 1100)
VAL_SEEDS = range(1100, 1150)

MIN_SIZE = 6
MAX_SIZE = 12

# Puzzle templates reused for generated levels. Symbols M/L/P are the ones
# the pygame visualizer knows how to draw.
PUZZLE_TEMPLATES = [
    {
        "id": "math_puzzle",
        "symbol": "M",
        "question": "What is 3 + 4?",
        "answer": "7",
        "reward": 20,
    },
    {
        "id": "logic_puzzle",
        "symbol": "L",
        "question": "I speak without a mouth and hear without ears. What am I?",
        "answer": "echo",
        "reward": 25,
    },
    {
        "id": "pattern_puzzle",
        "symbol": "P",
        "question": "Complete the sequence: 2, 4, 8, 16, ?",
        "answer": "32",
        "reward": 30,
    },
]


class LevelGenerator:
    """
    Procedural generator of random, guaranteed-solvable escape room levels.

    A level is built in two steps:

    1. The exit is placed in a "pocket": every neighbour of the exit becomes
       a wall except one, which becomes the door. This guarantees the exit is
       only reachable through the door, so the key/puzzle/door mechanic is
       always meaningful.
    2. Walls are scattered at the requested density and the start, key and
       puzzle positions are placed on free cells.

    Every candidate level is validated with the existing BFS checks in
    EscapeRoomEnv.validate_map (key and puzzles reachable before the door
    opens, exit reachable after). Invalid candidates are rejected and a new
    one is drawn from the same seeded RNG stream, so generation is fully
    deterministic per (rows, cols, wall_density, num_puzzles, seed).
    """

    def __init__(self, max_attempts=500):
        self.max_attempts = max_attempts

    def generate(self, rows, cols, wall_density=0.15, num_puzzles=3, seed=0):
        if not (MIN_SIZE <= rows <= MAX_SIZE and MIN_SIZE <= cols <= MAX_SIZE):
            raise ValueError(
                f"rows and cols must be within [{MIN_SIZE}, {MAX_SIZE}], "
                f"got {rows}x{cols}."
            )

        if not 1 <= num_puzzles <= len(PUZZLE_TEMPLATES):
            raise ValueError(
                f"num_puzzles must be within [1, {len(PUZZLE_TEMPLATES)}], "
                f"got {num_puzzles}."
            )

        rng = random.Random(
            f"{rows}x{cols}|{round(wall_density, 4)}|{num_puzzles}|{seed}"
        )

        for _ in range(self.max_attempts):
            level_data = self._build_candidate(
                rng, rows, cols, wall_density, num_puzzles, seed
            )

            if level_data is None:
                continue

            try:
                EscapeRoomEnv(level_data=level_data)
            except ValueError:
                continue

            return level_data

        raise RuntimeError(
            f"Could not generate a solvable {rows}x{cols} level "
            f"(density={wall_density}, puzzles={num_puzzles}, seed={seed}) "
            f"after {self.max_attempts} attempts."
        )

    def _build_candidate(self, rng, rows, cols, wall_density, num_puzzles, seed):
        all_cells = [(row, col) for row in range(rows) for col in range(cols)]

        # --- exit pocket: all exit neighbours become walls except the door ---
        exit_position = rng.choice(all_cells)
        exit_neighbors = self._grid_neighbors(exit_position, rows, cols)

        if len(exit_neighbors) < 2:
            return None

        door_position = rng.choice(exit_neighbors)
        pocket_walls = [cell for cell in exit_neighbors if cell != door_position]

        reserved = {exit_position, door_position} | set(pocket_walls)

        # --- start, key and puzzles on free cells ---
        free_cells = [cell for cell in all_cells if cell not in reserved]

        needed = 2 + num_puzzles
        if len(free_cells) < needed:
            return None

        special_cells = rng.sample(free_cells, needed)
        start_position = special_cells[0]
        key_position = special_cells[1]
        puzzle_positions = special_cells[2:]

        # --- scatter the remaining walls at the requested density ---
        target_walls = int(round(wall_density * rows * cols))
        extra_walls = max(0, target_walls - len(pocket_walls))

        candidates = [
            cell for cell in free_cells if cell not in set(special_cells)
        ]
        extra_walls = min(extra_walls, len(candidates))
        walls = pocket_walls + rng.sample(candidates, extra_walls)

        puzzles = []
        for template, position in zip(PUZZLE_TEMPLATES, puzzle_positions):
            puzzle = dict(template)
            puzzle["position"] = list(position)
            puzzles.append(puzzle)

        return {
            "level_id": seed,
            "level_name": f"Generated {rows}x{cols} (seed {seed})",
            "rows": rows,
            "cols": cols,
            "start_position": list(start_position),
            "key_position": list(key_position),
            "door_position": list(door_position),
            "exit_position": list(exit_position),
            "walls": [list(wall) for wall in walls],
            "puzzles": puzzles,
        }

    @staticmethod
    def _grid_neighbors(position, rows, cols):
        row, col = position

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        return [
            (r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < cols
        ]


def generate_from_seed(seed, wall_density=None, num_puzzles=None):
    """
    Generate a level whose size, wall density and puzzle count are all
    derived deterministically from the seed. Used to sample varied levels
    from the train/test seed ranges.
    """
    rng = random.Random(f"params|{seed}")

    rows = rng.randint(MIN_SIZE, MAX_SIZE)
    cols = rng.randint(MIN_SIZE, MAX_SIZE)

    if wall_density is None:
        wall_density = rng.uniform(0.10, 0.25)

    if num_puzzles is None:
        num_puzzles = rng.randint(1, 3)

    generator = LevelGenerator()
    return generator.generate(
        rows=rows,
        cols=cols,
        wall_density=wall_density,
        num_puzzles=num_puzzles,
        seed=seed,
    )
