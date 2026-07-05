import numpy as np

# Fixed observation size: every level (6x6 up to 12x12) is padded to this
# grid so a single CNN can handle all level sizes.
OBS_SIZE = 12

# Channel layout of the spatial observation.
CHANNEL_WALLS = 0
CHANNEL_AGENT = 1
CHANNEL_KEY = 2
CHANNEL_DOOR_CLOSED = 3
CHANNEL_DOOR_OPEN = 4
CHANNEL_EXIT = 5
CHANNEL_PUZZLES = 6
CHANNEL_HAS_KEY = 7
CHANNEL_DOOR_IS_OPEN = 8
CHANNEL_PUZZLES_SOLVED_FRACTION = 9
CHANNEL_STEPS_REMAINING_FRACTION = 10

NUM_CHANNELS = 11


def build_observation(env, size=OBS_SIZE):
    """
    Build a (NUM_CHANNELS, size, size) float32 observation of the current
    environment state.

    Channels 0-6 are one-hot spatial maps (walls, agent, key if not taken,
    closed door, open door, exit, unsolved puzzles). Channels 7-10 broadcast
    scalar state over the whole grid (has_key, door_open, fraction of solved
    puzzles, fraction of steps remaining).

    Levels smaller than `size` are padded at the bottom/right and the padded
    cells are marked as walls, so "outside the level" and "wall" share the
    same meaning: the agent cannot move there.
    """
    if env.rows > size or env.cols > size:
        raise ValueError(
            f"Level {env.rows}x{env.cols} does not fit in a {size}x{size} "
            f"observation."
        )

    observation = np.zeros((NUM_CHANNELS, size, size), dtype=np.float32)

    # padding outside the real grid counts as wall
    observation[CHANNEL_WALLS, env.rows:, :] = 1.0
    observation[CHANNEL_WALLS, :, env.cols:] = 1.0

    for row, col in env.walls:
        observation[CHANNEL_WALLS, row, col] = 1.0

    agent_row, agent_col = env.agent_position
    observation[CHANNEL_AGENT, agent_row, agent_col] = 1.0

    if not env.has_key:
        key_row, key_col = env.key_position
        observation[CHANNEL_KEY, key_row, key_col] = 1.0

    door_row, door_col = env.door_position
    door_channel = CHANNEL_DOOR_OPEN if env.door_open else CHANNEL_DOOR_CLOSED
    observation[door_channel, door_row, door_col] = 1.0

    exit_row, exit_col = env.exit_position
    observation[CHANNEL_EXIT, exit_row, exit_col] = 1.0

    for puzzle in env.puzzles:
        if not puzzle.solved:
            puzzle_row, puzzle_col = puzzle.position
            observation[CHANNEL_PUZZLES, puzzle_row, puzzle_col] = 1.0

    observation[CHANNEL_HAS_KEY] = float(env.has_key)
    observation[CHANNEL_DOOR_IS_OPEN] = float(env.door_open)

    if env.puzzles:
        solved = sum(1 for puzzle in env.puzzles if puzzle.solved)
        observation[CHANNEL_PUZZLES_SOLVED_FRACTION] = solved / len(env.puzzles)
    else:
        observation[CHANNEL_PUZZLES_SOLVED_FRACTION] = 1.0

    remaining = max(0, env.max_steps - env.steps)
    observation[CHANNEL_STEPS_REMAINING_FRACTION] = remaining / env.max_steps

    return observation
