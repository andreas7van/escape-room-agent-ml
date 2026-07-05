import json
import os
import random
from collections import deque

from environment.puzzles import Puzzle


class EscapeRoomEnv:
    """
    Escape Room Environment με υποστήριξη πολλαπλών levels.

    Μπορεί να λειτουργήσει με:
    1. ένα συγκεκριμένο level JSON
    2. τυχαία επιλογή level σε κάθε episode

    State χωρίς multi-level training:
        (row, col, has_key, solved_puzzle_1, solved_puzzle_2, solved_puzzle_3, door_open)

    State με include_level_in_state=True:
        (row, col, has_key, solved_puzzle_1, solved_puzzle_2, solved_puzzle_3, door_open, level_id)

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
        4 = PICK_KEY
        5 = SOLVE_PUZZLE
        6 = OPEN_DOOR
    """

    ACTIONS = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "PICK_KEY",
        5: "SOLVE_PUZZLE",
        6: "OPEN_DOOR",
    }

    DEFAULT_LEVEL_PATH = "config/levels/level_1.json"

    def __init__(
        self,
        max_steps=150,
        level_path=None,
        level_paths=None,
        random_level=False,
        include_level_in_state=False,
        level_data=None,
    ):
        self.max_steps = max_steps

        if level_path is None:
            level_path = self.DEFAULT_LEVEL_PATH

        if level_paths is None:
            level_paths = [level_path]

        self.level_path = level_path
        self.level_paths = level_paths
        self.random_level = random_level
        self.include_level_in_state = include_level_in_state

        self.action_space_size = len(self.ACTIONS)

        self.current_level_path = None
        self.current_level_index = 0
        self.level_name = ""

        if level_data is not None:
            self.set_level_data(level_data)
        else:
            self.load_level(self.level_paths[0])
            self.validate_map()
        self.reset()

    def load_level(self, level_path):
        if not os.path.exists(level_path):
            raise FileNotFoundError(f"Level file not found: {level_path}")

        with open(level_path, "r", encoding="utf-8") as file:
            level_data = json.load(file)

        self.current_level_path = level_path
        self._apply_level_data(level_data)

    def set_level_data(self, level_data):
        """Load a level from an in-memory dict (same schema as the JSON files)
        and validate it. Used by the procedural level generator."""
        self.current_level_path = None
        self._apply_level_data(level_data)
        self.validate_map()

    def _apply_level_data(self, level_data):
        self.current_level_index = int(level_data.get("level_id", 0))
        self.level_name = level_data.get("level_name", f"Level {self.current_level_index}")

        self.rows = int(level_data["rows"])
        self.cols = int(level_data["cols"])

        self.start_position = tuple(level_data["start_position"])
        self.key_position = tuple(level_data["key_position"])
        self.door_position = tuple(level_data["door_position"])
        self.exit_position = tuple(level_data["exit_position"])

        self.walls = {tuple(wall) for wall in level_data["walls"]}

        self.puzzles = []
        for item in level_data["puzzles"]:
            puzzle = Puzzle(
                puzzle_id=item["id"],
                symbol=item["symbol"],
                question=item["question"],
                answer=item["answer"],
                reward=item["reward"],
                position=item["position"],
            )
            self.puzzles.append(puzzle)

    def choose_random_level(self):
        selected_level_path = random.choice(self.level_paths)
        self.load_level(selected_level_path)
        self.validate_map()

    def reset(self):
        if self.random_level:
            self.choose_random_level()

        self.agent_position = self.start_position
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.done = False

        for puzzle in self.puzzles:
            puzzle.reset()

        return self.get_state()

    def get_state(self):
        row, col = self.agent_position
        solved_flags = tuple(int(puzzle.solved) for puzzle in self.puzzles)

        state = (
            row,
            col,
            int(self.has_key),
            *solved_flags,
            int(self.door_open),
        )

        if self.include_level_in_state:
            state = (*state, self.current_level_index)

        return state

    def all_puzzles_solved(self):
        return all(puzzle.solved for puzzle in self.puzzles)

    def get_puzzle_at_current_position(self):
        for puzzle in self.puzzles:
            if puzzle.position == self.agent_position:
                return puzzle

        return None

    def is_inside_grid(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, position):
        return position in self.walls

    def is_adjacent_to_door(self):
        row, col = self.agent_position
        door_row, door_col = self.door_position

        distance = abs(row - door_row) + abs(col - door_col)
        return distance == 1

    def get_neighbors(self, position, door_passable=False):
        row, col = position

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        neighbors = []

        for candidate in candidates:
            if not self.is_inside_grid(candidate):
                continue

            if self.is_wall(candidate):
                continue

            if candidate == self.door_position and not door_passable:
                continue

            neighbors.append(candidate)

        return neighbors

    def reachable_positions_from_start(self, door_passable=False):
        visited = set()
        queue = deque([self.start_position])

        while queue:
            position = queue.popleft()

            if position in visited:
                continue

            visited.add(position)

            for neighbor in self.get_neighbors(position, door_passable=door_passable):
                if neighbor not in visited:
                    queue.append(neighbor)

        return visited

    def validate_map(self):
        reachable_without_door = self.reachable_positions_from_start(
            door_passable=False
        )

        important_positions = [self.key_position]
        important_positions.extend([puzzle.position for puzzle in self.puzzles])

        for position in important_positions:
            if position not in reachable_without_door:
                raise ValueError(
                    f"Important position is not reachable in {self.level_name}: {position}"
                )

        door_row, door_col = self.door_position

        door_neighbors = [
            (door_row - 1, door_col),
            (door_row + 1, door_col),
            (door_row, door_col - 1),
            (door_row, door_col + 1),
        ]

        door_adjacent_reachable = any(
            neighbor in reachable_without_door for neighbor in door_neighbors
        )

        if not door_adjacent_reachable:
            raise ValueError(
                f"The door cannot be reached from an adjacent cell in {self.level_name}."
            )

        reachable_with_door = self.reachable_positions_from_start(
            door_passable=True
        )

        if self.exit_position not in reachable_with_door:
            raise ValueError(
                f"Exit position is not reachable after opening the door in {self.level_name}."
            )

    def move_agent(self, action):
        row, col = self.agent_position

        if action == 0:
            new_position = (row - 1, col)
        elif action == 1:
            new_position = (row + 1, col)
        elif action == 2:
            new_position = (row, col - 1)
        elif action == 3:
            new_position = (row, col + 1)
        else:
            return False

        if not self.is_inside_grid(new_position):
            return False

        if self.is_wall(new_position):
            return False

        if new_position == self.door_position and not self.door_open:
            return False

        self.agent_position = new_position
        return True

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {
                "action": "NONE",
                "message": "Episode already finished.",
            }

        self.steps += 1

        reward = -1
        info = {
            "action": self.ACTIONS.get(action, "UNKNOWN"),
            "message": "",
            "level_name": self.level_name,
            "level_id": self.current_level_index,
        }

        if action in [0, 1, 2, 3]:
            moved = self.move_agent(action)

            if not moved:
                reward = -6
                info["message"] = "Invalid move."

        elif action == 4:
            if self.agent_position == self.key_position and not self.has_key:
                self.has_key = True
                reward = 15
                info["message"] = "Key collected."
            elif self.agent_position == self.key_position and self.has_key:
                reward = -5
                info["message"] = "Key already collected."
            else:
                reward = -6
                info["message"] = "There is no key here."

        elif action == 5:
            puzzle = self.get_puzzle_at_current_position()

            if puzzle is None:
                reward = -6
                info["message"] = "There is no puzzle here."
            else:
                solved, puzzle_reward, message = puzzle.solve()
                reward = puzzle_reward
                info["message"] = message

        elif action == 6:
            if not self.is_adjacent_to_door():
                reward = -6
                info["message"] = "Agent is not adjacent to the door."

            elif not self.has_key:
                reward = -10
                info["message"] = "Door needs a key."

            elif not self.all_puzzles_solved():
                reward = -10
                info["message"] = "All puzzles must be solved before opening the door."

            elif self.door_open:
                reward = -5
                info["message"] = "Door is already open."

            else:
                self.door_open = True
                reward = 40
                info["message"] = "Door opened."

        else:
            reward = -10
            info["message"] = "Unknown action."

        if self.agent_position == self.exit_position and self.door_open:
            reward = 150
            self.done = True
            info["message"] = "Agent escaped successfully."

        if self.steps >= self.max_steps:
            self.done = True
            info["message"] = "Maximum steps reached."

        return self.get_state(), reward, self.done, info

    def get_cell_symbol(self, position):
        if position == self.agent_position:
            return "A"

        if position in self.walls:
            return "#"

        if position == self.key_position and not self.has_key:
            return "K"

        if position == self.door_position:
            return "O" if self.door_open else "D"

        if position == self.exit_position:
            return "E"

        for puzzle in self.puzzles:
            if puzzle.position == position and not puzzle.solved:
                return puzzle.symbol

        return "."

    def render(self):
        grid = []

        for row in range(self.rows):
            grid_row = []

            for col in range(self.cols):
                position = (row, col)
                grid_row.append(self.get_cell_symbol(position))

            grid.append(grid_row)

        print()
        print(f"Escape Room Map - {self.level_name}")
        print("-" * 30)

        for row in grid:
            print(" ".join(row))

        print("-" * 30)
        print(f"State: {self.get_state()}")
        print(f"Level: {self.level_name}")
        print(f"Position: {self.agent_position}")
        print(f"Has key: {self.has_key}")
        print(f"Door open: {self.door_open}")
        print(f"Steps: {self.steps}/{self.max_steps}")

        print("Puzzles:")
        for puzzle in self.puzzles:
            status = "solved" if puzzle.solved else "not solved"
            print(f"  {puzzle.symbol} - {puzzle.puzzle_id}: {status}")

        print()