import argparse
import os

import pandas as pd

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.escape_room_env import EscapeRoomEnv


LEVEL_PATHS = [
    "config/levels/level_1.json",
    "config/levels/level_2.json",
    "config/levels/level_3.json",
]


def evaluate_random_levels(
    agent_type="q",
    episodes_per_level=3,
    max_steps=150,
    render=True,
):
    agent_type = agent_type.lower()

    first_env = EscapeRoomEnv(
        max_steps=max_steps,
        level_path=LEVEL_PATHS[0],
        include_level_in_state=True,
    )

    first_state = first_env.reset()

    if agent_type == "q":
        model_path = "models/q_table_random_levels.pkl"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run train_random_levels.py first."
            )

        agent = QLearningAgent(action_space_size=first_env.action_space_size)
        agent.load(model_path)

    elif agent_type == "dqn":
        model_path = "models/dqn_random_levels.pth"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run train_random_levels.py first."
            )

        agent = DQNAgent(
            state_size=len(first_state),
            action_space_size=first_env.action_space_size,
            grid_rows=first_env.rows,
            grid_cols=first_env.cols,
        )
        agent.load(model_path)

    else:
        raise ValueError("agent_type must be either 'q' or 'dqn'.")

    evaluation_rows = []

    total_successes = 0
    total_episodes = 0
    total_rewards = 0
    total_steps = 0

    for level_path in LEVEL_PATHS:
        env = EscapeRoomEnv(
            max_steps=max_steps,
            level_path=level_path,
            include_level_in_state=True,
        )

        level_successes = 0
        level_rewards = 0
        level_steps = 0

        print()
        print("=" * 60)
        print(f"Evaluating {agent_type.upper()} on level: {env.level_name}")
        print("=" * 60)

        for episode in range(1, episodes_per_level + 1):
            state = env.reset()
            episode_reward = 0

            if render:
                env.render()

            for step in range(max_steps):
                action = agent.choose_action(state, training=False)
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                state = next_state

                if render:
                    print(
                        f"Step {step + 1:03d} | "
                        f"Action: {info['action']:<13} | "
                        f"Reward: {reward:<4} | "
                        f"Message: {info['message']}"
                    )
                    env.render()

                if done:
                    break

            success = env.agent_position == env.exit_position and env.door_open
            solved_puzzles = sum(1 for puzzle in env.puzzles if puzzle.solved)

            if success:
                level_successes += 1
                total_successes += 1

            level_rewards += episode_reward
            level_steps += env.steps

            total_rewards += episode_reward
            total_steps += env.steps
            total_episodes += 1

            evaluation_rows.append(
                {
                    "agent": agent_type.upper(),
                    "level_id": env.current_level_index,
                    "level_name": env.level_name,
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": env.steps,
                    "success": int(success),
                    "solved_puzzles": solved_puzzles,
                    "collected_key": int(env.has_key),
                    "door_open": int(env.door_open),
                }
            )

            print()
            print("Episode Summary")
            print("-" * 60)
            print(f"Level: {env.level_name}")
            print(f"Episode: {episode}")
            print(f"Reward: {episode_reward}")
            print(f"Steps: {env.steps}")
            print(f"Success: {success}")
            print(f"Solved puzzles: {solved_puzzles}/{len(env.puzzles)}")
            print(f"Collected key: {env.has_key}")
            print(f"Door open: {env.door_open}")

        print()
        print("Level Summary")
        print("-" * 60)
        print(f"Level: {env.level_name}")
        print(f"Success rate: {level_successes / episodes_per_level:.2%}")
        print(f"Average reward: {level_rewards / episodes_per_level:.2f}")
        print(f"Average steps: {level_steps / episodes_per_level:.2f}")

    output_path = f"results/{agent_type}_random_levels_evaluation.csv"
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(evaluation_rows)
    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print("Final Random Levels Evaluation Results")
    print("=" * 60)
    print(f"Agent: {agent_type.upper()}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {total_successes / total_episodes:.2%}")
    print(f"Average reward: {total_rewards / total_episodes:.2f}")
    print(f"Average steps: {total_steps / total_episodes:.2f}")
    print(f"Evaluation CSV saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        choices=["q", "dqn"],
        default="q",
        help="Choose agent type: q or dqn",
    )

    parser.add_argument(
        "--episodes-per-level",
        type=int,
        default=3,
        help="Evaluation episodes for each level",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum steps per episode",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable terminal rendering",
    )

    args = parser.parse_args()

    evaluate_random_levels(
        agent_type=args.agent,
        episodes_per_level=args.episodes_per_level,
        max_steps=args.max_steps,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()