import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.escape_room_env import EscapeRoomEnv


LEVEL_PATHS = [
    "config/levels/level_1.json",
    "config/levels/level_2.json",
    "config/levels/level_3.json",
]


def calculate_moving_average(values, window_size=100):
    moving_average = []

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        moving_average.append(sum(window) / len(window))

    return moving_average


def train_random_levels(
    agent_type="q",
    episodes=9000,
    max_steps=150,
    target_update_frequency=20,
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    agent_type = agent_type.lower()

    env = EscapeRoomEnv(
        max_steps=max_steps,
        level_paths=LEVEL_PATHS,
        random_level=True,
        include_level_in_state=True,
    )

    initial_state = env.reset()
    state_size = len(initial_state)

    if agent_type == "q":
        agent = QLearningAgent(
            action_space_size=env.action_space_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.997,
            min_epsilon=0.03,
        )

        model_path = "models/q_table_random_levels.pkl"
        stats_path = "results/q_random_levels_training_stats.csv"
        reward_plot_path = "results/q_random_levels_rewards.png"
        success_plot_path = "results/q_random_levels_success_rate.png"

    elif agent_type == "dqn":
        agent = DQNAgent(
            state_size=state_size,
            action_space_size=env.action_space_size,
            grid_rows=env.rows,
            grid_cols=env.cols,
            learning_rate=0.001,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.997,
            min_epsilon=0.03,
            batch_size=64,
        )

        print(f"Using device: {agent.device}")

        model_path = "models/dqn_random_levels.pth"
        stats_path = "results/dqn_random_levels_training_stats.csv"
        reward_plot_path = "results/dqn_random_levels_rewards.png"
        success_plot_path = "results/dqn_random_levels_success_rate.png"

    else:
        raise ValueError("agent_type must be either 'q' or 'dqn'.")

    training_data = []

    for episode in range(1, episodes + 1):
        state = env.reset()

        total_reward = 0
        success = False
        losses = []

        collected_key = False
        opened_door = False
        solved_puzzles_count = 0

        episode_level_id = env.current_level_index
        episode_level_name = env.level_name

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            if agent_type == "q":
                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

            elif agent_type == "dqn":
                agent.remember(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                loss = agent.train_step()

                if loss is not None:
                    losses.append(loss)

            state = next_state
            total_reward += reward

            solved_puzzles_count = sum(
                1 for puzzle in env.puzzles if puzzle.solved
            )

            if env.has_key:
                collected_key = True

            if env.door_open:
                opened_door = True

            if done:
                if env.agent_position == env.exit_position and env.door_open:
                    success = True
                break

        agent.decay_epsilon()

        if agent_type == "dqn" and episode % target_update_frequency == 0:
            agent.update_target_network()

        average_loss = sum(losses) / len(losses) if losses else 0

        training_data.append(
            {
                "episode": episode,
                "level_id": episode_level_id,
                "level_name": episode_level_name,
                "total_reward": total_reward,
                "steps": env.steps,
                "success": int(success),
                "epsilon": agent.epsilon,
                "loss": average_loss,
                "collected_key": int(collected_key),
                "solved_puzzles": solved_puzzles_count,
                "opened_door": int(opened_door),
            }
        )

        if episode % 100 == 0:
            recent_data = training_data[-100:]

            avg_reward = (
                sum(item["total_reward"] for item in recent_data)
                / len(recent_data)
            )

            success_rate = (
                sum(item["success"] for item in recent_data)
                / len(recent_data)
            )

            avg_steps = (
                sum(item["steps"] for item in recent_data)
                / len(recent_data)
            )

            avg_solved_puzzles = (
                sum(item["solved_puzzles"] for item in recent_data)
                / len(recent_data)
            )

            avg_loss = (
                sum(item["loss"] for item in recent_data)
                / len(recent_data)
            )

            print(
                f"Episode {episode}/{episodes} | "
                f"Agent: {agent_type.upper()} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Success Rate: {success_rate:.2%} | "
                f"Avg Steps: {avg_steps:.2f} | "
                f"Avg Solved Puzzles: {avg_solved_puzzles:.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save(model_path)

    df = pd.DataFrame(training_data)

    df["moving_avg_reward"] = calculate_moving_average(
        df["total_reward"].tolist(),
        window_size=100,
    )

    df["moving_success_rate"] = calculate_moving_average(
        df["success"].tolist(),
        window_size=100,
    )

    df.to_csv(stats_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["total_reward"], alpha=0.3, label="Reward")
    plt.plot(
        df["episode"],
        df["moving_avg_reward"],
        label="Moving Average Reward",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{agent_type.upper()} Training Rewards - Random Levels")
    plt.legend()
    plt.grid(True)
    plt.savefig(reward_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        df["episode"],
        df["moving_success_rate"],
        label="Moving Success Rate",
    )
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title(f"{agent_type.upper()} Success Rate - Random Levels")
    plt.legend()
    plt.grid(True)
    plt.savefig(success_plot_path)
    plt.close()

    print()
    print("Random-level training completed.")
    print(f"Agent: {agent_type.upper()}")
    print(f"Model saved at: {model_path}")
    print(f"Stats saved at: {stats_path}")
    print(f"Reward plot saved at: {reward_plot_path}")
    print(f"Success plot saved at: {success_plot_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        choices=["q", "dqn"],
        default="q",
        help="Choose agent type: q or dqn",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=9000,
        help="Number of training episodes",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum steps per episode",
    )

    args = parser.parse_args()

    train_random_levels(
        agent_type=args.agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()