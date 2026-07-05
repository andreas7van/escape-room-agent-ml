import os

import matplotlib.pyplot as plt
import pandas as pd

from agents.q_learning_agent import QLearningAgent
from environment.escape_room_env import EscapeRoomEnv


def calculate_moving_average(values, window_size=100):
    moving_average = []

    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        moving_average.append(sum(window) / len(window))

    return moving_average


def train_agent(
    episodes=6000,
    max_steps=150,
    model_path="models/q_table_fixed.pkl",
    stats_path="results/q_fixed_training_stats.csv",
    reward_plot_path="results/q_fixed_rewards.png",
    success_plot_path="results/q_fixed_success_rate.png",
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = EscapeRoomEnv(max_steps=max_steps)

    agent = QLearningAgent(
        action_space_size=env.action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.997,
        min_epsilon=0.03,
    )

    training_data = []

    for episode in range(1, episodes + 1):
        state = env.reset()

        total_reward = 0
        success = False

        solved_puzzles_count = 0
        collected_key = False
        opened_door = False

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

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

        training_data.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "steps": env.steps,
                "success": int(success),
                "epsilon": agent.epsilon,
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

            print(
                f"Episode {episode}/{episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Success Rate: {success_rate:.2%} | "
                f"Avg Steps: {avg_steps:.2f} | "
                f"Avg Solved Puzzles: {avg_solved_puzzles:.2f} | "
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
    plt.title("Q-Learning Training Rewards - Escape Room")
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
    plt.title("Q-Learning Success Rate - Escape Room")
    plt.legend()
    plt.grid(True)
    plt.savefig(success_plot_path)
    plt.close()

    print()
    print("Training completed.")
    print(f"Model saved at: {model_path}")
    print(f"Stats saved at: {stats_path}")
    print(f"Reward plot saved at: {reward_plot_path}")
    print(f"Success plot saved at: {success_plot_path}")


if __name__ == "__main__":
    train_agent()
