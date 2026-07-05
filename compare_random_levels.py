import os

import matplotlib.pyplot as plt
import pandas as pd


def ensure_file_exists(path, command_hint):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run this first:\n{command_hint}"
        )


def add_moving_columns(df):
    if "moving_avg_reward" not in df.columns:
        df["moving_avg_reward"] = df["total_reward"].rolling(
            window=100,
            min_periods=1,
        ).mean()

    if "moving_success_rate" not in df.columns:
        df["moving_success_rate"] = df["success"].rolling(
            window=100,
            min_periods=1,
        ).mean()

    return df


def compare_training_results(
    q_training_path="results/q_random_levels_training_stats.csv",
    dqn_training_path="results/dqn_random_levels_training_stats.csv",
):
    ensure_file_exists(
        q_training_path,
        "python train_random_levels.py --agent q --episodes 9000",
    )

    ensure_file_exists(
        dqn_training_path,
        "python train_random_levels.py --agent dqn --episodes 12000",
    )

    q_df = pd.read_csv(q_training_path)
    dqn_df = pd.read_csv(dqn_training_path)

    q_df = add_moving_columns(q_df)
    dqn_df = add_moving_columns(dqn_df)

    reward_plot_path = "results/random_levels_q_vs_dqn_reward.png"

    plt.figure(figsize=(10, 5))
    plt.plot(
        q_df["episode"],
        q_df["moving_avg_reward"],
        label="Q-Learning",
    )
    plt.plot(
        dqn_df["episode"],
        dqn_df["moving_avg_reward"],
        label="DQN",
    )
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Random Levels - Q-Learning vs DQN Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(reward_plot_path)
    plt.close()

    success_plot_path = "results/random_levels_q_vs_dqn_success.png"

    plt.figure(figsize=(10, 5))
    plt.plot(
        q_df["episode"],
        q_df["moving_success_rate"],
        label="Q-Learning",
    )
    plt.plot(
        dqn_df["episode"],
        dqn_df["moving_success_rate"],
        label="DQN",
    )
    plt.xlabel("Episode")
    plt.ylabel("Moving Success Rate")
    plt.title("Random Levels - Q-Learning vs DQN Success Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(success_plot_path)
    plt.close()

    q_last_100 = q_df.tail(100)
    dqn_last_100 = dqn_df.tail(100)

    training_summary = pd.DataFrame(
        [
            {
                "agent": "Q-Learning",
                "final_100_avg_reward": q_last_100["total_reward"].mean(),
                "final_100_success_rate": q_last_100["success"].mean(),
                "final_100_avg_steps": q_last_100["steps"].mean(),
                "final_100_avg_solved_puzzles": q_last_100[
                    "solved_puzzles"
                ].mean(),
            },
            {
                "agent": "DQN",
                "final_100_avg_reward": dqn_last_100["total_reward"].mean(),
                "final_100_success_rate": dqn_last_100["success"].mean(),
                "final_100_avg_steps": dqn_last_100["steps"].mean(),
                "final_100_avg_solved_puzzles": dqn_last_100[
                    "solved_puzzles"
                ].mean(),
            },
        ]
    )

    training_summary_path = "results/random_levels_training_comparison_summary.csv"
    training_summary.to_csv(training_summary_path, index=False)

    return {
        "reward_plot_path": reward_plot_path,
        "success_plot_path": success_plot_path,
        "training_summary_path": training_summary_path,
        "training_summary": training_summary,
    }


def compare_evaluation_results(
    q_eval_path="results/q_random_levels_evaluation.csv",
    dqn_eval_path="results/dqn_random_levels_evaluation.csv",
):
    ensure_file_exists(
        q_eval_path,
        "python evaluate_random_levels.py --agent q --no-render",
    )

    ensure_file_exists(
        dqn_eval_path,
        "python evaluate_random_levels.py --agent dqn --no-render",
    )

    q_eval = pd.read_csv(q_eval_path)
    dqn_eval = pd.read_csv(dqn_eval_path)

    q_eval["agent"] = "Q-Learning"
    dqn_eval["agent"] = "DQN"

    combined = pd.concat([q_eval, dqn_eval], ignore_index=True)

    level_summary = (
        combined.groupby(["agent", "level_id", "level_name"])
        .agg(
            success_rate=("success", "mean"),
            avg_reward=("reward", "mean"),
            avg_steps=("steps", "mean"),
            avg_solved_puzzles=("solved_puzzles", "mean"),
        )
        .reset_index()
    )

    overall_summary = (
        combined.groupby("agent")
        .agg(
            total_episodes=("success", "count"),
            success_rate=("success", "mean"),
            avg_reward=("reward", "mean"),
            avg_steps=("steps", "mean"),
            avg_solved_puzzles=("solved_puzzles", "mean"),
        )
        .reset_index()
    )

    level_summary_path = "results/random_levels_evaluation_by_level_summary.csv"
    overall_summary_path = "results/random_levels_evaluation_overall_summary.csv"

    level_summary.to_csv(level_summary_path, index=False)
    overall_summary.to_csv(overall_summary_path, index=False)

    reward_by_level_plot_path = "results/random_levels_reward_by_level.png"

    pivot_reward = level_summary.pivot(
        index="level_name",
        columns="agent",
        values="avg_reward",
    )

    pivot_reward.plot(kind="bar", figsize=(10, 5))
    plt.xlabel("Level")
    plt.ylabel("Average Reward")
    plt.title("Random Levels - Average Reward by Level")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(reward_by_level_plot_path)
    plt.close()

    success_by_level_plot_path = "results/random_levels_success_by_level.png"

    pivot_success = level_summary.pivot(
        index="level_name",
        columns="agent",
        values="success_rate",
    )

    pivot_success.plot(kind="bar", figsize=(10, 5))
    plt.xlabel("Level")
    plt.ylabel("Success Rate")
    plt.title("Random Levels - Success Rate by Level")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(success_by_level_plot_path)
    plt.close()

    steps_by_level_plot_path = "results/random_levels_steps_by_level.png"

    pivot_steps = level_summary.pivot(
        index="level_name",
        columns="agent",
        values="avg_steps",
    )

    pivot_steps.plot(kind="bar", figsize=(10, 5))
    plt.xlabel("Level")
    plt.ylabel("Average Steps")
    plt.title("Random Levels - Average Steps by Level")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(steps_by_level_plot_path)
    plt.close()

    return {
        "level_summary_path": level_summary_path,
        "overall_summary_path": overall_summary_path,
        "reward_by_level_plot_path": reward_by_level_plot_path,
        "success_by_level_plot_path": success_by_level_plot_path,
        "steps_by_level_plot_path": steps_by_level_plot_path,
        "level_summary": level_summary,
        "overall_summary": overall_summary,
    }


def compare_random_levels():
    os.makedirs("results", exist_ok=True)

    print()
    print("=" * 70)
    print("Comparing Q-Learning vs DQN on Randomized Levels")
    print("=" * 70)

    training_results = compare_training_results()
    evaluation_results = compare_evaluation_results()

    print()
    print("Training Comparison Summary")
    print("-" * 70)
    print(training_results["training_summary"])

    print()
    print("Evaluation Overall Summary")
    print("-" * 70)
    print(evaluation_results["overall_summary"])

    print()
    print("Evaluation By Level Summary")
    print("-" * 70)
    print(evaluation_results["level_summary"])

    print()
    print("Generated files")
    print("-" * 70)
    print(training_results["reward_plot_path"])
    print(training_results["success_plot_path"])
    print(training_results["training_summary_path"])
    print(evaluation_results["level_summary_path"])
    print(evaluation_results["overall_summary_path"])
    print(evaluation_results["reward_by_level_plot_path"])
    print(evaluation_results["success_by_level_plot_path"])
    print(evaluation_results["steps_by_level_plot_path"])


if __name__ == "__main__":
    compare_random_levels()