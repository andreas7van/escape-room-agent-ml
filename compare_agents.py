import os

import matplotlib.pyplot as plt
import pandas as pd


def compare_agents(
    q_learning_stats_path="results/q_fixed_training_stats.csv",
    dqn_stats_path="results/dqn_fixed_training_stats.csv",
    output_path="results/q_learning_vs_dqn_reward.png",
):
    if not os.path.exists(q_learning_stats_path):
        raise FileNotFoundError(
            f"Q-Learning stats file not found: {q_learning_stats_path}"
        )

    if not os.path.exists(dqn_stats_path):
        raise FileNotFoundError(
            f"DQN stats file not found: {dqn_stats_path}"
        )

    q_df = pd.read_csv(q_learning_stats_path)
    dqn_df = pd.read_csv(dqn_stats_path)

    if "moving_avg_reward" not in q_df.columns:
        q_df["moving_avg_reward"] = q_df["total_reward"].rolling(
            window=100,
            min_periods=1,
        ).mean()

    if "moving_success_rate" not in q_df.columns:
        q_df["moving_success_rate"] = q_df["success"].rolling(
            window=100,
            min_periods=1,
        ).mean()

    if "moving_avg_reward" not in dqn_df.columns:
        dqn_df["moving_avg_reward"] = dqn_df["total_reward"].rolling(
            window=100,
            min_periods=1,
        ).mean()

    if "moving_success_rate" not in dqn_df.columns:
        dqn_df["moving_success_rate"] = dqn_df["success"].rolling(
            window=100,
            min_periods=1,
        ).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(
        q_df["episode"],
        q_df["moving_avg_reward"],
        label="Q-Learning Reward",
    )
    plt.plot(
        dqn_df["episode"],
        dqn_df["moving_avg_reward"],
        label="DQN Reward",
    )
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Q-Learning vs DQN - Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    success_output_path = "results/q_learning_vs_dqn_success.png"

    plt.figure(figsize=(10, 5))
    plt.plot(
        q_df["episode"],
        q_df["moving_success_rate"],
        label="Q-Learning Success Rate",
    )
    plt.plot(
        dqn_df["episode"],
        dqn_df["moving_success_rate"],
        label="DQN Success Rate",
    )
    plt.xlabel("Episode")
    plt.ylabel("Moving Success Rate")
    plt.title("Q-Learning vs DQN - Success Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(success_output_path)
    plt.close()

    q_final_success_rate = q_df["success"].tail(100).mean()
    dqn_final_success_rate = dqn_df["success"].tail(100).mean()

    q_final_reward = q_df["total_reward"].tail(100).mean()
    dqn_final_reward = dqn_df["total_reward"].tail(100).mean()

    q_final_steps = q_df["steps"].tail(100).mean()
    dqn_final_steps = dqn_df["steps"].tail(100).mean()

    summary_path = "results/comparison_summary.csv"

    summary_df = pd.DataFrame(
        [
            {
                "agent": "Q-Learning",
                "final_100_avg_reward": q_final_reward,
                "final_100_success_rate": q_final_success_rate,
                "final_100_avg_steps": q_final_steps,
            },
            {
                "agent": "DQN",
                "final_100_avg_reward": dqn_final_reward,
                "final_100_success_rate": dqn_final_success_rate,
                "final_100_avg_steps": dqn_final_steps,
            },
        ]
    )

    summary_df.to_csv(summary_path, index=False)

    print("Comparison completed.")
    print(f"Reward comparison plot saved at: {output_path}")
    print(f"Success comparison plot saved at: {success_output_path}")
    print(f"Summary saved at: {summary_path}")
    print()
    print(summary_df)


if __name__ == "__main__":
    compare_agents()
