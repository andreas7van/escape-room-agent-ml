import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from agents.cnn_dqn_agent import CNNDQNAgent
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.escape_room_env import EscapeRoomEnv
from environment.level_generator import TEST_SEEDS, generate_from_seed
from environment.observation import build_observation


def max_steps_for(rows, cols):
    return 3 * rows * cols


def run_cnn_episode(env, agent):
    env.reset()
    observation = build_observation(env)

    total_reward = 0

    for _ in range(env.max_steps):
        action = agent.choose_action(observation, training=False)
        _, reward, done, _ = env.step(action)
        observation = build_observation(env)
        total_reward += reward

        if done:
            break

    success = env.agent_position == env.exit_position and env.door_open
    return success, env.steps, total_reward


def run_tuple_episode(env, agent):
    state = env.reset()

    total_reward = 0

    for _ in range(env.max_steps):
        action = agent.choose_action(state, training=False)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    success = env.agent_position == env.exit_position and env.door_open
    return success, env.steps, total_reward


def evaluate_agent(agent_name, seeds):
    """
    Evaluate one agent on the held-out levels. Returns one record per level.

    - 'cnn' is the generalizing agent (spatial observations, any level).
    - 'q_multi' is the old tabular agent trained on the 3 fixed levels; every
      generated level state is unseen, so its Q-values are all zero there.
    - 'dqn_multi' is the old multi-level DQN; its input layer only fits
      levels with exactly 3 puzzles, so it is evaluated on that subset.
    """
    records = []

    cnn_agent = None
    q_agent = None
    dqn_agent = None

    if agent_name == "cnn":
        cnn_agent = CNNDQNAgent(
            action_space_size=len(EscapeRoomEnv.ACTIONS),
            device=torch.device("cpu"),
        )
        cnn_agent.load(MODEL_PATHS["cnn"])

    elif agent_name == "q_multi":
        q_agent = QLearningAgent(action_space_size=len(EscapeRoomEnv.ACTIONS))
        q_agent.load(MODEL_PATHS["q_multi"])

    elif agent_name == "dqn_multi":
        dqn_agent = DQNAgent(
            state_size=8,
            action_space_size=len(EscapeRoomEnv.ACTIONS),
        )
        dqn_agent.load(MODEL_PATHS["dqn_multi"])

    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    for seed in seeds:
        level_data = generate_from_seed(seed)
        rows = level_data["rows"]
        cols = level_data["cols"]
        num_puzzles = len(level_data["puzzles"])

        if agent_name == "dqn_multi" and num_puzzles != 3:
            continue

        env = EscapeRoomEnv(
            level_data=level_data,
            max_steps=max_steps_for(rows, cols),
            include_level_in_state=(agent_name == "dqn_multi"),
        )

        if agent_name == "cnn":
            success, steps, total_reward = run_cnn_episode(env, cnn_agent)
        elif agent_name == "q_multi":
            success, steps, total_reward = run_tuple_episode(env, q_agent)
        else:
            success, steps, total_reward = run_tuple_episode(env, dqn_agent)

        records.append(
            {
                "agent": agent_name,
                "seed": seed,
                "rows": rows,
                "cols": cols,
                "grid_size": max(rows, cols),
                "num_puzzles": num_puzzles,
                "success": int(success),
                "steps": steps,
                "total_reward": total_reward,
            }
        )

    return records


MODEL_PATHS = {
    "cnn": "models/cnn_dqn_generalize.pth",
    "q_multi": "models/q_table_random_levels.pkl",
    "dqn_multi": "models/dqn_random_levels.pth",
}


def summarize(df):
    summary = (
        df.groupby("agent")
        .agg(
            levels=("seed", "count"),
            success_rate=("success", "mean"),
            avg_steps=("steps", "mean"),
            avg_reward=("total_reward", "mean"),
        )
        .reset_index()
    )
    return summary


def summarize_by_size(df):
    cnn = df[df["agent"] == "cnn"]
    by_size = (
        cnn.groupby("grid_size")
        .agg(
            levels=("seed", "count"),
            success_rate=("success", "mean"),
            avg_steps=("steps", "mean"),
        )
        .reset_index()
    )
    return by_size


def plot_results(df, by_size, chart_path="results/generalize_eval_by_size.png"):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        by_size["grid_size"].astype(str),
        by_size["success_rate"] * 100,
        color="#4098ff",
    )
    axes[0].axhline(80, color="green", linestyle=":", label="80% target")
    axes[0].set_xlabel("Grid size (max of rows, cols)")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("Generalizing agent on unseen levels, by size")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.4)

    summary = summarize(df)
    labels = {
        "cnn": "CNN DQN\n(generalizing)",
        "q_multi": "Q-Learning\n(3 fixed levels)",
        "dqn_multi": "DQN\n(3 fixed levels)",
    }
    order = [name for name in ["cnn", "q_multi", "dqn_multi"] if name in set(summary["agent"])]
    values = [
        float(summary.loc[summary["agent"] == name, "success_rate"].iloc[0]) * 100
        for name in order
    ]
    colors = ["#4098ff" if name == "cnn" else "#a0a6b4" for name in order]

    axes[1].bar([labels[name] for name in order], values, color=colors)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_title("Unseen levels: generalizing agent vs fixed-level baselines")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, axis="y", alpha=0.4)

    figure.tight_layout()
    figure.savefig(chart_path)
    plt.close(figure)

    print(f"Chart saved at: {chart_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agents on held-out procedurally generated levels."
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATHS["cnn"],
        help="Checkpoint of the generalizing CNN agent.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only evaluate the CNN agent.",
    )
    parser.add_argument(
        "--output-csv", default="results/generalize_evaluation.csv"
    )
    parser.add_argument(
        "--summary-csv", default="results/generalize_evaluation_summary.csv"
    )

    args = parser.parse_args()
    MODEL_PATHS["cnn"] = args.model_path

    os.makedirs("results", exist_ok=True)

    seeds = list(TEST_SEEDS)
    print(f"Evaluating on {len(seeds)} held-out levels (seeds {seeds[0]}-{seeds[-1]})")

    records = evaluate_agent("cnn", seeds)

    if not args.skip_baselines:
        records += evaluate_agent("q_multi", seeds)
        records += evaluate_agent("dqn_multi", seeds)

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    summary = summarize(df)
    by_size = summarize_by_size(df)

    summary.to_csv(args.summary_csv, index=False)
    by_size.to_csv(
        args.summary_csv.replace(".csv", "_by_size.csv"), index=False
    )

    print()
    print("=" * 60)
    print("Overall results on held-out unseen levels")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()
    print("Generalizing agent by grid size")
    print(by_size.to_string(index=False))

    plot_results(df, by_size)

    print(f"Per-level results saved at: {args.output_csv}")
    print(f"Summary saved at: {args.summary_csv}")


if __name__ == "__main__":
    main()
