import argparse
import json
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents.cnn_dqn_agent import CNNDQNAgent
from environment.escape_room_env import EscapeRoomEnv
from environment.level_generator import (
    TRAIN_SEEDS,
    VAL_SEEDS,
    LevelGenerator,
    generate_from_seed,
)
from environment.observation import build_observation

# Curriculum: each stage samples levels from the given ranges. The agent
# advances to the next stage once its rolling success rate on the current
# stage exceeds SUCCESS_THRESHOLD. The final stage matches the full
# distribution used for held-out evaluation.
CURRICULUM_STAGES = [
    {
        "name": "stage_1_small_easy",
        "rows": (6, 6),
        "cols": (6, 6),
        "wall_density": (0.08, 0.12),
        "num_puzzles": (1, 1),
    },
    {
        "name": "stage_2_small_medium",
        "rows": (6, 8),
        "cols": (6, 8),
        "wall_density": (0.10, 0.15),
        "num_puzzles": (1, 2),
    },
    {
        "name": "stage_3_medium",
        "rows": (8, 10),
        "cols": (8, 10),
        "wall_density": (0.12, 0.20),
        "num_puzzles": (2, 3),
    },
    {
        "name": "stage_4_large",
        "rows": (10, 12),
        "cols": (10, 12),
        "wall_density": (0.15, 0.25),
        "num_puzzles": (2, 3),
    },
    {
        "name": "stage_5_full_mix",
        "rows": (6, 12),
        "cols": (6, 12),
        "wall_density": (0.10, 0.25),
        "num_puzzles": (1, 3),
    },
]

SUCCESS_THRESHOLD = 0.80
ROLLING_WINDOW = 100
MIN_EPISODES_PER_STAGE = 100

# On a stage advance the task distribution changes, so exploration is
# partially re-enabled.
EPSILON_BOOST_ON_ADVANCE = 0.35


def sample_stage_level(stage, rng, generator):
    """Sample a random solvable level from the stage's parameter ranges,
    always using a training seed."""
    rows = rng.randint(*stage["rows"])
    cols = rng.randint(*stage["cols"])
    wall_density = rng.uniform(*stage["wall_density"])
    num_puzzles = rng.randint(*stage["num_puzzles"])
    seed = rng.choice(TRAIN_SEEDS)

    level_data = generator.generate(
        rows=rows,
        cols=cols,
        wall_density=wall_density,
        num_puzzles=num_puzzles,
        seed=seed,
    )

    return level_data, wall_density


def max_steps_for(rows, cols):
    return 3 * rows * cols


def run_validation(agent, validation_levels):
    """Greedy (epsilon=0) evaluation on the fixed held-out validation levels.
    Returns (success_rate, avg_steps). Used for model selection only; the
    test seed range stays untouched until the final evaluation."""
    successes = 0
    total_steps = 0

    for level_data in validation_levels:
        env = EscapeRoomEnv(
            level_data=level_data,
            max_steps=max_steps_for(level_data["rows"], level_data["cols"]),
        )
        env.reset()
        observation = build_observation(env)

        for _ in range(env.max_steps):
            action = agent.choose_action(observation, training=False)
            _, _, done, _ = env.step(action)
            observation = build_observation(env)
            if done:
                break

        if env.agent_position == env.exit_position and env.door_open:
            successes += 1
        total_steps += env.steps

    count = len(validation_levels)
    return successes / count, total_steps / count


def run_episode(env, agent, train_every=1):
    """Run one training episode; returns (total_reward, success, losses)."""
    env.reset()
    observation = build_observation(env)

    total_reward = 0
    success = False
    losses = []

    for step in range(env.max_steps):
        action = agent.choose_action(observation, training=True)
        _, reward, done, _ = env.step(action)
        next_observation = build_observation(env)

        agent.remember(observation, action, reward, next_observation, done)

        if step % train_every == 0:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        observation = next_observation
        total_reward += reward

        if done:
            if env.agent_position == env.exit_position and env.door_open:
                success = True
            break

    return total_reward, success, losses


def train_generalize_agent(
    episodes=30000,
    model_path="models/cnn_dqn_generalize.pth",
    stats_path="results/generalize_training_stats.csv",
    target_update_frequency=10,
    train_every=1,
    checkpoint_every=1000,
    rng_seed=0,
    learning_rate=0.0005,
    epsilon_decay=0.998,
    validate_every=500,
    validation_stats_path="results/generalize_validation_stats.csv",
    resume=False,
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Sidecar with the curriculum/run state that the agent checkpoint does
    # not carry, so an interrupted run can be resumed with --resume.
    run_state_path = model_path.replace(".pth", "_state.json")
    resuming = (
        resume and os.path.exists(model_path) and os.path.exists(run_state_path)
    )

    # Reproducibility: seed every RNG the run touches (level sampling uses
    # its own dedicated stream below; exploration uses the global `random`;
    # replay sampling uses `random`; network init uses torch). A resumed run
    # cannot replay the original streams mid-way, so it reseeds
    # deterministically off the resume point instead.
    if resuming:
        with open(run_state_path, "r", encoding="utf-8") as file:
            run_state = json.load(file)
        seed_base = rng_seed + run_state["episode"]
    else:
        run_state = None
        seed_base = rng_seed

    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)

    rng = random.Random(seed_base)
    generator = LevelGenerator()

    # Fixed held-out validation levels (disjoint seeds from both the training
    # and the test ranges), generated once so every validation pass sees the
    # exact same levels.
    validation_levels = [generate_from_seed(seed) for seed in VAL_SEEDS]

    agent = CNNDQNAgent(
        action_space_size=len(EscapeRoomEnv.ACTIONS),
        learning_rate=learning_rate,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        min_epsilon=0.05,
        batch_size=64,
    )

    print(f"Using device: {agent.device}")

    stage_index = 0
    stage_episode_count = 0
    stage_successes = deque(maxlen=ROLLING_WINDOW)
    start_episode = 1

    training_data = []
    validation_data = []
    best_validation_success = -1.0
    best_validation_steps = float("inf")

    if resuming:
        agent.load(model_path)
        # load() puts the networks in eval mode for inference use; training
        # continues, so restore train mode on the policy network.
        agent.policy_network.train()

        stage_index = run_state["stage_index"]
        stage_episode_count = run_state["stage_episode_count"]
        start_episode = run_state["episode"] + 1
        best_validation_success = run_state["best_validation_success"]
        best_validation_steps = run_state["best_validation_steps"]

        if os.path.exists(stats_path):
            training_data = pd.read_csv(stats_path).to_dict("records")
        if os.path.exists(validation_stats_path):
            validation_data = pd.read_csv(validation_stats_path).to_dict(
                "records"
            )

        # rebuild the rolling success window from the saved episode stats
        current_stage_rows = [
            row
            for row in training_data
            if row["stage"] == stage_index + 1
        ]
        for row in current_stage_rows[-ROLLING_WINDOW:]:
            stage_successes.append(int(row["success"]))

        print(
            f"Resuming from episode {run_state['episode']} "
            f"(stage {stage_index + 1}/{len(CURRICULUM_STAGES)}, "
            f"epsilon {agent.epsilon:.3f}, "
            f"best validation {best_validation_success:.2%})"
        )
    elif resume:
        print(
            "Resume requested but no checkpoint/state found; "
            "starting a fresh run."
        )

    def save_run_state(episode):
        with open(run_state_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "episode": episode,
                    "stage_index": stage_index,
                    "stage_episode_count": stage_episode_count,
                    "best_validation_success": best_validation_success,
                    "best_validation_steps": best_validation_steps,
                },
                file,
                indent=2,
            )

    for episode in range(start_episode, episodes + 1):
        stage = CURRICULUM_STAGES[stage_index]

        level_data, wall_density = sample_stage_level(stage, rng, generator)
        env = EscapeRoomEnv(
            level_data=level_data,
            max_steps=max_steps_for(level_data["rows"], level_data["cols"]),
        )

        total_reward, success, losses = run_episode(env, agent, train_every)

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        stage_episode_count += 1
        stage_successes.append(int(success))
        rolling_success = sum(stage_successes) / len(stage_successes)

        average_loss = sum(losses) / len(losses) if losses else 0

        training_data.append(
            {
                "episode": episode,
                "stage": stage_index + 1,
                "stage_name": stage["name"],
                "rows": env.rows,
                "cols": env.cols,
                "wall_density": round(wall_density, 4),
                "num_puzzles": len(env.puzzles),
                "level_seed": level_data["level_id"],
                "total_reward": total_reward,
                "steps": env.steps,
                "success": int(success),
                "rolling_success": rolling_success,
                "collected_key": int(env.has_key),
                "solved_puzzles": sum(1 for p in env.puzzles if p.solved),
                "opened_door": int(env.door_open),
                "epsilon": agent.epsilon,
                "loss": average_loss,
            }
        )

        # --- curriculum advancement ---
        if (
            stage_index < len(CURRICULUM_STAGES) - 1
            and stage_episode_count >= MIN_EPISODES_PER_STAGE
            and len(stage_successes) == ROLLING_WINDOW
            and rolling_success >= SUCCESS_THRESHOLD
        ):
            print(
                f"Episode {episode}: stage '{stage['name']}' passed "
                f"({rolling_success:.2%}) -> advancing to "
                f"'{CURRICULUM_STAGES[stage_index + 1]['name']}'"
            )
            stage_index += 1
            stage_episode_count = 0
            stage_successes.clear()
            agent.epsilon = max(agent.epsilon, EPSILON_BOOST_ON_ADVANCE)
            agent.save(model_path.replace(".pth", f"_stage{stage_index}.pth"))

        # --- periodic held-out validation for model selection ---
        if episode % validate_every == 0:
            val_success, val_steps = run_validation(agent, validation_levels)
            validation_data.append(
                {
                    "episode": episode,
                    "stage": stage_index + 1,
                    "stage_name": stage["name"],
                    "val_success_rate": val_success,
                    "val_avg_steps": val_steps,
                    "epsilon": agent.epsilon,
                }
            )
            pd.DataFrame(validation_data).to_csv(
                validation_stats_path, index=False
            )

            is_best = val_success > best_validation_success or (
                val_success == best_validation_success
                and val_steps < best_validation_steps
            )
            if is_best:
                best_validation_success = val_success
                best_validation_steps = val_steps
                agent.save(model_path.replace(".pth", "_best.pth"))

            print(
                f"Episode {episode}: validation on {len(validation_levels)} "
                f"held-out levels -> success {val_success:.2%}, "
                f"avg steps {val_steps:.1f}"
                f"{' (new best, checkpoint saved)' if is_best else ''}",
                flush=True,
            )

        if episode % checkpoint_every == 0:
            agent.save(model_path)
            pd.DataFrame(training_data).to_csv(stats_path, index=False)
            save_run_state(episode)

        if episode % 100 == 0:
            recent = training_data[-100:]
            avg_reward = sum(item["total_reward"] for item in recent) / len(recent)
            avg_steps = sum(item["steps"] for item in recent) / len(recent)
            avg_loss = sum(item["loss"] for item in recent) / len(recent)
            key_rate = sum(item["collected_key"] for item in recent) / len(recent)
            door_rate = sum(item["opened_door"] for item in recent) / len(recent)

            print(
                f"Episode {episode}/{episodes} | "
                f"Stage {stage_index + 1}/{len(CURRICULUM_STAGES)} "
                f"({stage['name']}) | "
                f"Rolling Success: {rolling_success:.2%} | "
                f"Key: {key_rate:.0%} | Door: {door_rate:.0%} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Steps: {avg_steps:.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f}",
                flush=True,
            )

    agent.save(model_path)
    save_run_state(episodes)

    df = pd.DataFrame(training_data)
    df.to_csv(stats_path, index=False)

    validation_df = pd.DataFrame(validation_data)
    if not validation_df.empty:
        validation_df.to_csv(validation_stats_path, index=False)

    plot_training_curves(df, validation_df=validation_df)

    print()
    print("Generalization training completed.")
    print(f"Final stage reached: {stage_index + 1}/{len(CURRICULUM_STAGES)}")
    print(
        f"Best validation success: {best_validation_success:.2%} "
        f"(checkpoint: {model_path.replace('.pth', '_best.pth')})"
    )
    print(f"Model saved at: {model_path}")
    print(f"Stats saved at: {stats_path}")
    print(f"Validation stats saved at: {validation_stats_path}")

    return df


def plot_training_curves(
    df,
    validation_df=None,
    reward_plot_path="results/generalize_rewards.png",
    success_plot_path="results/generalize_success_rate.png",
    loss_plot_path="results/generalize_loss.png",
):
    stage_starts = df.groupby("stage")["episode"].min().tolist()[1:]

    def add_stage_lines():
        for start in stage_starts:
            plt.axvline(start, color="gray", linestyle="--", alpha=0.5)

    window = min(ROLLING_WINDOW, max(1, len(df) // 10))

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["total_reward"], alpha=0.25, label="Reward")
    plt.plot(
        df["episode"],
        df["total_reward"].rolling(window, min_periods=1).mean(),
        label=f"Moving Average ({window} ep)",
    )
    add_stage_lines()
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Curriculum Training Rewards - Generalizing Agent")
    plt.legend()
    plt.grid(True)
    plt.savefig(reward_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        df["episode"],
        df["success"].rolling(window, min_periods=1).mean(),
        label=f"Success Rate ({window} ep window)",
    )
    if validation_df is not None and not validation_df.empty:
        plt.plot(
            validation_df["episode"],
            validation_df["val_success_rate"],
            marker="o",
            markersize=3,
            color="darkorange",
            label="Held-out validation success",
        )
    add_stage_lines()
    plt.axhline(SUCCESS_THRESHOLD, color="green", linestyle=":", alpha=0.7,
                label=f"Stage threshold ({SUCCESS_THRESHOLD:.0%})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Curriculum Success Rate - Generalizing Agent")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(success_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["loss"], alpha=0.25, label="Loss")
    plt.plot(
        df["episode"],
        df["loss"].rolling(window, min_periods=1).mean(),
        label=f"Moving Average ({window} ep)",
    )
    add_stage_lines()
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Curriculum Training Loss - Generalizing Agent")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()

    print(f"Reward plot saved at: {reward_plot_path}")
    print(f"Success plot saved at: {success_plot_path}")
    print(f"Loss plot saved at: {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum training of the generalizing CNN DQN agent."
    )
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--model-path", default="models/cnn_dqn_generalize.pth")
    parser.add_argument(
        "--stats-path", default="results/generalize_training_stats.csv"
    )
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument(
        "--validation-stats-path",
        default="results/generalize_validation_stats.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint and run-state sidecar at "
        "--model-path, if they exist.",
    )

    args = parser.parse_args()

    train_generalize_agent(
        episodes=args.episodes,
        model_path=args.model_path,
        stats_path=args.stats_path,
        train_every=args.train_every,
        target_update_frequency=args.target_update,
        checkpoint_every=args.checkpoint_every,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        rng_seed=args.seed,
        validate_every=args.validate_every,
        validation_stats_path=args.validation_stats_path,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
