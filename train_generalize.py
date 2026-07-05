import argparse
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd

from agents.cnn_dqn_agent import CNNDQNAgent
from environment.escape_room_env import EscapeRoomEnv
from environment.level_generator import TRAIN_SEEDS, LevelGenerator
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
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    rng = random.Random(rng_seed)
    generator = LevelGenerator()

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

    training_data = []
    best_final_stage_success = 0.0

    for episode in range(1, episodes + 1):
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

        # track the best rolling success on the final stage
        if (
            stage_index == len(CURRICULUM_STAGES) - 1
            and len(stage_successes) == ROLLING_WINDOW
            and rolling_success > best_final_stage_success
        ):
            best_final_stage_success = rolling_success
            agent.save(model_path.replace(".pth", "_best.pth"))

        if episode % checkpoint_every == 0:
            agent.save(model_path)
            pd.DataFrame(training_data).to_csv(stats_path, index=False)

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

    df = pd.DataFrame(training_data)
    df.to_csv(stats_path, index=False)

    plot_training_curves(df)

    print()
    print("Generalization training completed.")
    print(f"Final stage reached: {stage_index + 1}/{len(CURRICULUM_STAGES)}")
    print(f"Model saved at: {model_path}")
    print(f"Stats saved at: {stats_path}")

    return df


def plot_training_curves(
    df,
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
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train_generalize_agent(
        episodes=args.episodes,
        model_path=args.model_path,
        stats_path=args.stats_path,
        train_every=args.train_every,
        target_update_frequency=args.target_update,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        rng_seed=args.seed,
    )


if __name__ == "__main__":
    main()
