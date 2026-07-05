from agents.dqn_agent import DQNAgent
from environment.escape_room_env import EscapeRoomEnv


def evaluate_dqn_agent(
    episodes=5,
    max_steps=150,
    model_path="models/dqn_fixed.pth",
    render=True,
):
    env = EscapeRoomEnv(max_steps=max_steps)

    initial_state = env.reset()
    state_size = len(initial_state)

    agent = DQNAgent(
        state_size=state_size,
        action_space_size=env.action_space_size,
        grid_rows=env.rows,
        grid_cols=env.cols,
    )

    agent.load(model_path)

    total_successes = 0
    total_rewards = 0
    total_steps = 0

    for episode in range(1, episodes + 1):
        print()
        print("=" * 50)
        print(f"DQN Evaluation Episode {episode}")
        print("=" * 50)

        state = env.reset()
        episode_reward = 0

        if render:
            env.render()

        for step in range(max_steps):
            action = agent.choose_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            state = next_state

            print(
                f"Step {step + 1:03d} | "
                f"Action: {info['action']:<13} | "
                f"Reward: {reward:<4} | "
                f"Message: {info['message']}"
            )

            if render:
                env.render()

            if done:
                break

        success = env.agent_position == env.exit_position and env.door_open
        solved_puzzles = sum(1 for puzzle in env.puzzles if puzzle.solved)

        if success:
            total_successes += 1

        total_rewards += episode_reward
        total_steps += env.steps

        print()
        print("Episode Summary")
        print("-" * 50)
        print(f"Episode reward: {episode_reward}")
        print(f"Steps: {env.steps}")
        print(f"Success: {success}")
        print(f"Collected key: {env.has_key}")
        print(f"Solved puzzles: {solved_puzzles}/{len(env.puzzles)}")
        print(f"Door open: {env.door_open}")

    print()
    print("=" * 50)
    print("Final DQN Evaluation Results")
    print("=" * 50)
    print(f"Episodes: {episodes}")
    print(f"Successes: {total_successes}")
    print(f"Success rate: {total_successes / episodes:.2%}")
    print(f"Average reward: {total_rewards / episodes:.2f}")
    print(f"Average steps: {total_steps / episodes:.2f}")


if __name__ == "__main__":
    evaluate_dqn_agent()
