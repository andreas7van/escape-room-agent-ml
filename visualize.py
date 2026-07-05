import argparse

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.escape_room_env import EscapeRoomEnv
from ui.pygame_visualizer import PygameVisualizer

DEFAULT_MODELS = {
    ("q", "fixed"): "models/q_table_fixed.pkl",
    ("q", "multi"): "models/q_table_random_levels.pkl",
    ("dqn", "fixed"): "models/dqn_fixed.pth",
    ("dqn", "multi"): "models/dqn_random_levels.pth",
}


def run_visualization(agent_type="q", model_path=None, level=1, max_steps=150, delay=500):
    """
    Watch a trained agent play a level.

    The correct model is selected automatically: the fixed-level model is
    only valid on level 1, so for any other level the multi-level model
    (trained on randomized levels) is loaded instead.
    """
    level_path = f"config/levels/level_{level}.json"

    # The multi-level models were trained with the level id in the state,
    # so environment and model must agree on the state layout.
    use_multi = level != 1 or (model_path is not None and "random_levels" in model_path)

    if model_path is None:
        model_path = DEFAULT_MODELS[(agent_type, "multi" if use_multi else "fixed")]
    elif use_multi and "random_levels" not in model_path:
        print(
            f"Warning: {model_path} was trained on the fixed level only "
            f"and is not expected to solve level {level}."
        )

    env = EscapeRoomEnv(
        max_steps=max_steps,
        level_path=level_path,
        include_level_in_state=use_multi,
    )
    state = env.reset()

    if agent_type == "q":
        agent = QLearningAgent(action_space_size=env.action_space_size)
        agent.load(model_path)

    elif agent_type == "dqn":
        agent = DQNAgent(
            state_size=len(state),
            action_space_size=env.action_space_size,
            grid_rows=env.rows,
            grid_cols=env.cols,
        )
        agent.load(model_path)

    else:
        raise ValueError("agent_type must be either 'q' or 'dqn'.")

    print(f"Level: {env.level_name} | Agent: {agent_type} | Model: {model_path}")

    visualizer = PygameVisualizer(env)

    try:
        visualizer.draw(action_text="START", reward=0, message="Initial state")
        visualizer.wait(delay)

        for step in range(max_steps):
            action = agent.choose_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            visualizer.draw(
                action_text=info["action"],
                reward=reward,
                message=info["message"],
            )

            visualizer.wait(delay)

            state = next_state

            if done:
                visualizer.wait(2000)
                break

    finally:
        visualizer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["q", "dqn"],
        default="q",
        help="Choose trained agent: q or dqn",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Level to play. Levels 2 and 3 automatically use the multi-level model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained model. If omitted, the right model for the level is used.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=500,
        help="Delay between agent actions in milliseconds.",
    )

    args = parser.parse_args()

    run_visualization(
        agent_type=args.agent,
        model_path=args.model,
        level=args.level,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
