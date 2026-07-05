import argparse
import os

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


def run_generated_visualization(
    generate_spec,
    model_path="models/cnn_dqn_generalize.pth",
    delay=350,
    record_gif=None,
    headless=False,
):
    """
    Watch the generalizing CNN agent solve a procedurally generated level it
    has never seen. `generate_spec` is "rows,cols,seed".

    With record_gif, every frame is captured and assembled into a GIF; with
    headless=True pygame renders offscreen (useful for recording).
    """
    import torch

    from agents.cnn_dqn_agent import CNNDQNAgent
    from environment.level_generator import LevelGenerator
    from environment.observation import build_observation

    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    try:
        rows, cols, seed = (int(part) for part in generate_spec.split(","))
    except ValueError:
        raise ValueError(
            f'--generate expects "rows,cols,seed", got: {generate_spec}'
        )

    level_data = LevelGenerator().generate(rows=rows, cols=cols, seed=seed)
    env = EscapeRoomEnv(level_data=level_data, max_steps=4 * rows * cols)
    env.reset()

    agent = CNNDQNAgent(
        action_space_size=env.action_space_size,
        device=torch.device("cpu"),
    )
    agent.load(model_path)

    print(f"Level: {env.level_name} | Agent: cnn | Model: {model_path}")

    visualizer = PygameVisualizer(env)
    frames = []

    def capture():
        if record_gif is not None:
            frames.append(visualizer.get_frame())

    try:
        observation = build_observation(env)

        visualizer.draw(action_text="START", reward=0, message="Initial state")
        capture()
        visualizer.wait(delay)

        for step in range(env.max_steps):
            action = agent.choose_action(observation, training=False)
            _, reward, done, info = env.step(action)
            observation = build_observation(env)

            visualizer.draw(
                action_text=info["action"],
                reward=reward,
                message=info["message"],
            )
            capture()
            visualizer.wait(delay)

            if done:
                if not record_gif:
                    visualizer.wait(2000)
                break

    finally:
        visualizer.close()

    if record_gif is not None and frames:
        import imageio

        os.makedirs(os.path.dirname(record_gif) or ".", exist_ok=True)
        # hold the final frame a little longer
        frames.extend([frames[-1]] * 6)
        imageio.mimsave(record_gif, frames, duration=max(delay, 100) / 1000, loop=0)
        print(f"GIF saved at: {record_gif} ({len(frames)} frames)")


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
    parser.add_argument(
        "--generate",
        default=None,
        metavar="ROWS,COLS,SEED",
        help=(
            'Generate a level on the fly (e.g. "10,10,1042") and watch the '
            "generalizing CNN agent solve it. Ignores --level/--agent."
        ),
    )
    parser.add_argument(
        "--record-gif",
        default=None,
        metavar="PATH",
        help="Record the run as a GIF (only with --generate).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render offscreen (SDL dummy driver), e.g. for GIF recording.",
    )

    args = parser.parse_args()

    if args.generate is not None:
        run_generated_visualization(
            generate_spec=args.generate,
            model_path=args.model or "models/cnn_dqn_generalize.pth",
            delay=args.delay if args.delay != 500 else 350,
            record_gif=args.record_gif,
            headless=args.headless,
        )
        return

    run_visualization(
        agent_type=args.agent,
        model_path=args.model,
        level=args.level,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
