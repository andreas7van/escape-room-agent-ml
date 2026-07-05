from environment.escape_room_env import EscapeRoomEnv
from ui.pygame_visualizer import PygameVisualizer


def manual_play(max_steps=150):
    import pygame

    env = EscapeRoomEnv(max_steps=max_steps)
    env.reset()

    visualizer = PygameVisualizer(env)

    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
        pygame.K_k: 4,
        pygame.K_s: 5,
        pygame.K_o: 6,
    }

    last_action = "START"
    last_reward = 0
    last_message = (
        "Use arrows to move, K to pick key, S to solve puzzle, O to open door."
    )

    try:
        while True:
            visualizer.draw(
                action_text=last_action,
                reward=last_reward,
                message=last_message,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

                    if event.key in key_to_action and not env.done:
                        action = key_to_action[event.key]
                        _, reward, done, info = env.step(action)

                        last_action = info["action"]
                        last_reward = reward
                        last_message = info["message"]

                        if done:
                            last_message = (
                                info["message"]
                                + " Press ESC or close the window."
                            )

    finally:
        visualizer.close()


if __name__ == "__main__":
    manual_play()
