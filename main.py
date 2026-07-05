from compare_agents import compare_agents
from evaluate import evaluate_agent
from evaluate_dqn import evaluate_dqn_agent
from evaluate_random_levels import evaluate_random_levels
from train import train_agent
from train_dqn import train_dqn_agent
from train_random_levels import train_random_levels


def main():
    while True:
        print()
        print("=" * 65)
        print("Escape Room ML Agent")
        print("=" * 65)
        print("1. Train Q-Learning agent - fixed level")
        print("2. Evaluate Q-Learning agent - fixed level")
        print("3. Train DQN agent - fixed level")
        print("4. Evaluate DQN agent - fixed level")
        print("5. Compare Q-Learning vs DQN - fixed level")
        print("6. Manual Play with Pygame")
        print("7. Visualize trained Q-Learning agent")
        print("8. Visualize trained DQN agent")
        print("9. Train Q-Learning on randomized levels")
        print("10. Evaluate Q-Learning on all levels")
        print("11. Train DQN on randomized levels")
        print("12. Evaluate DQN on all levels")
        print("13. Exit")
        print("=" * 65)

        choice = input("Choose option: ")

        if choice == "1":
            train_agent(
                episodes=6000,
                max_steps=150,
            )

        elif choice == "2":
            evaluate_agent(
                episodes=5,
                max_steps=150,
                render=True,
            )

        elif choice == "3":
            train_dqn_agent(
                episodes=8000,
                max_steps=150,
            )

        elif choice == "4":
            evaluate_dqn_agent(
                episodes=5,
                max_steps=150,
                render=True,
            )

        elif choice == "5":
            compare_agents()

        elif choice == "6":
            from manual_play import manual_play

            manual_play()

        elif choice == "7":
            from visualize import run_visualization

            run_visualization(agent_type="q")

        elif choice == "8":
            from visualize import run_visualization

            run_visualization(agent_type="dqn")

        elif choice == "9":
            train_random_levels(
                agent_type="q",
                episodes=9000,
                max_steps=150,
            )

        elif choice == "10":
            evaluate_random_levels(
                agent_type="q",
                episodes_per_level=3,
                max_steps=150,
                render=True,
            )

        elif choice == "11":
            train_random_levels(
                agent_type="dqn",
                episodes=12000,
                max_steps=150,
            )

        elif choice == "12":
            evaluate_random_levels(
                agent_type="dqn",
                episodes_per_level=3,
                max_steps=150,
                render=True,
            )

        elif choice == "13":
            print("Exiting...")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()