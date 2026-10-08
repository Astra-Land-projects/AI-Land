"""
Reinforcement Learning - Q-Learning Maze Solver
Unlike every other project so far, this one has NO labeled training data.
The agent learns purely through trial and error: it takes actions, gets
rewards or penalties, and gradually learns the best strategy - the same
core idea behind things like game-playing AI and robotics control.

The agent must navigate a grid maze from a start point to a goal,
avoiding obstacles, learning only from rewards (+10 for reaching the goal,
-10 for hitting an obstacle, -1 for each step to encourage efficiency).
"""

import numpy as np
import matplotlib.pyplot as plt

# Maze layout: 0 = open path, 1 = wall/obstacle, S = start, G = goal
MAZE = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
]

START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]
ACTION_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

N_ROWS = len(MAZE)
N_COLS = len(MAZE[0])


def is_valid(state):
    row, col = state
    if row < 0 or row >= N_ROWS or col < 0 or col >= N_COLS:
        return False
    if MAZE[row][col] == 1:
        return False
    return True


def get_reward(state):
    if state == GOAL:
        return 10
    return -1  # small penalty per step to encourage shorter paths


def take_action(state, action):
    move = ACTION_MOVES[action]
    new_state = (state[0] + move[0], state[1] + move[1])
    if not is_valid(new_state):
        return state, -10  # hitting a wall/boundary: stay in place, get penalized
    return new_state, get_reward(new_state)


def train_agent(episodes=2000, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.05):
    # Q-table: one row per state (grid cell), one column per action
    q_table = np.zeros((N_ROWS, N_COLS, len(ACTIONS)))

    rewards_per_episode = []

    for episode in range(episodes):
        state = START
        total_reward = 0
        steps = 0
        max_steps = 100

        while state != GOAL and steps < max_steps:
            # Epsilon-greedy: sometimes explore randomly, sometimes exploit known best action
            if np.random.rand() < exploration_rate:
                action_index = np.random.randint(len(ACTIONS))
            else:
                action_index = np.argmax(q_table[state[0], state[1]])

            action = ACTIONS[action_index]
            next_state, reward = take_action(state, action)

            # Q-learning update rule (Bellman equation)
            best_next_q = np.max(q_table[next_state[0], next_state[1]])
            current_q = q_table[state[0], state[1], action_index]
            q_table[state[0], state[1], action_index] = current_q + learning_rate * (
                reward + discount_factor * best_next_q - current_q
            )

            state = next_state
            total_reward += reward
            steps += 1

        exploration_rate = max(min_exploration, exploration_rate * exploration_decay)
        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode


def get_optimal_path(q_table, max_steps=50):
    state = START
    path = [state]

    for _ in range(max_steps):
        if state == GOAL:
            break
        action_index = np.argmax(q_table[state[0], state[1]])
        action = ACTIONS[action_index]
        state, _ = take_action(state, action)
        path.append(state)

    return path


def print_maze_with_path(path):
    print("\nOptimal path found by the agent:")
    display = [row.copy() for row in MAZE]
    for r, c in path:
        if (r, c) != START and (r, c) != GOAL:
            display[r][c] = "*"

    for r in range(N_ROWS):
        row_str = ""
        for c in range(N_COLS):
            if (r, c) == START:
                row_str += " S "
            elif (r, c) == GOAL:
                row_str += " G "
            elif display[r][c] == "*":
                row_str += " * "
            elif MAZE[r][c] == 1:
                row_str += " # "
            else:
                row_str += " . "
        print(row_str)


def plot_learning_curve(rewards, save_path="rl_learning_curve.png"):
    # Smooth the curve with a rolling average for readability
    window = 50
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel(f"Total reward (rolling avg over {window} episodes)")
    plt.title("Q-Learning Agent: Learning Progress Over Time")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nLearning curve saved to: {save_path}")
    plt.close()


def main():
    print("=" * 50)
    print("Q-Learning Maze Solver")
    print("=" * 50)
    print("\nMaze layout (S=start, G=goal, #=wall):")
    for r in range(N_ROWS):
        row_str = ""
        for c in range(N_COLS):
            if (r, c) == START:
                row_str += " S "
            elif (r, c) == GOAL:
                row_str += " G "
            elif MAZE[r][c] == 1:
                row_str += " # "
            else:
                row_str += " . "
        print(row_str)

    print("\nTraining the agent through trial and error (no labeled data)...")
    q_table, rewards = train_agent(episodes=2000)

    print(f"\nTraining complete. Average reward (last 100 episodes): {np.mean(rewards[-100:]):.1f}")

    path = get_optimal_path(q_table)
    print_maze_with_path(path)
    print(f"\nPath length: {len(path)} steps")

    plot_learning_curve(rewards)


if __name__ == "__main__":
    main()