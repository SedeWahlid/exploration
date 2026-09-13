"""
Grid World Q-Learning Agent
===========================

A from-scratch, model-free Q-Learning agent that learns to navigate a
10x10 Grid World maze from a Start cell to a Goal cell while avoiding
walls and traps.

Everything here is built from first principles using only:
    numpy, random, matplotlib, time

No external RL libraries (gym, stable-baselines, ...) are used so that
every part of the algorithm is visible and easy to follow.

Core idea of Q-Learning
-----------------------
We keep a table Q[state][action] that estimates "how good" it is to take
`action` from `state`. We improve those estimates using the update rule:

    Q(s,a) <- Q(s,a) + alpha * [ R + gamma * max_a' Q(s',a') - Q(s,a) ]

    alpha  (learning rate)    : how much we trust each new experience
    gamma  (discount factor)  : how much we value future reward
    R                         : reward received after taking action 'a'
    max_a' Q(s',a')           : best estimated value of the next state s'
"""

import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
# We label the four moves with integers 0..3 so they can index the Q-table.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# maximum actions per game
MAX_STEPS = 200

# our gamma 
GAMMA = 0.9

# our alpha
ALPHA = 0.1

# How each action changes (row, col). Row 0 is the top of the grid.
ACTION_DELTAS = {
    UP:    (-1, 0),
    DOWN:  (1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, 1),
}

# Arrows used when we print the learned policy.
ACTION_ARROWS = {UP: "^", DOWN: "v", LEFT: "<", RIGHT: ">"}


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------
class GridWorld:
    """A simple deterministic 10x10 maze.

    Rewards:
        +10  reaching the Goal      (episode ends, success)
        -10  stepping on a Trap     (episode ends, failure)
         -1  bumping a Wall/border  (agent does NOT move)
        -0.04  every normal step      (small cost -> encourages short paths)
    """

    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # Fixed walls: cells the agent cannot enter (it stays put if it tries).
        self.walls = {
            (0, 4), (1, 1), (1, 2), (1, 4), (2, 4), (2, 6), (2, 7),
            (3, 1), (3, 2), (4, 4), (4, 5), (4, 6), (4, 8),
            (5, 2), (6, 2), (6, 4), (6, 6), (6, 7), (6, 8),
            (7, 4), (8, 1), (8, 6), (3, 7)}

        # Fixed traps: stepping here ends the episode with a big penalty.
        self.traps = {(2, 2), (4, 1), (6, 0), (3, 5), (8, 4)}

        # Reward constants.
        self.goal_reward = 10.0
        self.trap_reward = -10.0
        self.wall_reward = -1.0
        self.step_cost = -0.2
        self.agent_pos = self.start

    # -- helpers -----------------------------------------------------------
    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def _is_blocked(self, pos):
        """A cell is blocked if it is off-grid or a wall."""
        return (not self._in_bounds(pos)) or (pos in self.walls)

    # -- gym-like API ------------------------------------------------------
    def reset(self):
        """Put the agent back at the start and return the start state."""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        """Apply `action` and return (next_state, reward, done).

        Deterministic dynamics: walls/borders block movement.
        """
        dr, dc = ACTION_DELTAS[action]
        cur = self.agent_pos
        proposed = (cur[0] + dr, cur[1] + dc)

        # Hitting a wall or the border: stay put and take the wall penalty.
        if self._is_blocked(proposed):
            self.agent_pos = cur
            return cur, self.wall_reward, False

        # Otherwise the move succeeds.
        self.agent_pos = proposed

        if proposed == self.goal:
            return proposed, self.goal_reward, True
        if proposed in self.traps:
            return proposed, self.trap_reward, True

        # A normal step pays the small living cost.
        return proposed, self.step_cost, False


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------
class QAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    def __init__(self, grid_size, n_actions=4,
                 alpha=ALPHA, gamma=GAMMA,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        # Q-table: one value per (row, col, action). Start optimistic-free at 0.
        self.q_table = np.zeros((grid_size, grid_size, n_actions))

        self.alpha = alpha              # learning rate
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # current exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def choose_action(self, state):
        """Epsilon-greedy action selection.

        EXPLORATION vs EXPLOITATION
        ---------------------------
        With probability `epsilon` we EXPLORE (pick a random action) so the
        agent discovers new parts of the maze. Otherwise we EXPLOIT current
        knowledge by picking the action with the highest Q-value.

        Because epsilon starts near 1.0 and decays toward ~0.01, early
        episodes are almost pure exploration, while later episodes become
        almost pure exploitation of what has been learned.
        """
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)            # explore
        r, c = state
        # Exploit: argmax over actions. (np.argmax breaks ties toward the
        # lowest index, which is fine for this demo.)
        return int(np.argmax(self.q_table[r, c]))

    def update(self, state, action, reward, next_state, done):
        """Apply the Q-Learning update rule for one transition."""
        r, c = state
        nr, nc = next_state

        # Best achievable value from the next state. If the episode ended,
        # there is no future, so that term is 0.
        best_next = 0.0 if done else np.max(self.q_table[nr, nc])

        td_target = reward + self.gamma * best_next      # what we now believe
        td_error = td_target - self.q_table[r, c, action]
        self.q_table[r, c, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Gradually shift from exploration to exploitation after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(env, agent, episodes=1000, max_steps=MAX_STEPS, verbose=True,
          record_episodes=None):
    """Run Q-Learning for a number of episodes.

    Returns:
        rewards_per_episode : list with the total reward of each episode
                              (used for the learning-curve plot).
        recorded            : list of (episode_number, trajectory, outcome)
                              tuples for the episodes whose 1-based number is
                              in `record_episodes`. These let us later REPLAY
                              the agent's actual moves and watch it improve.
    """
    record_set = set(record_episodes or [])
    rewards_per_episode = []
    recorded = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        ep_num = episode + 1

        # Only build a trajectory for episodes we were asked to record.
        recording = ep_num in record_set
        trajectory = [state] if recording else None
        outcome = "max-steps"

        # max_steps prevents an unlucky/early policy from looping forever.
        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            if recording:
                trajectory.append(state)
            if done:
                outcome = "GOAL" if state == env.goal else "TRAP"
                break

        # One full episode is over: anneal exploration a little.
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        if recording:
            recorded.append((ep_num, trajectory, outcome))

        if verbose and ep_num % 100 == 0:
            recent_avg = np.mean(rewards_per_episode[-100:])
            print(f"Episode {ep_num:>4} | "
                  f"epsilon = {agent.epsilon:0.3f} | "
                  f"avg reward (last 100) = {recent_avg:6.2f}")

    return rewards_per_episode, recorded


# ---------------------------------------------------------------------------
# Visualisation & Output
# ---------------------------------------------------------------------------
def print_policy(env, agent):
    """Print the greedy policy (best action per cell) as a grid of arrows."""
    print("\nLearned Policy")
    print("  S=Start  G=Goal  #=Wall  T=Trap  arrows=best move")
    print("  +" + "---" * env.size + "+")
    for r in range(env.size):
        row_cells = []
        for c in range(env.size):
            cell = (r, c)
            if cell == env.start:
                row_cells.append(" S ")
            elif cell == env.goal:
                row_cells.append(" G ")
            elif cell in env.walls:
                row_cells.append(" # ")
            elif cell in env.traps:
                row_cells.append(" T ")
            else:
                best = int(np.argmax(agent.q_table[r, c]))
                row_cells.append(f" {ACTION_ARROWS[best]} ")
        print("  |" + "".join(row_cells) + "|")
    print("  +" + "---" * env.size + "+")
    print(agent.q_table)

def plot_learning_curve(rewards, window=50, save_path="learning_curve.png"):
    """Plot total reward per episode plus a moving average.

    The raw curve is noisy because of exploration; the moving average makes
    the upward learning trend obvious.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="lightsteelblue",
             linewidth=0.8, label="Reward per episode")

    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window, len(rewards) + 1), moving_avg,
                 color="crimson", linewidth=2.0,
                 label=f"{window}-episode moving average")

    plt.title("Q-Learning on Grid World: Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nLearning curve saved to: {save_path}")

    # Display interactively only when a GUI backend is available (e.g. when
    # run from a desktop). On a headless/Agg backend we skip it silently.
    if "agg" not in plt.get_backend().lower():
        plt.show()

# ---------------------------------------------------------------------------
# Animated 2D maze visualisation
# ---------------------------------------------------------------------------
# Cell colour codes used by the maze image:
#   0 free   1 wall   2 trap   3 start   4 goal
_CELL_CMAP = ListedColormap(
    ["#f0f0f0", "#404040", "#e74c3c", "#2ecc71", "#f1c40f"]
)


def _build_base_grid(env):
    """Turn the maze into a 2D array of cell-type codes for `imshow`."""
    grid = np.zeros((env.size, env.size), dtype=int)
    for (r, c) in env.walls:
        grid[r, c] = 1
    for (r, c) in env.traps:
        grid[r, c] = 2
    grid[env.start] = 3
    grid[env.goal] = 4
    return grid


def _draw_grid(ax, env, title):
    """Render the static maze (cells, grid lines, S/G labels) onto `ax`."""
    ax.imshow(_build_base_grid(env), cmap=_CELL_CMAP, vmin=0, vmax=4)

    # White grid lines between cells.
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    # Note: imshow column = x, row = y, so we pass (col, row).
    ax.text(env.start[1], env.start[0], "S",
            ha="center", va="center", color="white", fontweight="bold")
    ax.text(env.goal[1], env.goal[0], "G",
            ha="center", va="center", color="black", fontweight="bold")


def _render_or_save(anim, fig, save_path, fps):
    """Show the animation live if a GUI backend exists, else save a GIF."""
    if "agg" in plt.get_backend().lower():
        anim.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"  (headless backend) animation saved to: {save_path}")
    else:
        plt.show()


def greedy_rollout(env, agent, max_steps=MAX_STEPS):
    """Replay the FINAL trained policy with no exploration (epsilon = 0).

    At every cell we simply take argmax of the Q-table, so this shows the
    pure 'exploitation' behaviour the agent has learned.

    Returns (states, step_rewards, outcome).
    """
    state = env.reset()
    states = [state]
    step_rewards = []
    outcome = "max-steps"

    for _ in range(max_steps):
        action = int(np.argmax(agent.q_table[state[0], state[1]]))
        next_state, reward, done = env.step(action)
        states.append(next_state)
        step_rewards.append(reward)
        state = next_state
        if done:
            outcome = "GOAL" if state == env.goal else "TRAP"
            break

    return states, step_rewards, outcome


def animate_maze(env, states, step_rewards, outcome,
                 interval=300, save_path="maze_run.gif"):
    """Animate the trained agent walking the maze, one move per frame."""
    fig, ax = plt.subplots(figsize=(6, 6.5))
    _draw_grid(ax, env, "Trained agent - greedy run")

    trail, = ax.plot([], [], color="#3498db", linewidth=2.5, alpha=0.6)
    agent_dot, = ax.plot([], [], "o", color="#2c3e50", markersize=18)
    info = ax.text(0.5, -0.05, "", transform=ax.transAxes,
                   ha="center", va="top", fontsize=11)

    xs = [s[1] for s in states]
    ys = [s[0] for s in states]
    cum_reward = np.concatenate(([0.0], np.cumsum(step_rewards)))

    def update(i):
        agent_dot.set_data([xs[i]], [ys[i]])
        trail.set_data(xs[:i + 1], ys[:i + 1])
        end_tag = f"   ->   {outcome}" if i == len(states) - 1 else ""
        info.set_text(f"step {i}   |   reward {cum_reward[i]:+.2f}{end_tag}")
        return agent_dot, trail, info

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval, blit=False, repeat=False)
    fig.tight_layout()
    _render_or_save(anim, fig, save_path, fps=max(1, int(1000 / interval)))


def animate_training(env, recorded, interval=120,
                     save_path="training_progress.gif"):
    """Replay several recorded TRAINING episodes back-to-back.

    Early episodes wander randomly (high epsilon -> exploration); later
    episodes head almost straight to the goal (low epsilon -> exploitation),
    so you literally watch the agent learn.
    """
    if not recorded:
        return

    fig, ax = plt.subplots(figsize=(6, 6.5))
    _draw_grid(ax, env, "Training progress - watch it learn")

    trail, = ax.plot([], [], color="#9b59b6", linewidth=2.0, alpha=0.55)
    agent_dot, = ax.plot([], [], "o", color="#2c3e50", markersize=18)
    info = ax.text(0.5, -0.05, "", transform=ax.transAxes,
                   ha="center", va="top", fontsize=11)

    # Flatten every episode into per-move frames, pausing briefly at the end
    # of each episode so the final position is readable.
    frames = []
    for ep_num, traj, outcome in recorded:
        for i in range(len(traj)):
            frames.append((ep_num, i, traj, outcome))
        frames.extend([(ep_num, len(traj) - 1, traj, outcome)] * 6)

    def update(f):
        ep_num, i, traj, outcome = frames[f]
        xs = [s[1] for s in traj[:i + 1]]
        ys = [s[0] for s in traj[:i + 1]]
        agent_dot.set_data([xs[-1]], [ys[-1]])
        trail.set_data(xs, ys)
        end_tag = f"   ({outcome})" if i == len(traj) - 1 else ""
        info.set_text(
            f"Episode {ep_num}   |   step {i}/{len(traj) - 1}{end_tag}")
        return agent_dot, trail, info

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval, blit=False, repeat=False)
    fig.tight_layout()
    _render_or_save(anim, fig, save_path, fps=max(1, int(1000 / interval)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Fix the seeds so the run is reproducible from one execution to the next.
    random.seed(42)
    np.random.seed(42)

    env = GridWorld(size=10)
    agent = QAgent(grid_size=env.size,
                   alpha=ALPHA, gamma=GAMMA,
                   epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

    print("Training Q-Learning agent on a 10x10 Grid World...\n")
    start_time = time.time()
    # Record a handful of episodes spread across training so we can later
    # replay them and visually compare early (random) vs late (learned) runs.
    rewards, recorded = train(env, agent, episodes=1000, max_steps=MAX_STEPS,
                              record_episodes=[1, 25, 100, 300, 600, 1000])
    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:0.2f}s over 1000 episodes.")

    # Show the policy the agent ended up with...
    print_policy(env, agent)

    # and the learning curve that should trend upward over time.
    plot_learning_curve(rewards, window=50)

    # watch the agent learn across recorded training episodes,
    print("\nReplaying training progress (watch it learn)...")
    animate_training(env, recorded)
    # then watch the final, fully-trained greedy run move-by-move.
    print("Replaying the final trained policy (greedy run)...")
    states, step_rewards, outcome = greedy_rollout(env, agent)
    animate_maze(env, states, step_rewards, outcome)


if __name__ == "__main__":
    main()
