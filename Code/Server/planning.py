# Demo implementation of dynamic A* pathfinding on a binary occupancy grid
# - Inflates obstacles (clearance) assumed to be handled inside AdvancedMap
# - Runs A* on a 2D grid (8-neighbors: orthogonal + diagonal)
# - Follows the path for a fixed number of steps, then rebuilds/replans
# - Prevents corner-cutting for diagonal moves (only allow diagonal if both adjacent orthogonals are free)
#
# This code runs a demonstration on a 100x100 grid (can scale to 500x500).

import heapq
import time
import threading
from motor import Ordinary_Car
from mapping import AdvancedMap
import matplotlib.pyplot as plt
import numpy as np
import math

class PathPlanner:
    def __init__(self, grid: AdvancedMap, replan_steps=5):
        self.grid = grid
        self.replan_steps = replan_steps
        self.pos = (10, 10)  # start position of the car (row, col)
        self.facing = "up"  # starting facing direction
        self.last_path = []  # store last planned *directions* for visualization
        self.path_history = [] # Store all visited positions
        np.set_printoptions(threshold=np.inf)

    @staticmethod
    def astar(grid, start, target):
        """
        A* over 8-connected grid:
         - orthogonal moves cost = 1
         - diagonal moves cost = sqrt(2)
         - heuristic = octile heuristic (admissible for these costs)
         - prevents corner-cutting: diagonal allowed only if both adjacent orthogonals are free
        Returns: list of direction labels (e.g., ["up-right", "up", "left", ...]) or None if no path.
        """

        # neighbor: (dx, dy) where dx = row offset, dy = col offset
        neighbors = {
            (-1, 0): "up",
            (1, 0): "down",
            (0, -1): "left",
            (0, 1): "right",
            (-1, -1): "up-left",
            (-1, 1): "up-right",
            (1, -1): "down-left",
            (1, 1): "down-right"
        }

        def octile_heuristic(a, b):
            # a, b are (row, col)
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            D = 1.0
            D2 = math.sqrt(2)
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

        start = tuple(start)
        target = tuple(target)

        open_heap = []
        g_score = {start: 0.0}
        f_score = {start: octile_heuristic(start, target)}
        heapq.heappush(open_heap, (f_score[start], start))
        parent = {}

        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                # reconstruct path as direction labels
                directions = []
                while current in parent:
                    prev = parent[current]
                    move = (current[0] - prev[0], current[1] - prev[1])
                    directions.append(neighbors[move])
                    current = prev
                directions.reverse()
                return directions

            for (dx, dy), label in neighbors.items():
                neighbor = (current[0] + dx, current[1] + dy)

                if not grid.in_bounds(neighbor):
                    continue

                # If neighbor is occupied, skip
                if not grid.is_free(neighbor):
                    continue

                # Prevent corner-cutting: if moving diagonally, both orthogonal neighbors must be free
                if dx != 0 and dy != 0:
                    orth1 = (current[0] + dx, current[1])   # vertical neighbor
                    orth2 = (current[0], current[1] + dy)   # horizontal neighbor
                    if not (grid.in_bounds(orth1) and grid.in_bounds(orth2)):
                        continue
                    if not (grid.is_free(orth1) and grid.is_free(orth2)):
                        # one of the adjacent orthogonals blocked -> disallow diagonal
                        continue

                # Movement cost
                step_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    f = tentative_g + octile_heuristic(neighbor, target)
                    if f < f_score.get(neighbor, float("inf")):
                        f_score[neighbor] = f
                        heapq.heappush(open_heap, (f, neighbor))

        return None


    def planLoop(self, start, target):
        # start and target are (row, col)
        self.pos = tuple(start)
        if not self.grid.in_bounds(target) or not self.grid.is_free(target):
            print("Invalid target")
            return None

        # 8 orientations in clockwise order (45-degree increments)
        facing_order = [
            "up",        # 0
            "up-right",  # 1
            "right",     # 2
            "down-right",# 3
            "down",      # 4
            "down-left", # 5
            "left",      # 6
            "up-left"    # 7
        ]

        def turn_to(facing, desired):
            """
            Compute minimal rotation from facing to desired.
            Returns tuple (direction, steps) where direction is "right" or "left" and steps is number of 45-degree steps.
            """
            i = facing_order.index(facing)
            j = facing_order.index(desired)
            steps_right = (j - i) % 8
            steps_left = (i - j) % 8
            if steps_right <= steps_left:
                return ("right", steps_right)
            else:
                return ("left", steps_left)

        # map direction labels to offsets (row delta, col delta)
        move_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
            "up-left": (-1, -1),
            "up-right": (-1, 1),
            "down-left": (1, -1),
            "down-right": (1, 1)
        }

        # keep planning until we reach the target
        while self.pos != tuple(target):
            plan = self.astar(self.grid, self.pos, target)
            self.last_path = plan  # store planned direction labels for visualization

            if not plan:
                print("No path found")
                break

            # iterate through plan for replan_steps steps
            for step_label in plan[:self.replan_steps]:
                # turn toward step_label
                # Note: step_label is also one of the facing_order values (same names), so we can reuse
                if step_label not in facing_order:
                    # Shouldn't happen, but safety
                    desired_facing = "up"
                else:
                    desired_facing = step_label

                turn_dir, turn_steps = turn_to(self.facing, desired_facing)

                # perform turns (simulate time and update facing)
                if turn_steps > 0:
                    # choose per-45-degree rotation time (example)
                    time_per_45deg = 0.25  # seconds per 45-degree turn (tune as needed)
                    print(f"Turning {turn_dir} {turn_steps} x 45°")
                    # simulate turning by updating facing step-by-step
                    for _ in range(turn_steps):
                        if turn_dir == "right":
                            new_index = (facing_order.index(self.facing) + 1) % 8
                        else:
                            new_index = (facing_order.index(self.facing) - 1) % 8
                        self.facing = facing_order[new_index]
                        time.sleep(time_per_45deg)
                else:
                    # already facing the correct direction
                    pass

                # move forward one cell (could be diagonal)
                print(f"Moving {step_label}")
                time.sleep(0.2)  # movement duration (tune as needed)

                delta = move_to_delta[step_label]
                new_pos = (self.pos[0] + delta[0], self.pos[1] + delta[1])

                # safety check before moving
                if not self.grid.in_bounds(new_pos) or not self.grid.is_free(new_pos):
                    print(f"Blocked at intended move {new_pos}; will replan.")
                    break

                # Update position and history
                self.pos = new_pos
                self.path_history.append(self.pos)
                self.grid.car_row = self.pos[0]
                self.grid.car_col = self.pos[1]
                self.grid.car_angle = facing_order.index(self.facing) * 90.0

                # If reached target
                if self.pos == tuple(target):
                    print("Reached target")
                    self.last_path = []
                    return True

            # allow the grid to update (dynamic environment)
            self.grid.update_map()

        return False


    def visualize_live(self):
        plt.ion()
        fig, ax = plt.subplots()

        # show the grid in background
        img = ax.imshow(self.grid.environment_map, cmap="gray_r", origin="lower")

        ax.set_title('A* Pathfinding (8-connected)')
        ax.set_xlabel('X (col)')
        ax.set_ylabel('Y (row)')
        ax.set_aspect('equal')

        planned_path_line, = ax.plot([], [], "-", linewidth=2, label="Planned Path")
        car_point, = ax.plot([], [], "ro", markersize=6, label="Car Position")
        completed_path_line, = ax.plot([], [], "g-", linewidth=1, label="Completed Path")

        # scatter plot for obstacles (will be updated each frame)
        obstacle_scatter = ax.scatter([], [], c="black", s=20, marker="s", label="Obstacles")

        ax.legend()

        move_to_delta = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
            "up-left": (-1, -1),
            "up-right": (-1, 1),
            "down-left": (1, -1),
            "down-right": (1, 1)
        }

        while True:
            # === Update environment ===
            env = np.array(self.grid.environment_map)

            # find obstacle cells (where value == 1)
            obs_rows, obs_cols = np.where(env == 1)

            # update obstacle scatter (big black squares)
            obstacle_scatter.set_offsets(np.c_[obs_cols, obs_rows])

            # car position
            car_point.set_data([self.pos[1]], [self.pos[0]])

            # completed path history
            if self.path_history:
                hist_rows, hist_cols = zip(*self.path_history)
                completed_path_line.set_data(hist_cols, hist_rows)

            # planned path (based on direction steps)
            if self.last_path:
                path_cols, path_rows = [], []
                r, c = self.pos
                for step_label in self.last_path:
                    dr, dc = move_to_delta.get(step_label, (0, 0))
                    r += dr
                    c += dc
                    path_rows.append(r)
                    path_cols.append(c)
                planned_path_line.set_data(path_cols, path_rows)
            else:
                planned_path_line.set_data([], [])

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.15)


# === MAIN ===
if __name__ == "__main__":
    grid = AdvancedMap(map_dim=100)
    planner = PathPlanner(grid)

    # start planning loop in background thread
    threading.Thread(
        target=planner.planLoop,
        args=((5, 5), (90, 90)),
        daemon=True
    ).start()

    # run visualization in main thread
    planner.visualize_live()
