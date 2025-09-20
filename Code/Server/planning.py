# # Demo implementation of dynamic A* pathfinding on a binary occupancy grid
# # - Inflates obstacles (clearance)
# # - Runs A* on a 2D grid (4-neighbors)
# # - Follows the path for a fixed number of steps, then rebuilds/replans
# # This code runs a demonstration on a 100x100 grid (can scale to 500x500).

import heapq, time, threading
from motor import Ordinary_Car
from mapping import AdvancedMap
import matplotlib.pyplot as plt

class PathPlanner:
    def __init__(self, grid: AdvancedMap, replan_steps=5):
        self.grid = grid
        self.replan_steps = replan_steps
        self.car = Ordinary_Car()
        self.pos = (250, 250)  # start position of the car
        self.facing = "up"     # assume we start facing up
        self.last_path = []    # store last planned path for visualization

    def astar(grid, start, target):
        # possible moves: up, down, left, right
        neighbors = {
            (1,0): "down",
            (-1,0): "up",
            (0,1): "right",
            (0,-1): "left"
        }

        # Manhattan distance
        def heuristic(a,b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        
        # used to calculate the best next step
        heap = [] 
        g_score = {start: 0}
        f_score = {start: heuristic(start,target)}
        heapq.heappush(heap, (f_score[start], start))

        # used to reconstruct path
        parent = {}                            
        
        while heap:
            # get node with smallest f_score
            _, current = heapq.heappop(heap)
            
            # check if we are by the target
            if current == target:
                directions = []

                # get directions to get to target from start
                while current in parent:
                    prev = parent[current]
                    move = (current[0]-prev[0], current[1]-prev[1])
                    directions.append(neighbors[move])
                    current = prev

                directions.reverse()
                return directions
            
            # explore neighbors (up/down/left/right)
            for dx,dy in neighbors:               
                neighbor = (current[0]+dx, current[1]+dy)

                # make sure neighboor is valid
                if not grid.in_bounds(neighbor):       # skip out-of-bounds
                    continue
                if not grid.is_free(neighbor):         # skip obstacles
                    continue
                
                next_g = g_score[current] + 1  # cost of moving to neighbor
                if neighbor not in g_score or next_g < g_score[neighbor]:
                    g_score[neighbor] = next_g
                    parent[neighbor] = current

                    f = next_g + heuristic(neighbor, target)

                    if f_score.get(neighbor, 10**9) > f:
                        f_score[neighbor] = f
                        heapq.heappush(heap, (f, neighbor))
        
        # no Path found
        return None

    def plan(self, start, target):
        return self.astar(self.grid, start, target)

    def planLoop(self, start, target):
        if not self.grid.in_bounds(target) or not self.grid.is_free(target):
            print("Invalid target")
            return None

        facing_order = ["up", "right", "down", "left"]

        def turn_needed(facing, desired):
            i = facing_order.index(facing)
            j = facing_order.index(desired)
            return (j - i) % 4

        while self.pos != target:
            plan = self.plan(self.pos, target)
            self.last_path = plan  # store path for visualization
            if not plan:
                print("No path found")
                break

            for step in plan[:self.replan_steps]:
                # determine if the car needs to turn
                turns = turn_needed(self.facing, step)

                if turns == 1:  # right
                    self.car.set_motor_model(2000, 2000, -2000, -2000)
                    time.sleep(1)
                    self.facing = "right" if self.facing == "up" else facing_order[(facing_order.index(self.facing) + 1) % 4]

                elif turns == 3:  # left
                    self.car.set_motor_model(-2000, -2000, 2000, 2000)
                    time.sleep(0.5)
                    self.facing = facing_order[(facing_order.index(self.facing) - 1) % 4]

                elif turns == 2:  # turn around
                    self.car.set_motor_model(2000, 2000, -2000, -2000)
                    time.sleep(2)
                    self.facing = facing_order[(facing_order.index(self.facing) + 2) % 4]

                # go forward
                self.car.set_motor_model(1000, 1000, 1000, 1000)
                time.sleep(0.5)

                # stop motors
                self.car.set_motor_model(0, 0, 0, 0)

                # update position
                if step == "up":
                    self.pos = (self.pos[0]-1, self.pos[1])
                elif step == "down":
                    self.pos = (self.pos[0]+1, self.pos[1])
                elif step == "left":
                    self.pos = (self.pos[0], self.pos[1]-1)
                elif step == "right":
                    self.pos = (self.pos[0], self.pos[1]+1)

                if self.pos == target:
                    print("Reached target")
                    return True

            # here you’d scan for obstacles and update grid if needed
            self.grid.update_map()

    def visualize_live(self):
        plt.ion()
        fig, ax = plt.subplots()
        img = ax.imshow(self.grid.environment_map, cmap="gray_r", origin="lower")

        while True:
            img.set_data(self.grid.environment_map)

            # draw car (x=col, y=row)
            ax.plot(self.pos[1], self.pos[0], "ro")

            # draw last planned path
            if self.last_path:
                path_x, path_y = [], []
                r, c = self.pos
                for step in self.last_path:
                    if step == "up":    r -= 1
                    if step == "down":  r += 1
                    if step == "left":  c -= 1
                    if step == "right": c += 1
                    path_y.append(r)
                    path_x.append(c)
                ax.plot(path_x, path_y, "b-")

            plt.draw()
            plt.pause(0.1)
            ax.lines.clear()


# === MAIN ===
grid = AdvancedMap(map_dim=500)
planner = PathPlanner(grid)

# start planning loop in background thread
threading.Thread(target=planner.planLoop, args=((250,250), (400,400)), daemon=True).start()

# run visualization in main thread
planner.visualize_live()
