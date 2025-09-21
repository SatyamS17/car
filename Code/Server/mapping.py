import math
from time import sleep

import numpy as np

# from servo import Servo
from ultrasonic import Ultrasonic
import matplotlib.pyplot as plt
import time


class AdvancedMap:
    def __init__(
        self, map_dim: int = 100, cell_size: float = 1.0, ultrasonic_range: float = 50.0
    ) -> None:
        self.ultrasonic = Ultrasonic()
        # self.servo = Servo()

        self.map_dim = map_dim
        self.cell_size = cell_size
        self.ultrasonic_range = ultrasonic_range

        # set map as 0 or 1 for objects (non-probabilistic)
        self.environment_map = np.zeros((map_dim, map_dim), dtype=np.uint8)

        # start car position at center of map
        self.car_row = map_dim // 2
        self.car_col = map_dim // 2
        self.car_angle = 0.0

    def polar_to_cartesian(self, distance: float, angle: float):
        # convert the polar coordinates (dist:15cm at 30deg) to cartesian for map
        angle_rad = math.radians(angle)
        x_cartesian = distance * math.cos(angle_rad)
        y_cartesian = distance * math.sin(angle_rad)
        return x_cartesian, y_cartesian

    def cartesian_coords_to_map(self, x: float, y: float):
        # calculate new coordinates if car rotates
        # rotation formula to base it off of car's reference x and y coordinates
        heading_rad = math.radians(self.car_angle)
        x_rotated = x * math.cos(heading_rad) - y * math.sin(heading_rad)
        y_rotated = x * math.sin(heading_rad) + y * math.cos(heading_rad)

        # convert to map indices
        # place with respect to map origin (0, 0) which is where car's starting position was
        map_row = int(self.car_row + y_rotated / self.cell_size)
        map_col = int(self.car_col + x_rotated / self.cell_size)

        # bound checking within map
        map_row = max(0, min(self.map_dim - 1, map_row))
        map_col = max(0, min(self.map_dim - 1, map_col))

        return map_row, map_col

    def scan_environment(self, start_angle=30, end_angle=150, step=10):
        scan_data = []

        for angle in range(start_angle, end_angle + 1, step):
            # self.servo.set_servo_pwm("0", angle)
            sleep(0.1)

            distance = self.ultrasonic.get_distance()

            if (distance > 0) and (distance < self.ultrasonic_range):
                scan_data.append((angle, distance))

        # self.servo.set_servo_pwm("0", 90)
        return scan_data

    def interpolate_line(self, r1: int, c1: int, r2: int, c2: int):
        # Bresenham's line on (row, col)
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc
        r, c = r1, c1

        while True:
            if 0 <= r < self.map_dim and 0 <= c < self.map_dim:
                self.environment_map[r, c] = 1

            if r == r2 and c == c2:
                break

            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    # combines the above functions to map the sensor readings to the map
    def update_map_with_scan(self, scan_data):
        if not scan_data:
            return

        map_points = []
        for angle, distance in scan_data:
            x_cart, y_cart = self.polar_to_cartesian(distance, angle)
            map_row, map_col = self.cartesian_coords_to_map(x_cart, y_cart)

            self.environment_map[map_row, map_col] = 1
            map_points.append((map_row, map_col))

        for i in range(len(map_points) - 1):
            r1, c1 = map_points[i]
            r2, c2 = map_points[i + 1]

            dist = math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2) * self.cell_size
            if dist > 20:
                continue

            self.interpolate_line(r1, c1, r2, c2)

    def update_car_position(
        self, velocity: float, angular_velocity: float, dt: float = 0.1
    ):
        self.car_angle = (self.car_angle + angular_velocity * dt) % 360

        distance = velocity * dt
        heading_rad = math.radians(self.car_angle)

        dr = distance * math.sin(heading_rad) / self.cell_size
        dc = distance * math.cos(heading_rad) / self.cell_size

        self.car_row += dr
        self.car_col += dc

        self.car_row = max(0, min(self.map_dim - 1, self.car_row))
        self.car_col = max(0, min(self.map_dim - 1, self.car_col))

    def update_map(self):
        data = self.scan_environment()
        self.update_map_with_scan(scan_data=data)

    def in_bounds(self, pos):
        r, c = pos
        return (
            0 <= r < self.environment_map.shape[0]
            and 0 <= c < self.environment_map.shape[1]
        )

    def is_free(self, pos):
        r, c = pos
        return self.in_bounds(pos) and self.environment_map[r, c] == 0
        
    def visualize(self, refresh=0.2):
        plt.ion()  # interactive mode ON
        fig, ax = plt.subplots()
        img = ax.imshow(self.environment_map, cmap='binary', origin='lower', interpolation='nearest')
        ax.set_title("Environment Map")
        ax.set_xlabel("X (col)")
        ax.set_ylabel("Y (row)")
        ax.set_aspect("equal")

        while True:
            # Update the image data
            img.set_data(self.environment_map)

            # Redraw the canvas
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(refresh)  # control update rate
