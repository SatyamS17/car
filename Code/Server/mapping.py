import math
from time import sleep

import numpy as np
from servo import Servo
from ultrasonic import Ultrasonic


class AdvancedMap:
    def __init__(
        self, map_dim: int = 500, cell_size: float = 1.0, ultrasonic_range: float = 50.0
    ) -> None:
        self.ultrasonic = Ultrasonic()
        self.servo = Servo()

        self.map_dim = map_dim
        self.cell_size = cell_size
        self.ultrasonic_range = ultrasonic_range

        # set map as 0 or 1 for objects (non-probabilistic)
        self.environment_map = np.zeros((map_dim, map_dim), dtype=np.uint8)

        # start car position at center of map
        self.car_x_pos = map_dim // 2
        self.car_y_pos = map_dim // 2
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
        map_x = int(self.car_x_pos + x_rotated / self.cell_size)
        map_y = int(self.car_y_pos + y_rotated / self.cell_size)

        # bound checking within map
        map_x = max(0, min(self.map_dim - 1, map_x))
        map_y = max(0, min(self.map_dim - 1, map_y))

        return map_x, map_y

    def scan_environment(
        self, servo, ultrasonic, start_angle=30, end_angle=150, step=10
    ):
        scan_data = []

        for angle in range(start_angle, end_angle + 1, step):
            self.servo.set_servo_pwm("0", angle)
            sleep(0.2)

            distance = self.ultrasonic.get_distance()

            if (distance > 0) and (distance < self.ultrasonic_range):
                scan_data.append((angle, distance))

        self.servo.set_servo_pwm("0", 90)
        return scan_data

    def interpolate_line(self, x1: int, y1: int, x2: int, y2: int):
        # interpolation algorithm using Bresenham's line algorithm
        # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

        # horizontal and vertical distance of the line coverage
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # determines direction of line to draw in from start to end in steps
        if x1 < x2:
            sx = 1
        else:
            sx = -1

        if y1 < y2:
            sy = 1
        else:
            sy = -1

        # error term
        err = dx - dy

        # starting pixel corrdinates
        x, y = x1, y1

        # loop to form a line between starting x, y coords and x2, y2 end
        while True:
            if 0 <= x < self.map_dim and 0 <= y < self.map_dim:
                self.environment_map[y, x] = 1

            if x == x2 and y == y2:
                break

            # draws line with respect to error term
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # combines the above functions to map the sensor readings to the map
    def update_map_with_scan(self, scan_data):
        if not scan_data:
            return

        map_points = []

        for angle, distance in scan_data:
            x_cart, y_cart = self.polar_to_cartesian(distance, angle)

            map_x, map_y = self.cartesian_coords_to_map(x_cart, y_cart)

            self.environment_map[map_y, map_x] = 1
            map_points.append((map_x, map_y))

        for i in range(len(map_points) - 1):
            x1, y1 = map_points[i]
            x2, y2 = map_points[i + 1]

            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * self.cell_size
            if dist > 20:
                continue

            self.interpolate_line(x1, y1, x2, y2)

    def update_car_position(
        self, velocity: float, angular_velocity: float, dt: float = 0.1
    ):
        # calculate angular displacement for turning
        self.car_angle += angular_velocity * dt

        # comopute car_angle in terms of 0-360 range
        self.car_angle = self.car_angle % 360

        # calculate displacement for physical car
        distance = velocity * dt
        heading_rad = math.radians(self.car_angle)

        # movement to map coordinates
        dx = distance * math.cos(heading_rad) / self.cell_size
        dy = distance * math.sin(heading_rad) / self.cell_size

        # updates car position for map
        self.car_x_pos += dx
        self.car_y_pos += dy

        self.car_x_pos = max(0, min(self.map_dim - 1, self.car_x_pos))
        self.car_y_pos = max(0, min(self.map_dim - 1, self.car_y_pos))
