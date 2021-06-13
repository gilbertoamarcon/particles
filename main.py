#!/usr/bin/env python

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


class ParticleFilter:
    """A simple 2D particle Filter"""
    def __init__(self, min_range, max_range, error_cost):
        self.odometer = 0
        self.error_cost = error_cost
        self.particles = []
        for p in range(min_range, max_range):
            self.particles.append(Particle(p))

    def get_arrays(self):
        """Getting the particle positions and probabilities as two arrays for plotting"""
        positions = [particle.x + self.odometer for particle in self.particles]
        probabilities = [particle.probability for particle in self.particles]
        return positions, probabilities

    def update(self, world, odometer, radar_terrain):
        """Updating all particles using odometry and the radar-measured terrain height"""
        self.odometer = odometer
        for particle in self.particles:
            particle.update(world, self.odometer, radar_terrain, self.error_cost)
        self.normalize()

    def normalize(self):
        """Normalizing the particle probabilities to the 0-1 range"""
        max_probability = max([particle.probability for particle in self.particles])
        for particle in self.particles:
            particle.probability /= max_probability


class Particle:
    """A single particle"""
    def __init__(self, x):
        self.x = x
        self.probability = 1.0

    def update(self, world, odometer, radar_terrain, error_cost):
        """Updating the particle probability by comparting the radar-measured ground altitude versus the expected altitude"""
        expected_terrain = world.get_terrain_altitude(self.x + odometer)
        terrain_error = expected_terrain - radar_terrain
        self.probability *= 1.0-abs(math.atan(terrain_error * error_cost)/math.pi)


class Robot:
    """A simple 2D robot"""
    def __init__(self, x, altitude):
        self.x = x
        self.altitude = altitude
        self.odometer = 0

    def displace(self, displacement):
        """Moving the robot left or right by distance 'displacement'"""
        self.x += displacement
        self.odometer += displacement

    def get_radar_altitude(self, world):
        """Measuring the distance to the ground using the downward point radar"""
        return self.altitude-world.get_terrain_altitude(self.x)

    def get_radar_terrain(self, radar_altitude):
        """Estimating the terrain height (ground altitude) using the radar altitude"""
        return self.altitude-radar_altitude


class World:
    """A 2D terrain world"""
    def __init__(self):
        self.terrain = 100.0*np.ones(1000)
        self.make_mountain(100, 150, 1000.0)
        self.make_mountain(400, 100, 500.0)
        self.make_mountain(500, 50, 250.0)
        self.make_mountain(600, 50, 250.0)
        self.make_mountain(800, 80, 1500.0)

    def get_terrain_altitude(self, x):
        """Reading the terrain altitude (ground height) at position x"""
        if x < 0:
            return self.terrain[0]
        if x >= len(self.terrain):
            return self.terrain[-1]
        return self.terrain[x]

    def make_mountain(self, loc, width, height):
        """Making a simple triangular mountain"""
        half_width = int(width/2)
        self.terrain[loc:loc+half_width] += np.linspace(0.0, height, half_width)
        self.terrain[loc+half_width:loc+width] += np.linspace(height, 0.0, half_width)

def on_key(event, world, robot, particle_filter, filter_handle, robot_handle):
    """Getting user input, moving the robot, and updating the particle filter"""
    sys.stdout.flush()
    if event.key == "right":
        robot.displace(20)
        update(world, robot, particle_filter, filter_handle, robot_handle)
    if event.key == "left":
        robot.displace(-20)
        update(world, robot, particle_filter, filter_handle, robot_handle)

def update(world, robot, particle_filter, filter_handle, robot_handle):
    """Moving the robot, and updating the particle filter"""
    # Updating the particle filter
    radar_altitude = robot.get_radar_altitude(world)
    radar_terrain = robot.get_radar_terrain(radar_altitude)
    particle_filter.update(world, robot.odometer, radar_terrain)
    # Plotting the results
    positions, probabilities = particle_filter.get_arrays()
    filter_handle.set_xdata(positions)
    filter_handle.set_ydata(probabilities)
    robot_handle.set_xdata(robot.x)
    robot_handle.set_ydata(robot.altitude)
    plt.draw()

def setup_terrain_plot(ax1, world):
    """Setting up the 2D terrain plot"""
    ax1.plot(world.terrain, color="brown")
    ax1.set_xlabel("longitude (km)")
    ax1.set_ylabel("altitude (m)")
    ax1.set_xlim([0.0, 1000.0])
    ax1.set_ylim([0.0, 5000.0])

def setup_robot_plot(ax1):
    """Setting up the robot position plot"""
    handle, = ax1.plot([], [], color="green", marker="*")
    return handle

def setup_particle_filter_plot(ax1):
    """Setting up the particle filter plot"""
    ax2 = ax1.twinx() 
    handle, = ax2.plot([], [])
    ax2.set_xlabel("longitude (km)")
    ax2.set_ylabel("probability (0-1)")
    ax2.set_ylim([0.0, 1.1])
    return handle

def main():
    # Initializing world, robot, and filter
    world = World()
    robot = Robot(x=320, altitude=3000.0)
    particle_filter = ParticleFilter(0, 1000, 0.005)

    # Setting up plots
    fig, ax1 = plt.subplots()
    setup_terrain_plot(ax1, world)
    robot_handle = setup_robot_plot(ax1)
    filter_handle = setup_particle_filter_plot(ax1)

    # Run time
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, world, robot, particle_filter, filter_handle, robot_handle))
    plt.show()



if __name__ == "__main__":
    main()
