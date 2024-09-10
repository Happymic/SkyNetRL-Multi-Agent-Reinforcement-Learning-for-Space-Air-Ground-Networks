import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.visualization_dir = "visualizations"
        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)

    def visualize(self, env, agents, episode):
        plt.figure(figsize=(10, 10))
        plt.xlim(0, self.config.area_size)
        plt.ylim(0, self.config.area_size)

        # Plot PoIs
        plt.scatter(env.pois[:, 0], env.pois[:, 1], c='g', label='PoIs')

        # Plot satellites
        plt.scatter(env.satellites[:, 0], env.satellites[:, 1], c='r', marker='^', s=100, label='Satellites')

        # Plot UAVs
        plt.scatter(env.uavs[:, 0], env.uavs[:, 1], c='b', marker='s', s=50, label='UAVs')

        # Plot ground stations
        plt.scatter(env.ground_stations[:, 0], env.ground_stations[:, 1], c='y', marker='o', s=50, label='Ground Stations')

        # Plot coverage areas
        for sat in env.satellites:
            circle = plt.Circle((sat[0], sat[1]), env.satellite_range, color='r', fill=False, alpha=0.3)
            plt.gca().add_artist(circle)

        for uav in env.uavs:
            circle = plt.Circle((uav[0], uav[1]), env.uav_range, color='b', fill=False, alpha=0.3)
            plt.gca().add_artist(circle)

        for gs in env.ground_stations:
            circle = plt.Circle((gs[0], gs[1]), env.ground_station_range, color='y', fill=False, alpha=0.3)
            plt.gca().add_artist(circle)

        plt.legend()
        plt.title(f"Space-Air-Ground Network Coverage - Episode {episode}")
        plt.savefig(os.path.join(self.visualization_dir, f"sag_coverage_episode_{episode}.png"))
        plt.close()