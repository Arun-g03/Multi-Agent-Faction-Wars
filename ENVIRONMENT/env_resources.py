"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from ENVIRONMENT.Resources.apple_tree import AppleTree
from ENVIRONMENT.Resources.gold_lump import GoldLump


class ResourceManager:
    def __init__(self, terrain, tensorboard_logger=None):
        """
        Manage resources in the environment, such as apple trees and gold lumps.
        """
        self.resources = []
        self.terrain = terrain
        self.gold_count = 0
        self.apple_tree_count = 0
        self.generate_resources()

        """The current episode"""

    #     ____                           _
    #    / ___| ___ _ __   ___ _ __ __ _| |_ ___   _ __ ___  ___  ___  _   _ _ __ ___ ___  ___
    #   | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \ | '__/ _ \/ __|/ _ \| | | | '__/ __/ _ \/ __|
    #   | |_| |  __/ | | |  __/ | | (_| | ||  __/ | | |  __/\__ \ (_) | |_| | | | (_|  __/\__ \
    #    \____|\___|_| |_|\___|_|  \__,_|\__\___| |_|  \___||___/\___/ \__,_|_|  \___\___||___/
    #

    def generate_resources(
        self,
        add_trees=0,
        add_gold_lumps=0,
        tree_density=None,
        gold_zone_probability=None,
        gold_spawn_density=None,
        episode=0,
    ):
        """
        Generate resources (trees and gold lumps) either dynamically (specific amounts) or via density and probability for startup.
        :param add_trees: Number of trees to add dynamically.
        :param add_gold_lumps: Number of gold lumps to add dynamically.
        :param tree_density: Density of tree placement (used for startup).
        :param gold_zone_probability: Probability of starting a gold zone (used for startup).
        :param gold_spawn_density: Density of gold lumps within a gold zone (used for startup).
        """
        # Reset resource counts before generation
        self.apple_tree_count = 0
        self.gold_count = 0

        # Use default density/probability if not provided (for startup)
        tree_density = (
            tree_density if tree_density is not None else utils_config.TREE_DENSITY
        )
        gold_zone_probability = (
            gold_zone_probability
            if gold_zone_probability is not None
            else utils_config.GOLD_ZONE_PROBABILITY
        )
        gold_spawn_density = (
            gold_spawn_density
            if gold_spawn_density is not None
            else utils_config.GOLD_SPAWN_DENSITY
        )

        # Track how many resources have been added
        added_trees = 0
        added_gold_lumps = 0

        # Get world dimensions (in grid units)
        grid_width = len(self.terrain.grid)
        grid_height = len(self.terrain.grid[0])

        # Iterate through terrain grid
        for x in range(grid_width):
            for y in range(grid_height):
                cell = self.terrain.grid[x][y]

                # Ensure the cell is valid for placing a resource
                if cell["type"] == "land" and not cell["occupied"]:
                    # Check bounds (prevent placing outside the world grid)
                    if not (0 <= x < grid_width and 0 <= y < grid_height):
                        continue

                    # Add specific number of trees
                    if add_trees > 0 and added_trees < add_trees:
                        self.resources.append(
                            AppleTree(
                                x * utils_config.CELL_SIZE,
                                y * utils_config.CELL_SIZE,
                                x,
                                y,
                                self.terrain,
                                self,
                            )
                        )
                        cell["occupied"] = True
                        cell["resource_type"] = "apple_tree"
                        self.apple_tree_count += 1
                        added_trees += 1
                        continue

                    # Add specific number of gold lumps
                    if add_gold_lumps > 0 and added_gold_lumps < add_gold_lumps:
                        gold_lump = GoldLump(
                            x * utils_config.CELL_SIZE,
                            y * utils_config.CELL_SIZE,
                            x,
                            y,
                            self.terrain,
                            self,
                        )
                        self.resources.append(gold_lump)
                        cell["occupied"] = True
                        cell["resource_type"] = "gold_lump"
                        self.gold_count += 1
                        added_gold_lumps += 1
                        continue

                    # Use density/probability for startup
                    if add_trees == 0 and random.random() < tree_density:
                        self.resources.append(
                            AppleTree(
                                x * utils_config.CELL_SIZE,
                                y * utils_config.CELL_SIZE,
                                x,
                                y,
                                self.terrain,
                                self,
                            )
                        )
                        cell["occupied"] = True
                        cell["resource_type"] = "apple_tree"
                        self.apple_tree_count += 1

                    elif (
                        add_gold_lumps == 0 and random.random() < gold_zone_probability
                    ):
                        # Generate gold lumps in a zone
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = x + dx, y + dy
                                if (
                                    0 <= nx < grid_width and 0 <= ny < grid_height
                                ):  # Bounds check
                                    gold_cell = self.terrain.grid[nx][ny]
                                    if (
                                        gold_cell["type"] == "land"
                                        and not gold_cell["occupied"]
                                    ):
                                        if random.random() < gold_spawn_density:
                                            gold_lump = GoldLump(
                                                nx * utils_config.CELL_SIZE,
                                                ny * utils_config.CELL_SIZE,
                                                nx,
                                                ny,
                                                self.terrain,
                                                self,
                                            )
                                            self.resources.append(gold_lump)
                                            gold_cell["occupied"] = True
                                            gold_cell["resource_type"] = "gold_lump"
                                            self.gold_count += 1

        # Log to TensorBoard
        print(
            f"Generated {self.apple_tree_count} apple trees and {self.gold_count} gold lumps."
        )
        if utils_config.ENABLE_TENSORBOARD:
            tensorboard_logger.log_scalar(
                "Resources/AppleTrees_Added", self.apple_tree_count, episode
            )
            tensorboard_logger.log_scalar(
                "Resources/GoldLumps_Added", self.gold_count, episode
            )

    #    _   _           _       _          __    _                    _  __       _            _      _           ___
    #   | | | |_ __   __| | __ _| |_ ___   / /___| | ___  __ _ _ __   (_)/ _|   __| | ___ _ __ | | ___| |_ ___  __| \ \
    #   | | | | '_ \ / _` |/ _` | __/ _ \ | |/ __| |/ _ \/ _` | '_ \  | | |_   / _` |/ _ \ '_ \| |/ _ \ __/ _ \/ _` || |
    #   | |_| | |_) | (_| | (_| | ||  __/ | | (__| |  __/ (_| | | | | | |  _| | (_| |  __/ |_) | |  __/ ||  __/ (_| || |
    #    \___/| .__/ \__,_|\__,_|\__\___| | |\___|_|\___|\__,_|_| |_| |_|_|    \__,_|\___| .__/|_|\___|\__\___|\__,_|| |
    #         |_|                          \_\                                           |_|                        /_/

    def update(self):
        """Update resources, removing any that are depleted."""
        for resource in self.resources[
            :
        ]:  # Iterate over a copy to avoid mutation during iteration
            if resource.is_depleted():
                print(
                    f"Depleted resource detected at ({resource.grid_x}, {resource.grid_y}). Removing..."
                )
                resource.remove_from_terrain()


# AppleTree and GoldLump classes have been moved to:
# - ENVIRONMENT/Resources/apple_tree.py
# - ENVIRONMENT/Resources/gold_lump.py


# class Boulder:
#     """ Sole purpose is to block movement and prevent units from passing through it. """
#     def __init__(self, grid_x, grid_y):
#         raise NotImplementedError("Boulder class is not implemented yet.")
#         self.grid_x = grid_x
#         self.grid_y = grid_y
#         self.x = grid_x * utils_config.CELL_SIZE  # Convert grid to screen coordinates
#         self.y = grid_y * utils_config.CELL_SIZE
#         self.width = utils_config.CELL_SIZE
#         self.height = utils_config.CELL_SIZE


#     def render(self, screen, camera):
#         """Render the Boulder object in the world."""
#         raise NotImplementedError("Boulder render method is not implemented yet.")
#         screen_x = (self.x - camera.x) * camera.zoom
#         screen_y = (self.y - camera.y) * camera.zoom
# pygame.draw.rect(screen, (139, 69, 19), (screen_x, screen_y, self.width
# * camera.zoom, self.height * camera.zoom))  # Brown colour for boulders
