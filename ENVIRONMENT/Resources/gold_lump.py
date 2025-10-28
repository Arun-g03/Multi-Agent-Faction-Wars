"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config


class GoldLump:
    def __init__(self, x, y, grid_x, grid_y, terrain, resource_manager, quantity=5):
        """
        Initialise a gold lump with a given quantity of gold.
        :param x: X position of the gold lump.
        :param y: Y position of the gold lump.
        :param grid_x: X position on the grid.
        :param grid_y: Y position on the grid.
        :param terrain: Reference to the terrain object for interaction.
        :param quantity: Initial quantity of gold in the lump.
        """
        self.x = x
        self.y = y
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.terrain = terrain
        self.quantity = utils_config.GoldLump_base_quantity  # Set quantity of gold
        self.resource_manager = resource_manager

        if not utils_config.HEADLESS_MODE:
            # Load and scale the gold image
            self.image = pygame.image.load(utils_config.GOLD_IMAGE_PATH).convert_alpha()
            # Scale to 3% of screen height
            image_size = int(utils_config.SCREEN_HEIGHT * 0.03)
            self.image = pygame.transform.scale(self.image, (image_size, image_size))

    def mine(self, credit_callback=None):
        """Mine gold from the lump and credit it using the provided callback."""
        if self.quantity > 0:
            self.quantity -= 1

            if credit_callback:
                credit_callback(1)  # Credit the mined gold to the faction

            if self.is_depleted():
                # print(f"Gold Lump at ({self.grid_x}, {self.grid_y}) is now depleted. Removing from terrain.")
                self.remove_from_terrain()
            return 1
        else:
            # Final backup to remove resource if it still exists
            if self.is_depleted():
                print(
                    f"Gold Lump at ({self.grid_x}, {self.grid_y}) is depleted but still present. Removing from terrain."
                )
                self.remove_from_terrain()
        return 0

    def is_depleted(self):
        """
        Check if the gold lump is depleted of gold.
        :return: True if the gold lump has no gold left, False otherwise.
        """
        return self.quantity == 0

    def remove_from_terrain(self):
        """Remove the gold lump from the terrain and mark it as no longer available."""
        # print(f"Attempting to remove Gold Lump at ({self.grid_x}, {self.grid_y}). Quantity: {self.quantity}")

        # Ensure resource is actually depleted
        if self.quantity > 0:
            print(
                f"Error: Trying to remove Gold Lump at ({self.grid_x}, {self.grid_y}) before depletion."
            )
            return  # Abort removal if the resource is not depleted

        self.terrain.grid[self.grid_x][self.grid_y]["occupied"] = False
        self.terrain.grid[self.grid_x][self.grid_y]["resource_type"] = None
        if self in self.resource_manager.resources:
            self.resource_manager.resources.remove(self)
            # print(f"\033[31mResource successfully removed: Gold Lump at ({self.grid_x}, {self.grid_y})\033[0m")
        else:
            raise RuntimeError(
                f"Gold Lump at ({self.grid_x}, {self.grid_y}) not found in resource manager."
            )

    def render(self, screen, camera):
        """
        Render the gold lump, centering it within the grid cell.
        """
        # Skip rendering in headless mode
        if utils_config.HEADLESS_MODE:
            return

        # Calculate the screen position using the camera
        screen_x, screen_y = camera.apply((self.x, self.y))

        # Calculate the final size
        final_size = int(
            utils_config.CELL_SIZE
            * utils_config.SCALING_FACTOR
            * camera.zoom
            * utils_config.GoldLump_Scale_Img
        )
        gold_image_scaled = pygame.transform.scale(self.image, (final_size, final_size))

        # Offset to center the image in the cell
        offset_x = final_size // 2
        offset_y = final_size // 2

        screen.blit(gold_image_scaled, (screen_x - offset_x, screen_y - offset_y))
