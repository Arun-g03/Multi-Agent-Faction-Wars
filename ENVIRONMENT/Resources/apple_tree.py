"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config


class AppleTree:
    def __init__(self, x, y, grid_x, grid_y, terrain, resource_manager, quantity=10):
        """
        Initialise an apple tree with a given quantity of apples.
        :param x: X position of the tree.
        :param y: Y position of the tree.
        :param grid_x: X position on the grid.
        :param grid_y: Y position on the grid.
        :param terrain: Reference to the terrain object for interaction.
        :param quantity: Initial number of apples on the tree.
        """
        self.x = x
        self.y = y
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.position = (x, y)
        self.grid_position = (grid_x, grid_y)
        self.terrain = terrain
        self.resource_manager = resource_manager
        self.max_quantity = quantity  # Store max quantity
        # By default, natural trees spawn mature with apples
        # Planted trees will be set to growth_stage 0 and is_planted True after init
        self.growth_stage = 5  # 0-5: from sapling to mature tree (default to mature)
        self.last_growth_time = time.time()  # Track when to grow to next stage
        self.is_planted = False  # Track if this is a planted tree vs naturally spawned

        # Set initial quantity based on whether quantity parameter indicates a sapling
        # If quantity=0, treat as planted sapling, otherwise natural tree with apples
        if quantity == 0:
            # This is a sapling (planted tree starting at stage 0)
            self.quantity = 0  # Saplings have no apples yet
        else:
            self.quantity = (
                utils_config.Apple_Base_quantity
            )  # Natural trees spawn with apples

        self.last_regen_time = time.time()  # Track last regeneration time

        # Cache sprite sheet (shared across all trees to avoid reloading)
        if not hasattr(AppleTree, "_sprite_sheet"):
            if not utils_config.HEADLESS_MODE:
                AppleTree._sprite_sheet = pygame.image.load(
                    utils_config.TREE_IMAGE_PATH
                ).convert_alpha()
            else:
                AppleTree._sprite_sheet = None

        self.update()

    def update(self):
        """Update the tree's growth stage and apple quantity based on regeneration time."""
        current_time = time.time()

        # Growth logic: planted trees grow over time
        if self.is_planted and self.growth_stage < 5:
            growth_time_required = 30.0  # 30 seconds per growth stage
            if current_time - self.last_growth_time >= growth_time_required:
                self.growth_stage += 1
                self.last_growth_time = current_time

                # Only produce apples when mature enough
                if self.growth_stage >= 3:  # Stages 3-5 can produce apples
                    max_apples = min(10, (self.growth_stage - 2) * 2)
                    if self.quantity < max_apples:
                        self.quantity = max_apples

        # Apple regeneration: only when tree is mature enough
        if self.growth_stage >= 3 and self.quantity < self.max_quantity:
            if (
                current_time - self.last_regen_time >= utils_config.APPLE_REGEN_TIME
                and self.quantity < self.max_quantity
            ):  # 10 seconds
                self.quantity += 1
                self.last_regen_time = current_time

    def gather(self, amount):
        """
        Gather apples from the tree. Reduces the quantity of apples by the specified amount.
        If the tree becomes depleted, it is removed from the terrain.
        Only mature trees (stage 3+) can be foraged.
        :param amount: Number of apples to gather.
        :return: The number of apples actually gathered.
        """
        # Saplings (stages 0-2) have no apples to gather
        if self.growth_stage < 3:
            return 0

        if self.quantity > 0:
            gathered = min(amount, self.quantity)
            self.quantity -= gathered

            if self.is_depleted():
                """print(
                "\033[92m" +
                f"Apple Tree at ({self.grid_x}, {self.grid_y}) is now depleted. Removing from terrain." +
                "\033[0m")"""
                self.remove_from_terrain()

            return gathered
        else:
            # Final backup to remove resource if it still exists
            if self.is_depleted():
                print(
                    f"Apple Tree at ({self.grid_x}, {self.grid_y}) is depleted but still present. Removing from terrain."
                )
                self.remove_from_terrain()
            return 0

    def is_depleted(self):
        """
        Check if the tree is depleted of apples.
        :return: True if the tree has no apples left, False otherwise.
        """
        return self.quantity == 0

    def remove_from_terrain(self):
        """Remove the apple tree from the terrain and mark it as no longer available."""
        # print(
        #     f"Attempting to remove Apple Tree at ({self.grid_x}, {self.grid_y}). Quantity: {self.quantity}")

        # Ensure resource is actually depleted
        if self.quantity > 0:
            print(
                f"Error: Trying to remove Apple Tree at ({self.grid_x}, {self.grid_y}) before depletion."
            )
            return  # Abort removal if the resource is not depleted

        # Update the terrain grid
        self.terrain.grid[self.grid_x][self.grid_y]["occupied"] = False
        self.terrain.grid[self.grid_x][self.grid_y]["resource_type"] = None

        # Remove the tree from the resource manager
        if self in self.resource_manager.resources:
            self.resource_manager.resources.remove(self)
            self.resource_manager.apple_tree_count -= 1
            # print(
            #     "\033[92m" +
            #     f"Resource successfully removed: Apple Tree at ({self.grid_x}, {self.grid_y})" +
            #     "\033[0m")
        else:
            print(
                f"Apple Tree at ({self.grid_x}, {self.grid_y}) not found in resource manager."
            )

    def render(self, screen, camera):
        """
        Render the apple tree, aligning the stump with the grid cell and shifting it to the left by half its width.
        """
        # Skip rendering if headless mode is active
        if utils_config.HEADLESS_MODE:
            return

        # Calculate the screen position based on the camera
        screen_x, screen_y = camera.apply((self.x, self.y))

        # Use cached sprite sheet and extract current growth stage
        if AppleTree._sprite_sheet is None:
            return  # Can't render in headless mode
        sprite_sheet = AppleTree._sprite_sheet
        sprite_width = sprite_sheet.get_width() // 6
        sprite_height = sprite_sheet.get_height()

        # Extract sprite for current growth stage
        current_sprite = sprite_sheet.subsurface(
            pygame.Rect(
                sprite_width * self.growth_stage, 0, sprite_width, sprite_height
            )
        )

        # Calculate the final size of the tree
        final_size = int(
            utils_config.CELL_SIZE
            * utils_config.SCALING_FACTOR
            * camera.zoom
            * utils_config.Tree_Scale_Img
        )
        tree_image_scaled = pygame.transform.scale(
            current_sprite, (final_size, final_size)
        )

        # Offset to align stump
        offset_x = final_size // 2
        offset_y = final_size - (utils_config.CELL_SIZE * camera.zoom)

        # Draw tree image at corrected position
        screen.blit(tree_image_scaled, (screen_x - offset_x, screen_y - offset_y))
