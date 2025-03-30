import pygame
import random
import time
from utils_config import (
    CELL_SIZE, 
    GOLD_IMAGE_PATH, 
    TREE_IMAGE_PATH, 
    TREE_DENSITY, 
    GOLD_ZONE_PROBABILITY, 
    GOLD_SPAWN_DENSITY, 
    SCALING_FACTOR, 
    SCREEN_HEIGHT, 
    GoldLump_Scale_Img, 
    Tree_Scale_Img,
    APPLE_REGEN_TIME)

from utils_logger import TensorBoardLogger



class ResourceManager:
    def __init__(self, terrain):
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



    def generate_resources(self, add_trees=0, add_gold_lumps=0, tree_density=None, gold_zone_probability=None, gold_spawn_density=None, episode=0):
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
        tree_density = tree_density if tree_density is not None else TREE_DENSITY
        gold_zone_probability = gold_zone_probability if gold_zone_probability is not None else GOLD_ZONE_PROBABILITY
        gold_spawn_density = gold_spawn_density if gold_spawn_density is not None else GOLD_SPAWN_DENSITY

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
                if cell['type'] == 'land' and not cell['occupied']:
                    # Check bounds (prevent placing outside the world grid)
                    if not (0 <= x < grid_width and 0 <= y < grid_height):
                        continue

                    # Add specific number of trees
                    if add_trees > 0 and added_trees < add_trees:
                        self.resources.append(AppleTree(x * CELL_SIZE, y * CELL_SIZE, x, y, self.terrain, self))
                        cell['occupied'] = True
                        cell['resource_type'] = 'apple_tree'
                        self.apple_tree_count += 1
                        added_trees += 1
                        continue

                    # Add specific number of gold lumps
                    if add_gold_lumps > 0 and added_gold_lumps < add_gold_lumps:
                        gold_lump = GoldLump(
                            x * CELL_SIZE, y * CELL_SIZE, x, y,
                            self.terrain, self
                        )
                        self.resources.append(gold_lump)
                        cell['occupied'] = True
                        cell['resource_type'] = 'gold_lump'
                        self.gold_count += 1
                        added_gold_lumps += 1
                        continue

                    # Use density/probability for startup
                    if add_trees == 0 and random.random() < tree_density:
                        self.resources.append(AppleTree(x * CELL_SIZE, y * CELL_SIZE, x, y, self.terrain, self))
                        cell['occupied'] = True
                        cell['resource_type'] = 'apple_tree'
                        self.apple_tree_count += 1

                    elif add_gold_lumps == 0 and random.random() < gold_zone_probability:
                        # Generate gold lumps in a zone
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < grid_width and 0 <= ny < grid_height:  # Bounds check
                                    gold_cell = self.terrain.grid[nx][ny]
                                    if gold_cell['type'] == 'land' and not gold_cell['occupied']:
                                        if random.random() < gold_spawn_density:
                                            gold_lump = GoldLump(
                                                nx * CELL_SIZE, ny * CELL_SIZE, nx, ny,
                                                self.terrain, self
                                            )
                                            self.resources.append(gold_lump)
                                            gold_cell['occupied'] = True
                                            gold_cell['resource_type'] = 'gold_lump'
                                            self.gold_count += 1

        

        # Log to TensorBoard
        print (f"Generated {self.apple_tree_count} apple trees and {self.gold_count} gold lumps.")
        TensorBoardLogger().log_scalar("Resources/AppleTrees_Added", self.apple_tree_count, episode)
        TensorBoardLogger().log_scalar("Resources/GoldLumps_Added", self.gold_count, episode)

            

        


    #    _   _           _       _          __    _                    _  __       _            _      _           ___  
    #   | | | |_ __   __| | __ _| |_ ___   / /___| | ___  __ _ _ __   (_)/ _|   __| | ___ _ __ | | ___| |_ ___  __| \ \ 
    #   | | | | '_ \ / _` |/ _` | __/ _ \ | |/ __| |/ _ \/ _` | '_ \  | | |_   / _` |/ _ \ '_ \| |/ _ \ __/ _ \/ _` || |
    #   | |_| | |_) | (_| | (_| | ||  __/ | | (__| |  __/ (_| | | | | | |  _| | (_| |  __/ |_) | |  __/ ||  __/ (_| || |
    #    \___/| .__/ \__,_|\__,_|\__\___| | |\___|_|\___|\__,_|_| |_| |_|_|    \__,_|\___| .__/|_|\___|\__\___|\__,_|| |
    #         |_|                          \_\                                           |_|                        /_/ 



    def update(self):
        """Update resources, removing any that are depleted."""
        for resource in self.resources[:]:  # Iterate over a copy to avoid mutation during iteration
            if resource.is_depleted():
                print(f"Depleted resource detected at ({resource.grid_x}, {resource.grid_y}). Removing...")
                resource.remove_from_terrain()








#       _                _        _____                     _               
#      / \   _ __  _ __ | | ___  |_   _| __ ___  ___    ___| | __ _ ___ ___ 
#     / _ \ | '_ \| '_ \| |/ _ \   | || '__/ _ \/ _ \  / __| |/ _` / __/ __|
#    / ___ \| |_) | |_) | |  __/   | || | |  __/  __/ | (__| | (_| \__ \__ \
#   /_/   \_\ .__/| .__/|_|\___|   |_||_|  \___|\___|  \___|_|\__,_|___/___/
#           |_|   |_|                                                       



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
        self.terrain = terrain
        self.resource_manager = resource_manager
        self.max_quantity = quantity  # Store max quantity
        self.quantity =  random.randint(1, 6) # Number of apples
        self.last_regen_time = time.time()  # Track last regeneration time

        # Load the sprite sheet
        sprite_sheet = pygame.image.load(TREE_IMAGE_PATH).convert_alpha()

        # Dimensions of the sprite sheet and individual sprites
        sprite_width = sprite_sheet.get_width() // 6  # Assuming 6 frames horizontally
        sprite_height = sprite_sheet.get_height()

        # Extract the final tree (6th frame)
        self.image = sprite_sheet.subsurface(
            pygame.Rect(sprite_width * 5, 0, sprite_width, sprite_height)
        )
        self.update()

    def update(self):
        """Update the tree's apple quantity based on regeneration time."""
        current_time = time.time()
        if current_time - self.last_regen_time >= APPLE_REGEN_TIME and self.quantity < self.max_quantity:  # 10 seconds
            self.quantity += 1
            self.last_regen_time = current_time
            

    def gather(self, amount):
        """
        Gather apples from the tree. Reduces the quantity of apples by the specified amount.
        If the tree becomes depleted, it is removed from the terrain.
        :param amount: Number of apples to gather.
        :return: The number of apples actually gathered.
        """
        if self.quantity > 0:
            gathered = min(amount, self.quantity)
            self.quantity -= gathered
            print(f"Apple Tree at ({self.grid_x}, {self.grid_y}) foraged. Remaining: {self.quantity}")

            if self.is_depleted():
                print(f"Apple Tree at ({self.grid_x}, {self.grid_y}) is now depleted. Removing from terrain.")
                self.remove_from_terrain()

            return gathered
        else:
            # Final backup to remove resource if it still exists
            if self.is_depleted():
                print(f"Apple Tree at ({self.grid_x}, {self.grid_y}) is depleted but still present. Removing from terrain.")
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
        print(f"Attempting to remove Apple Tree at ({self.grid_x}, {self.grid_y}). Quantity: {self.quantity}")
        
        # Ensure resource is actually depleted
        if self.quantity > 0:
            print(f"Error: Trying to remove Apple Tree at ({self.grid_x}, {self.grid_y}) before depletion.")
            return  # Abort removal if the resource is not depleted

        # Update the terrain grid
        self.terrain.grid[self.grid_x][self.grid_y]['occupied'] = False
        self.terrain.grid[self.grid_x][self.grid_y]['resource_type'] = None

        # Remove the tree from the resource manager
        if self in self.resource_manager.resources:
            self.resource_manager.resources.remove(self)
            self.resource_manager.apple_tree_count -= 1
            print(f"Resource successfully removed: Apple Tree at ({self.grid_x}, {self.grid_y})")
        else:
            print(f"Apple Tree at ({self.grid_x}, {self.grid_y}) not found in resource manager.")

    def render(self, screen, camera):
        """
        Render the apple tree, aligning the stump with the grid cell and shifting it to the left by half its width.
        """
        # Calculate the screen position based on the camera's position and zoom
        screen_x = (self.x - camera.x) * camera.zoom
        screen_y = (self.y - camera.y) * camera.zoom

        # Calculate the final size of the tree, incorporating Tree_Scale_Img
        final_size = int(CELL_SIZE * SCALING_FACTOR * camera.zoom * Tree_Scale_Img)
        tree_image_scaled = pygame.transform.scale(self.image, (final_size, final_size))

        # Offset the image so the bottom (stump) aligns with the cell and shift left by half its width
        offset_x = final_size // 2  # Half the width
        offset_y = final_size - CELL_SIZE * camera.zoom  # Ensure alignment with the cell
        screen.blit(tree_image_scaled, (screen_x - offset_x, screen_y - offset_y))
        





#     ____       _     _   _                             ____ _               
#    / ___| ___ | | __| | | |   _   _ _ __ ___  _ __    / ___| | __ _ ___ ___ 
#   | |  _ / _ \| |/ _` | | |  | | | | '_ ` _ \| '_ \  | |   | |/ _` / __/ __|
#   | |_| | (_) | | (_| | | |__| |_| | | | | | | |_) | | |___| | (_| \__ \__ \
#    \____|\___/|_|\__,_| |_____\__,_|_| |_| |_| .__/   \____|_|\__,_|___/___/
#                                              |_|                            


class GoldLump:
    def __init__(self, x, y, grid_x, grid_y, terrain, resource_manager,quantity=5):
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
        self.quantity = random.randint(1, 20)  # Set quantity of gold
        self.resource_manager = resource_manager

        # Load and scale the gold image
        self.image = pygame.image.load(GOLD_IMAGE_PATH).convert_alpha()
        image_size = int(SCREEN_HEIGHT * 0.03)  # Scale to 3% of screen height
        self.image = pygame.transform.scale(self.image, (image_size, image_size))

    def mine(self, credit_callback=None):
        """Mine gold from the lump and credit it using the provided callback."""
        if self.quantity > 0:
            self.quantity -= 1
            print(f"Gold Lump at ({self.grid_x}, {self.grid_y}) mined. Remaining: {self.quantity}")

            if credit_callback:
                credit_callback(1)  # Credit the mined gold to the faction

            if self.is_depleted():
                print(f"Gold Lump at ({self.grid_x}, {self.grid_y}) is now depleted. Removing from terrain.")
                self.remove_from_terrain()
            return 1
        else:
            # Final backup to remove resource if it still exists
            if self.is_depleted():
                print(f"Gold Lump at ({self.grid_x}, {self.grid_y}) is depleted but still present. Removing from terrain.")
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
        print(f"Attempting to remove Gold Lump at ({self.grid_x}, {self.grid_y}). Quantity: {self.quantity}")
        
        # Ensure resource is actually depleted
        if self.quantity > 0:
            print(f"Error: Trying to remove Gold Lump at ({self.grid_x}, {self.grid_y}) before depletion.")
            return  # Abort removal if the resource is not depleted

        self.terrain.grid[self.grid_x][self.grid_y]['occupied'] = False
        self.terrain.grid[self.grid_x][self.grid_y]['resource_type'] = None
        if self in self.resource_manager.resources:
            self.resource_manager.resources.remove(self)
            print(f"Resource successfully removed: Gold Lump at ({self.grid_x}, {self.grid_y})")
        else:
            print(f"Gold Lump at ({self.grid_x}, {self.grid_y}) not found in resource manager.")





    def render(self, screen, camera):
        """
        Render the gold lump, centering it within the grid cell.
        """
        # Calculate the screen position based on the camera's position and zoom
        screen_x = (self.x - camera.x) * camera.zoom
        screen_y = (self.y - camera.y) * camera.zoom

        # Calculate the final size based on CELL_SIZE, SCALING_FACTOR, and camera zoom
        final_size = int(CELL_SIZE * SCALING_FACTOR * camera.zoom * GoldLump_Scale_Img)
        gold_image_scaled = pygame.transform.scale(self.image, (final_size, final_size))

        # Offset the image to center it within the grid cell
        offset_x = final_size // 2
        offset_y = final_size // 2
        screen.blit(gold_image_scaled, (screen_x - offset_x, screen_y - offset_y))






# class Boulder:
#     """ Sole purpose is to block movement and prevent units from passing through it. """
#     def __init__(self, grid_x, grid_y):
#         raise NotImplementedError("Boulder class is not implemented yet.")
#         self.grid_x = grid_x
#         self.grid_y = grid_y
#         self.x = grid_x * CELL_SIZE  # Convert grid to screen coordinates
#         self.y = grid_y * CELL_SIZE
#         self.width = CELL_SIZE
#         self.height = CELL_SIZE
        

#     def render(self, screen, camera):
#         """Render the Boulder object in the world."""
#         raise NotImplementedError("Boulder render method is not implemented yet.")
#         screen_x = (self.x - camera.x) * camera.zoom
#         screen_y = (self.y - camera.y) * camera.zoom
#         pygame.draw.rect(screen, (139, 69, 19), (screen_x, screen_y, self.width * camera.zoom, self.height * camera.zoom))  # Brown colour for boulders
