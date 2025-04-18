import pygame
import noise  #Noise is a package for perlin noise. It helps to generate the landscape.
import numpy as np
from scipy.ndimage import gaussian_filter
import utils_config

from camera import Camera

# colours


class Terrain:
    def __init__(self):
        """
        
        Generate the terrain using Perlin noise and create a structured array for each cell.
        """
        self.grid = self.generate_noise_map(width = int(utils_config.WORLD_WIDTH // utils_config.CELL_SIZE),
                                            height = int(utils_config.WORLD_HEIGHT // utils_config.CELL_SIZE),
                                            scale=utils_config.NOISE_SCALE, 
                                            octaves=utils_config.NOISE_OCTAVES, 
                                            persistence=utils_config.NOISE_PERSISTENCE, 
                                            lacunarity=utils_config.NOISE_LACUNARITY)
        self.grid = self.smooth_noise_map(self.grid)
        


    
    #                                    _                         _ _                     _          
    #     __ _  ___ _ __   ___ _ __ __ _| |_ ___   _ __   ___ _ __| (_)_ __    _ __   ___ (_)___  ___ 
    #    / _` |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \ | '_ \ / _ \ '__| | | '_ \  | '_ \ / _ \| / __|/ _ \
    #   | (_| |  __/ | | |  __/ | | (_| | ||  __/ | |_) |  __/ |  | | | | | | | | | | (_) | \__ \  __/
    #    \__, |\___|_| |_|\___|_|  \__,_|\__\___| | .__/ \___|_|  |_|_|_| |_| |_| |_|\___/|_|___/\___|
    #    |___/                                    |_|                                                 

    def generate_noise_map(self, width, height, scale, octaves, persistence, lacunarity, random=utils_config.RandomiseTerrainBool):
        """
        Generate Perlin noise-based terrain with structured data for each cell.
        """

                
        #    ____                  _  __                  _ _       _       _        
        #   / ___| _ __   ___  ___(_)/ _|_   _    ___ ___| | |   __| | __ _| |_ __ _ 
        #   \___ \| '_ \ / _ \/ __| | |_| | | |  / __/ _ \ | |  / _` |/ _` | __/ _` |
        #    ___) | |_) |  __/ (__| |  _| |_| | | (_|  __/ | | | (_| | (_| | || (_| |
        #   |____/| .__/ \___|\___|_|_|  \__, |  \___\___|_|_|  \__,_|\__,_|\__\__,_|
        #         |_|                    |___/                                       

        # Define a structured NumPy array with fields 'type', 'occupied', 'faction', and 'resource_type'
        dtype = [('type', 'U10'),      # 'land' or 'water'
                ('occupied', 'bool'),  # True if the cell is occupied by an agent
                ('faction', 'U10'),    # Name or ID of the faction occupying the cell, None if unoccupied
                ('resource_type', 'U15')]  # Resource type (e.g., 'apple_tree', 'gold_lump'), None if no resource
        grid = np.zeros((width, height), dtype=dtype)

        # Generate Perlin noise for terrain elevation
        x_indices = np.linspace(0, width / scale, width)
        y_indices = np.linspace(0, height / scale, height)
        nx, ny = np.meshgrid(x_indices, y_indices)

        noise_map = np.zeros((width, height))
        base_seed = np.random.randint(0, 1000) if random else utils_config.Terrain_Seed
        for i in range(width):
            for j in range(height):
                noise_map[i][j] = noise.pnoise2(
                                                nx[i][j], 
                                                ny[i][j], 
                                                octaves=octaves, 
                                                persistence=persistence, 
                                                lacunarity=lacunarity, 
                                                repeatx=1024, 
                                                repeaty=1024, 
                                                base=base_seed)

        # Apply water threshold
        threshold = np.percentile(noise_map, utils_config.WATER_COVERAGE * 100)
        
        for i in range(width):
            for j in range(height):
                if noise_map[i][j] < threshold:  # Water
                    grid[i][j]['type'] = 'water'
                else:  # Land
                    grid[i][j]['type'] = 'land'
                grid[i][j]['occupied'] = False
                grid[i][j]['faction'] = 'None'
                grid[i][j]['resource_type'] = 'None'

        return grid    


    def smooth_noise_map(self, grid):
        """
        Smooth the terrain using Gaussian filtering.
        """
        elevation_map = np.array([[cell['type'] == 'land' for cell in row] for row in grid])  # Binary map of land vs. water
        smoothed_map = gaussian_filter(elevation_map.astype(float), sigma=1)

        # Reassign the smoothed terrain back to the grid type
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if smoothed_map[i][j] < 0.5:
                    grid[i][j]['type'] = 'water'
                else:
                    grid[i][j]['type'] = 'land'

        return grid
    
    
    #    ____                       _                      _       
    #   |  _ \ _ __ __ ___      __ | |_ ___ _ __ _ __ __ _(_)_ __  
    #   | | | | '__/ _` \ \ /\ / / | __/ _ \ '__| '__/ _` | | '_ \ 
    #   | |_| | | | (_| |\ V  V /  | ||  __/ |  | | | (_| | | | | |
    #   |____/|_|  \__,_| \_/\_/    \__\___|_|  |_|  \__,_|_|_| |_|
    #                                                              

    def draw(self, screen, camera, factions, render_ids=False):
        """
        Draw the terrain grid on the screen, with zoom adjustments from the camera, 
        and optionally display faction IDs.

        :param factions: List of all factions to map faction IDs to colours.
        :param render_ids: Boolean to toggle rendering of faction IDs on cells.
        """
        utils_config.CELL_SIZE = utils_config.CELL_SIZE * camera.zoom
        grid_width = len(self.grid)
        grid_height = len(self.grid[0])

        # Create a mapping of faction IDs to their RGB colours
        faction_colours = {str(faction.id): faction.colour for faction in factions}
        
        from render_display import get_font

        # Initialise Pygame font for rendering text
        font = get_font(int(utils_config.CELL_SIZE * 0.5))  # Adjust font size to cell size

        # Load the grass texture
        grass_texture = pygame.image.load(utils_config.Grass_Texture_Path).convert_alpha()

        # Load water textures (animate water)
        water_sprite_sheet = pygame.image.load(utils_config.Water_Texture_Path).convert_alpha()

        # Define the frames for the water animation (frames 3, 4, and 5)
        water_frames = []
        frame_width = 16  # width of each frame (192 / 12)
        frame_height = 16  # height of each frame (224 / 14)

        # Extract frames 3, 4, and 5 (0-indexed, so frames 2, 3, and 4 in terms of index)
        water_frames.append(water_sprite_sheet.subsurface((2 * frame_width, 2 * frame_height, frame_width, frame_height)))  # Frame 3
        water_frames.append(water_sprite_sheet.subsurface((3 * frame_width, 2 * frame_height, frame_width, frame_height)))  # Frame 4
        water_frames.append(water_sprite_sheet.subsurface((4 * frame_width, 2 * frame_height, frame_width, frame_height)))  # Frame 5

        # Determine the number of frames to display per cycle and the time interval between frames
        frame_duration = 200  # Time per frame (in milliseconds)
        current_frame = (pygame.time.get_ticks() // frame_duration) % len(water_frames)  # Cycle through frames

        for x in range(grid_width):
            for y in range(grid_height):
                world_x = (x * utils_config.CELL_SIZE) - camera.x * camera.zoom
                world_y = (y * utils_config.CELL_SIZE) - camera.y * camera.zoom

                cell = self.grid[x][y]

                if cell['type'] == 'water':
                    # Animate the water texture and apply on screen
                    
                    play_animation = utils_config.WaterAnimationToggle  # bool to toggle the animation
                    if play_animation:
                        frame = water_frames[current_frame]
                        scaled_frame = pygame.transform.scale(frame, (int(utils_config.CELL_SIZE), int(utils_config.CELL_SIZE)))
                        screen.blit(scaled_frame, (world_x, world_y))                
                    else:
                        static_frame = water_frames[0]  # Use the first frame as static
                        scaled_static_frame = pygame.transform.scale(static_frame, (int(utils_config.CELL_SIZE), int(utils_config.CELL_SIZE)))
                        screen.blit(scaled_static_frame, (world_x, world_y))                
                elif cell['type'] == 'land':
                    # Scale the grass texture to fit the cell size
                    scaled_grass_texture = pygame.transform.scale(grass_texture, (int(utils_config.CELL_SIZE), int(utils_config.CELL_SIZE)))

                    # Create a tint surface and scale it to fit the cell size
                    tint_surface = pygame.Surface((int(utils_config.CELL_SIZE), int(utils_config.CELL_SIZE)), pygame.SRCALPHA)
                    tint_colour = (0,160,0,255)  # Adjust tint colour and transparency (R, G, B, A)
                    tint_surface.fill(tint_colour)

                    # Apply the tint by blitting the tint surface onto the scaled texture
                    tinted_grass_texture = scaled_grass_texture.copy()
                    tinted_grass_texture.blit(tint_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

                    # Draw the tinted grass texture on the screen
                    screen.blit(tinted_grass_texture, (world_x, world_y))

                    # If the cell is owned by a faction, overlay the faction's colour with some transparency
                    if cell['faction'] != 'None':
                        faction_colour = faction_colours.get(cell['faction'], (0, 255, 0))  # Default to green
                        overlay = pygame.Surface((int(utils_config.CELL_SIZE), int(utils_config.CELL_SIZE)), pygame.SRCALPHA)
                        overlay.fill((*faction_colour, 128))  # Semi-transparent overlay
                        screen.blit(overlay, (world_x, world_y))
                else:
                    # Default fallback for other cell types (e.g., green for undefined cells)
                    rect = pygame.Rect(world_x, world_y, utils_config.CELL_SIZE, utils_config.CELL_SIZE)
                    pygame.draw.rect(screen, GREEN, rect)

                # Render faction ID only if render_ids is True
                if render_ids:
                    faction_id = cell['faction'] if cell['faction'] != 'None' else '0'
                    text = font.render(str(faction_id), True, (255, 255, 255))  # White text
                    text_rect = text.get_rect(center=(world_x + utils_config.CELL_SIZE // 2, world_y + utils_config.CELL_SIZE // 2))
                    screen.blit(text, text_rect)



        """ # Draw gridlines
        for x in range(grid_width + 1):
            start_x = (x * utils_config.CELL_SIZE) - camera.x * camera.zoom
            pygame.draw.line(screen, (255, 255, 255), (start_x, 0), (start_x, utils_config.SCREEN_HEIGHT), 1)  # Vertical lines

        for y in range(grid_height + 1):
            start_y = (y * utils_config.CELL_SIZE) - camera.y * camera.zoom
            pygame.draw.line(screen, (255, 255, 255), (0, start_y), (SCREEN_WIDTH, start_y), 1)  # Horizontal lines
        """