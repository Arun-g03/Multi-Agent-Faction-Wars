"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
from SHARED.core_imports import *
from RENDER.Game_Renderer import get_font
from RENDER.Common import GREEN, get_text_surface
import UTILITIES.utils_config as utils_config


class Terrain:
    def __init__(self):
        """

        Generate the terrain using Perlin noise and create a structured array for each cell.
        """
        self.grid = self.generate_noise_map(
            width=int(utils_config.WORLD_WIDTH // utils_config.CELL_SIZE),
            height=int(utils_config.WORLD_HEIGHT // utils_config.CELL_SIZE),
            scale=utils_config.NOISE_SCALE,
            octaves=utils_config.NOISE_OCTAVES,
            persistence=utils_config.NOISE_PERSISTENCE,
            lacunarity=utils_config.NOISE_LACUNARITY,
        )
        self.grid = self.smooth_noise_map(self.grid)
        # Calculate total traversable tiles (land tiles) for percentage calculations
        self.max_traversable_tiles = self.get_traversable_tile_count()

        # Cache for terrain rendering
        self._scaled_tiles = {}  # Cache scaled tiles by cell size
        self._last_cell_size = None
        self._base_textures_loaded = False

    def ensure_connected_land(self, grid, min_land_ratio=0.6):
        """
        Ensures that most of the land is in a single connected component.
        Disconnects small patches.
        """
        width, height = grid.shape
        land_mask = np.array([[cell["type"] == "land" for cell in row] for row in grid])

        # Label connected components
        labeled, num_features = label(land_mask)

        # Count size of each component
        counts = np.bincount(labeled.flatten())
        if len(counts) <= 1:
            return grid  # No land detected

        # Identify largest component
        largest_label = counts[1:].argmax() + 1

        # Keep only the largest landmass
        for i in range(width):
            for j in range(height):
                if labeled[i, j] != largest_label:
                    grid[i][j]["type"] = "water"

        return grid

    #                                    _                         _ _                     _
    #     __ _  ___ _ __   ___ _ __ __ _| |_ ___   _ __   ___ _ __| (_)_ __    _ __   ___ (_)___  ___
    #    / _` |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \ | '_ \ / _ \ '__| | | '_ \  | '_ \ / _ \| / __|/ _ \
    #   | (_| |  __/ | | |  __/ | | (_| | ||  __/ | |_) |  __/ |  | | | | | | | | | | (_) | \__ \  __/
    #    \__, |\___|_| |_|\___|_|  \__,_|\__\___| | .__/ \___|_|  |_|_|_| |_| |_| |_|\___/|_|___/\___|
    #    |___/                                    |_|

    def generate_noise_map(
        self,
        width,
        height,
        scale,
        octaves,
        persistence,
        lacunarity,
        random=utils_config.RandomiseTerrainBool,
    ):
        """
        Generate Perlin noise-based terrain with structured data for each cell.
        Regenerates if the resulting terrain has poor land connectivity.
        """

        dtype = [
            ("type", "U10"),  # 'land' or 'water'
            ("occupied", "bool"),  # True if occupied by an agent
            ("faction", "U10"),  # Faction name or ID
            ("resource_type", "U15"),  # e.g., 'apple_tree', 'gold_lump'
        ]
        grid = np.zeros((width, height), dtype=dtype)
        x_indices = np.linspace(0, width / scale, width)
        y_indices = np.linspace(0, height / scale, height)
        nx, ny = np.meshgrid(x_indices, y_indices)

        max_retries = 500
        attempts = 0
        land_ratio = 0

        while land_ratio < 0.6 and attempts <= max_retries:
            base_seed = (
                np.random.randint(0, 1000) if random else utils_config.Terrain_Seed
            )
            noise_map = np.zeros((width, height))

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
                        base=base_seed,
                    )

            threshold = np.percentile(noise_map, utils_config.WATER_COVERAGE * 100)

            for i in range(width):
                for j in range(height):
                    grid[i][j]["type"] = (
                        "land" if noise_map[i][j] >= threshold else "water"
                    )
                    grid[i][j]["occupied"] = False
                    grid[i][j]["faction"] = "None"
                    grid[i][j]["resource_type"] = "None"

            grid = self.ensure_connected_land(grid)

            land_tiles = np.count_nonzero(grid["type"] == "land")
            total_tiles = width * height
            land_ratio = land_tiles / total_tiles
            attempts += 1

        if land_ratio < 0.6:
            print(
                f"[WARNING] Terrain generation produced only {land_ratio:.2%} land after {attempts} attempts."
            )

        return grid

    def smooth_noise_map(self, grid):
        """
        Smooth the terrain using Gaussian filtering.
        """
        elevation_map = np.array(
            [[cell["type"] == "land" for cell in row] for row in grid]
        )  # Binary map of land vs. water
        smoothed_map = gaussian_filter(elevation_map.astype(float), sigma=1)

        # Reassign the smoothed terrain back to the grid type
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if smoothed_map[i][j] < 0.5:
                    grid[i][j]["type"] = "water"
                else:
                    grid[i][j]["type"] = "land"

        return grid

    def get_traversable_tile_count(self):
        """
        Count the total number of traversable (land) tiles in the world.
        This is used to calculate ownership percentages.
        """
        land_count = np.count_nonzero(self.grid["type"] == "land")
        return land_count

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

        # Initialise Pygame font for rendering text
        # Adjust font size to cell size
        font = get_font(int(utils_config.CELL_SIZE * 0.5))

        # Initialize cache on first call or when cell size changes
        cell_size_int = int(utils_config.CELL_SIZE)
        if cell_size_int != self._last_cell_size or not self._base_textures_loaded:
            self._last_cell_size = cell_size_int
            self._base_textures_loaded = True
            self._scaled_tiles = {}

            # Load base textures once
            self._base_grass_texture = pygame.image.load(
                utils_config.Grass_Texture_Path
            ).convert_alpha()

            # Load water textures
            self._base_water_sheet = pygame.image.load(
                utils_config.Water_Texture_Path
            ).convert_alpha()

            # Pre-scale grass texture
            self._scaled_tiles["land"] = pygame.transform.scale(
                self._base_grass_texture, (cell_size_int, cell_size_int)
            )

            # Pre-scale water frames
            frame_width = 16
            frame_height = 16
            self._water_frames = []
            for i in range(3, 6):  # Frames 3, 4, 5
                frame = self._base_water_sheet.subsurface(
                    (i * frame_width, 2 * frame_height, frame_width, frame_height)
                )
                self._scaled_tiles[f"water_{i-3}"] = pygame.transform.scale(
                    frame, (cell_size_int, cell_size_int)
                )
                self._water_frames.append(self._scaled_tiles[f"water_{i-3}"])

        # Determine the number of frames to display per cycle and the time
        # interval between frames
        frame_duration = 200  # Time per frame (in milliseconds)
        # Cycle through frames
        current_frame = (pygame.time.get_ticks() // frame_duration) % len(
            self._water_frames
        )

        for x in range(grid_width):
            for y in range(grid_height):
                world_x = (x * utils_config.CELL_SIZE) - camera.x * camera.zoom
                world_y = (y * utils_config.CELL_SIZE) - camera.y * camera.zoom

                cell = self.grid[x][y]

                if cell["type"] == "water":
                    # Use cached scaled water frame
                    play_animation = utils_config.WaterAnimationToggle
                    if play_animation:
                        screen.blit(
                            self._water_frames[current_frame], (world_x, world_y)
                        )
                    else:
                        screen.blit(self._water_frames[0], (world_x, world_y))
                elif cell["type"] == "land":
                    # Use cached scaled grass texture
                    grass_texture = self._scaled_tiles["land"]

                    # Create tint surface (cache this too if needed)
                    if "tint_surface" not in self._scaled_tiles:
                        tint_surface = pygame.Surface(
                            (cell_size_int, cell_size_int),
                            pygame.SRCALPHA,
                        )
                        tint_colour = (0, 160, 0, 255)
                        tint_surface.fill(tint_colour)
                        self._scaled_tiles["tint_surface"] = tint_surface

                    # Apply tint (only if not already tinted in cache)
                    if "land_tinted" not in self._scaled_tiles:
                        tinted_grass = self._scaled_tiles["land"].copy()
                        tinted_grass.blit(
                            self._scaled_tiles["tint_surface"],
                            (0, 0),
                            special_flags=pygame.BLEND_RGBA_MULT,
                        )
                        self._scaled_tiles["land_tinted"] = tinted_grass

                    screen.blit(self._scaled_tiles["land_tinted"], (world_x, world_y))

                    # If the cell is owned by a faction, overlay the faction's
                    # colour with some transparency
                    if cell["faction"] != "None":
                        faction_colour = faction_colours.get(
                            cell["faction"], (0, 255, 0)
                        )  # Default to green
                        overlay = pygame.Surface(
                            (int(utils_config.CELL_SIZE), int(utils_config.CELL_SIZE)),
                            pygame.SRCALPHA,
                        )
                        # Semi-transparent overlay
                        overlay.fill((*faction_colour, 128))
                        screen.blit(overlay, (world_x, world_y))
                else:
                    # Default fallback for other cell types (e.g., green for
                    # undefined cells)
                    rect = pygame.Rect(
                        world_x, world_y, utils_config.CELL_SIZE, utils_config.CELL_SIZE
                    )
                    pygame.draw.rect(screen, GREEN, rect)

                # Render faction ID only if render_ids is True
                if render_ids:
                    faction_id = cell["faction"] if cell["faction"] != "None" else "0"
                    text = get_text_surface(
                        str(faction_id), "arial", 12, (255, 255, 255)
                    )  # White text
                    text_rect = text.get_rect(
                        center=(
                            world_x + utils_config.CELL_SIZE // 2,
                            world_y + utils_config.CELL_SIZE // 2,
                        )
                    )
                    screen.blit(text, text_rect)

        """ # Draw gridlines
        for x in range(grid_width + 1):
            start_x = (x * utils_config.CELL_SIZE) - camera.x * camera.zoom
            pygame.draw.line(screen, (255, 255, 255), (start_x, 0), (start_x, utils_config.SCREEN_HEIGHT), 1)  # Vertical lines

        for y in range(grid_height + 1):
            start_y = (y * utils_config.CELL_SIZE) - camera.y * camera.zoom
            pygame.draw.line(screen, (255, 255, 255), (0, start_y), (SCREEN_WIDTH, start_y), 1)  # Horizontal lines
        """
