import pygame
import utils_config
from env_resources import AppleTree, GoldLump
import sys
import traceback
import logging
from utils_logger import Logger
logger = Logger(log_file="Renderer.txt", log_level=logging.DEBUG)


# Constants for the menu screen
FONT_NAME = "Arial"


# Define some reusable colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
GREEN = (50, 255, 50)
RED = (255, 50, 50)
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)


#    ____                _
#   |  _ \ ___ _ __   __| | ___ _ __ ___ _ __
#   | |_) / _ \ '_ \ / _` |/ _ \ '__/ _ \ '__|
#   |  _ <  __/ | | | (_| |  __/ | |  __/ |
#   |_| \_\___|_| |_|\__,_|\___|_|  \___|_|
#

pygame.init()
pygame.font.init()

FONT_CACHE = {}


def get_font(size):
    """
    Retrieves a font from cache or creates a new one if not cached.
    """
    if size not in FONT_CACHE:
        FONT_CACHE[size] = pygame.font.SysFont("Bahnschrift", size)
    return FONT_CACHE[size]


class GameRenderer:
    def __init__(
            self,
            screen,
            terrain,
            resources,
            factions,
            agents,
            camera=None):
        pygame.font.init()  # Initialise the font module
        self.screen = screen  # Use the screen passed as argument
        pygame.display.set_caption(
            'Multi-agent competitive and cooperative strategy (MACCS) - Simulation')
        # Use cached font for displaying faction IDs on bases
        self.font = get_font(24)
        self.attack_sprite_sheet = pygame.image.load(
            "images/Attack_Animation.png").convert_alpha()
        self.attack_frames = self.load_frames(
            self.attack_sprite_sheet, frame_width=64, frame_height=64)
        self.camera = camera  # Set camera if passed, or use None by default
        self.active_animations = []
        self.faction_base_image = pygame.image.load(
            "images/castle-7440761_1280.png").convert_alpha()

    def render(
            self,
            camera,
            terrain,
            resources,
            factions,
            agents,
            episode,
            step,
            resource_counts,
            enable_cell_tooltip=True):
        """
        Render the entire game state, including terrain, resources, home bases, and agents.
        Display tooltips for resources or agents if the mouse hovers over them.
        Optionally, show the grid cell under the mouse cursor relative to the world coordinates.
        """
        try:
            # Clear the screen with black background
            self.screen.fill((0, 0, 0))
            self.update_animations()

            # Render terrain, passing factions for territory visualisation
            terrain.draw(self.screen, camera, factions)

            # Get mouse position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            tooltip_text = None  # To hold tooltip text if hovering over a resource or agent

            # Render resources and check for hover
            for resource in resources:
                if not hasattr(resource, 'x') or not hasattr(resource, 'y'):
                    print(
                        f"Warning: Resource {resource} does not have 'x' and 'y' attributes. Skipping render.")
                    continue  # Skip this resource if invalid
                resource.render(self.screen, camera)  # Render the resource

                # Recalculate the resource's screen position based on the
                # camera
                screen_x = (resource.x - camera.x) * camera.zoom
                screen_y = (resource.y - camera.y) * camera.zoom

                # Calculate the size of the resource for collision detection
                if isinstance(resource, AppleTree):
                    final_size = int(
                        utils_config.CELL_SIZE *
                        utils_config.SCALING_FACTOR *
                        camera.zoom *
                        utils_config.Tree_Scale_Img)
                elif isinstance(resource, GoldLump):
                    final_size = int(
                        utils_config.CELL_SIZE *
                        utils_config.SCALING_FACTOR *
                        camera.zoom *
                        utils_config.GoldLump_Scale_Img)
                else:
                    final_size = utils_config.CELL_SIZE  # Fallback for unknown resources

                # Create a rectangle for resource collision detection
                resource_rect = pygame.Rect(
                    screen_x - final_size // 2,  # Center the rectangle on the resource
                    screen_y - final_size // 2,
                    final_size,
                    final_size,
                )

                # Check if the mouse is over the resource
                if resource_rect.collidepoint(mouse_x, mouse_y):
                    # Use grid_x and grid_y directly for the tooltip
                    tooltip_text = (
                        f"Type: {type(resource).__name__}\n"
                        f"Quantity: {resource.quantity}\n"
                        f"Position: ({resource.grid_x}, {resource.grid_y})"
                    )

            # Render home bases and agents
            self.render_home_bases(factions, camera, mouse_x, mouse_y)
            self.render_agents(agents, camera)

            # Render tooltip if any
            if tooltip_text:
                self.render_tooltip(tooltip_text)

            # Render mouse cell tooltip if enabled
            if enable_cell_tooltip:
                # Calculate world-relative grid cell under the mouse
                world_mouse_x = (mouse_x / camera.zoom) + camera.x
                world_mouse_y = (mouse_y / camera.zoom) + camera.y

                # Convert world coordinates to grid coordinates
                mouse_grid_x = int(world_mouse_x // utils_config.CELL_SIZE)
                mouse_grid_y = int(world_mouse_y // utils_config.CELL_SIZE)

                # Create the tooltip text for the world-relative cell and
                # screen position
                cell_tooltip_text = (
                    f"Screen Pos: ({mouse_x}, {mouse_y})\n"  # Screen position
                    # Grid coordinates
                    f"World Cell: ({mouse_grid_x}, {mouse_grid_y})"
                )

                # Calculate bottom-left position for the tooltip
                tooltip_x = 10  # Offset from the left edge
                # 50px offset from the bottom edge (adjust based on tooltip
                # size)
                tooltip_y = self.screen.get_height() - 50

                # Render the tooltip using the existing tooltip function
                self.render_tooltip(cell_tooltip_text,
                                    position=(tooltip_x, tooltip_y))

            #  Render the HUD with episode and step data
            self.render_hud(self.screen, episode, step,
                            factions, resource_counts)

            #  Ensure Pygame updates the display

            # Return True to indicate successful rendering
            return True

        except Exception as e:
            print(f"An error occurred in the render method: {e}")
            import traceback
            traceback.print_exc()
            return False  # Return False to signal the rendering loop should stop


#    _   _                      _                       ___   _  ___
#   | | | | ___  _ __ ___   ___| |__   __ _ ___  ___   / / | | |/ _ \
#   | |_| |/ _ \| '_ ` _ \ / _ \ '_ \ / _` / __|/ _ \ / /| |_| | | | |
#   |  _  | (_) | | | | | |  __/ |_) | (_| \__ \  __// / |  _  | |_| |
#   |_| |_|\___/|_| |_| |_|\___|_.__/ \__,_|___/\___/_/  |_| |_|\__\_\
#


    def render_home_bases(self, factions, camera, mouse_x, mouse_y):
        for faction in factions:
            if faction.home_base["position"]:
                x, y = faction.home_base["position"]
                size = faction.home_base["size"]
                colour = faction.home_base["colour"]

                # Adjust position and scale based on camera
                screen_x, screen_y = camera.apply((x, y))
                adjusted_size = size * camera.zoom

                # Draw the home base square
                # Load and scale the PNG image for the faction base
                base_image_scaled = pygame.transform.scale(
                    self.faction_base_image, (int(adjusted_size), int(adjusted_size)))
                # Blit the base image instead of drawing a square
                self.screen.blit(base_image_scaled, (screen_x, screen_y))

                # Display the faction ID at the center of the base
                text = self.font.render(str(faction.id), True, (0, 0, 0))
                text_rect = text.get_rect(
                    center=(
                        screen_x +
                        adjusted_size //
                        2 -
                        12,
                        screen_y +
                        adjusted_size //
                        2))
                self.screen.blit(text, text_rect)
                # Highlight resources and threats known to the faction if
                # hovering over the base
                base_rect = pygame.Rect(
                    screen_x, screen_y, adjusted_size, adjusted_size)
                if base_rect.collidepoint(mouse_x, mouse_y):

                    # Highlight resources
                    for resource in faction.global_state["resources"]:
                        resource_screen_x = (
                            resource["location"][0] * utils_config.CELL_SIZE - camera.x) * camera.zoom
                        resource_screen_y = (
                            resource["location"][1] * utils_config.CELL_SIZE - camera.y) * camera.zoom

                        pygame.draw.rect(
                            self.screen,
                            (0, 255, 0),  # Green colour for highlighting resources
                            pygame.Rect(
                                resource_screen_x - utils_config.CELL_SIZE * camera.zoom // 2,
                                resource_screen_y - utils_config.CELL_SIZE * camera.zoom // 2,
                                utils_config.CELL_SIZE * camera.zoom,
                                utils_config.CELL_SIZE * camera.zoom
                            ),
                            width=2
                        )

                    # Highlight threats
                    for threat in faction.global_state["threats"]:
                        try:
                            if isinstance(
                                    threat, dict) and "location" in threat:
                                threat_x, threat_y = threat["location"]
                            else:
                                continue  # Skip invalid threats

                            # Convert grid coordinates to screen coordinates
                            threat_screen_x = (
                                threat_x - camera.x) * camera.zoom
                            threat_screen_y = (
                                threat_y - camera.y) * camera.zoom

                            # Draw the threat box
                            box_size = utils_config.CELL_SIZE * camera.zoom
                            pygame.draw.rect(
                                self.screen,
                                (255, 0, 0),  # Red colour for threats
                                pygame.Rect(
                                    threat_screen_x - box_size // 2,
                                    threat_screen_y - box_size // 2,
                                    box_size,
                                    box_size
                                ),
                                width=2  # Border thickness
                            )
                        except Exception as e:
                            print(f"Error highlighting threat: {e}")
                            continue

                    # Gather faction metrics
                    resource_counts = {
                        "apple_trees": sum(
                            1 for res in faction.global_state["resources"] if res.get("type") == "AppleTree"),
                        "gold_lumps": sum(
                            1 for res in faction.global_state["resources"] if res.get("type") == "GoldLump"),
                    }
                    threat_count = len(faction.global_state["threats"])
                    peacekeeper_count = sum(
                        1 for agent in faction.agents if agent.role == "peacekeeper")
                    gatherer_count = sum(
                        1 for agent in faction.agents if agent.role == "gatherer")

                    # Display all metrics in the tooltip

                    overlay_text = (
                        f"Faction ID: {faction.id}\n\n"
                        f"Known Entities:\n"
                        f"Apple Trees: {resource_counts['apple_trees']}\n"
                        f"Gold Lumps: {resource_counts['gold_lumps']}\n"
                        f"Threats: {threat_count}\n\n"
                        f"Faction Metrics:\n"
                        f"Gold: {faction.gold_balance}\n"
                        f"Food: {faction.food_balance}\n"
                        f"Peacekeepers: {peacekeeper_count}\n"
                        f"Gatherers: {gatherer_count}\n\n"
                        f"Current Strategy: {faction.current_strategy}"
                    )
                    self.render_tooltip(overlay_text)


#    _   _ _   _ ____
#   | | | | | | |  _ \
#   | |_| | | | | | | |
#   |  _  | |_| | |_| |
#   |_| |_|\___/|____/
#


    def render_hud(self, screen, episode, step, factions, resource_counts):
        font = get_font(20)
        lines = []

        # Episode and Step
        lines.append(
            font.render(
                f"Episode: {episode}/{utils_config.EPISODES_LIMIT} | Step: {step} /{utils_config.STEPS_PER_EPISODE}",
                True,
                (255,
                 255,
                 255)))

        # Time
        elapsed_time = pygame.time.get_ticks() // 1000
        hours, minutes, seconds = elapsed_time // 3600, (elapsed_time %
                                                         3600) // 60, elapsed_time % 60
        lines.append(
            font.render(
                f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
                True,
                (255,
                 255,
                 255)))

        # Resources
        lines.append(font.render(
            "Resources available:", True, (255, 255, 255)))
        lines.append(
            font.render(
                f"Gold Lumps: {resource_counts.get('gold_lumps', 0)}, Gold: {resource_counts.get('gold_quantity', 0)}",
                True,
                (255,
                 255,
                 255)))
        lines.append(
            font.render(
                f"Apple Trees: {resource_counts.get('apple_trees', 0)}, Apples: {resource_counts.get('apple_quantity', 0)}",
                True,
                (255,
                 255,
                 255)))

        # Leaderboard
        lines.append(font.render(
            "Faction Leaderboard:", True, (255, 255, 255)))
        sorted_factions = sorted(
            factions, key=lambda f: f.gold_balance, reverse=True)
        for faction in sorted_factions[:4]:
            lines.append(
                font.render(
                    f"Faction {faction.id} - Gold: {faction.gold_balance}, Food: {faction.food_balance}, Agents: {len(faction.agents)}",
                    True,
                    (255,
                     255,
                     255)))

        # 🧮 Compute background size dynamically
        padding = 10
        line_height = font.get_height() + 4
        max_width = max(line.get_width() for line in lines)
        total_height = len(lines) * line_height

        # 🎨 Create dynamic HUD background
        hud_background = pygame.Surface(
            (max_width + 2 * padding, total_height + 2 * padding))
        hud_background.set_alpha(128)
        hud_background.fill((0, 0, 0))
        screen.blit(hud_background, (10, 10))

        # 🖼 Draw text lines
        for i, line in enumerate(lines):
            screen.blit(line, (10 + padding, 10 + padding + i * line_height))

        pygame.display.update()

#       _                    _
#      / \   __ _  ___ _ __ | |_ ___
#     / _ \ / _` |/ _ \ '_ \| __/ __|
#    / ___ \ (_| |  __/ | | | |_\__ \
#   /_/   \_\__, |\___|_| |_|\__|___/
#           |___/

    def render_agents(self, agents, camera):
        """
        Render each agent and display a tooltip if the mouse hovers over an agent.
        Ensure proper handling of target types for gatherers and peacekeepers.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for agent in agents:
            if not hasattr(agent, 'x') or not hasattr(agent, 'y'):
                print(
                    f"Warning: Agent {agent} does not have 'x' and 'y' attributes. Skipping render.")
                continue

            screen_x, screen_y = camera.apply((agent.x, agent.y))
            agent_sprite = agent.sprite

            if agent_sprite:
                agent_rect = agent_sprite.get_rect(
                    center=(int(screen_x), int(screen_y)))
                self.screen.blit(agent_sprite, agent_rect)

                if agent_rect.collidepoint(mouse_x, mouse_y):

                    # Field of View
                    pygame.draw.circle(
                        self.screen,
                        (255,
                         255,
                         0),
                        (screen_x,
                         screen_y),
                        utils_config.Agent_field_of_view *
                        utils_config.CELL_SIZE *
                        camera.zoom,
                        width=2)

                    # Interact Range
                    pygame.draw.circle(
                        self.screen,
                        (173,
                         216,
                         230),
                        (screen_x,
                         screen_y),
                        utils_config.Agent_Interact_Range *
                        utils_config.CELL_SIZE *
                        camera.zoom,
                        width=2)

                    # 🎯 Highlight task target
                    if isinstance(agent.current_task, dict):
                        task_type = agent.current_task.get("type", "unknown")
                        target = agent.current_task.get("target")

                        if target:
                            if utils_config.ENABLE_LOGGING:
                                logger.log_msg(
                                    f"[HUD TARGET] Agent {agent.agent_id} Task -> Type: {task_type}, Target: {target}",
                                    level=logging.DEBUG)

                            target_position = target.get("position")
                            target_type = target.get("type")
                            target_x = target_y = None

                            if task_type == "eliminate" and target_type == "agent" and "id" in target:
                                target_id = target["id"]
                                target_agent = next(
                                    (a for a in agents if a.agent_id == target_id), None)

                                if target_agent:
                                    target_x, target_y = target_agent.x, target_agent.y
                                else:
                                    if utils_config.ENABLE_LOGGING:
                                        logger.log_msg(
                                            f"[WARN] Target agent {target_id} not found.", level=logging.WARNING)

                            elif target_position:
                                target_x, target_y = target_position
                                target_x *= utils_config.CELL_SIZE
                                target_y *= utils_config.CELL_SIZE

                            if target_x is not None and target_y is not None:
                                target_screen_x, target_screen_y = camera.apply(
                                    (target_x, target_y))
                                if utils_config.ENABLE_LOGGING:
                                    logger.log_msg(
                                        f"[DEBUG] Target Screen Coordinates: ({target_screen_x}, {target_screen_y})",
                                        level=logging.DEBUG)

                                if task_type == "eliminate":
                                    box_colour = (255, 0, 0)
                                elif task_type == "gather":
                                    box_colour = (0, 255, 0)
                                else:
                                    box_colour = (255, 255, 0)

                                box_rect = pygame.Rect(
                                    target_screen_x -
                                    (utils_config.CELL_SIZE * camera.zoom) // 2,
                                    target_screen_y -
                                    (utils_config.CELL_SIZE * camera.zoom) // 2,
                                    utils_config.CELL_SIZE * camera.zoom,
                                    utils_config.CELL_SIZE * camera.zoom
                                )

                                if utils_config.ENABLE_LOGGING:
                                    logger.log_msg(
                                        f"[DEBUG] Drawing Box: {box_rect}", level=logging.DEBUG)
                                pygame.draw.rect(
                                    self.screen, box_colour, box_rect, width=2)
                            else:
                                if utils_config.ENABLE_LOGGING:
                                    logger.log_msg(
                                        f"[ERROR] No valid target coordinates for Agent {agent.agent_id}",
                                        level=logging.ERROR)

                    # 📝 Tooltip Info
                    if isinstance(agent.current_task, dict):
                        task_type = agent.current_task.get("type", "Unknown")
                        target = agent.current_task.get("target", {})
                        target_type = target.get(
                            "type", "Unknown") if isinstance(
                            target, dict) else "Unknown"
                        target_position = target.get(
                            "position", "Unknown") if isinstance(
                            target, dict) else "Unknown"
                        target_quantity = target.get(
                            "quantity", "Unknown") if isinstance(
                            target, dict) else "N/A"
                        task_state = agent.current_task.get("state", "Unknown")

                        task_info = (
                            f"Task: {task_type}\n"
                            f"Target: {target_type} at {target_position}\n"
                            f"Quantity: {target_quantity}\n"
                            f"State: {task_state}"
                        )
                    else:
                        task_info = "Task: None"

                    action = (
                        agent.role_actions[agent.current_action]
                        if 0 <= agent.current_action < len(agent.role_actions)
                        else f"Idle ({agent.current_action})"
                    )

                    self.render_tooltip(
                        f"ID: {agent.agent_id}\n"
                        f"Role: {agent.role}\n"
                        f"{task_info}\n"
                        f"Action: {action}\n"
                        f"Health: {agent.Health}\n"
                        f"Position: ({round(agent.x)}, {round(agent.y)})"
                    )


#    _____           _ _   _
#   |_   _|__   ___ | | |_(_)_ __  ___
#     | |/ _ \ / _ \| | __| | '_ \/ __|
#     | | (_) | (_) | | |_| | |_) \__ \
#     |_|\___/ \___/|_|\__|_| .__/|___/
#                           |_|


    def render_tooltip(self, text, position=None):
        """
        Render a tooltip box with the given text at the specified position.
        If no position is provided, the tooltip will be displayed in the bottom-right corner of the screen.
        """
        font = get_font(16)  # Font for tooltip text
        padding = 5  # Padding inside the tooltip box
        line_height = 20  # Height of each line
        lines = text.split("\n")  # Split text into lines

        # Calculate tooltip size
        tooltip_width = max(font.size(line)[0] for line in lines) + 2 * padding
        tooltip_height = len(lines) * line_height + 2 * padding

        # Determine tooltip position
        if position is None:
            # Default to bottom-right corner
            tooltip_x = self.screen.get_width() - tooltip_width - \
                10  # Offset from the right edge
            tooltip_y = self.screen.get_height() - tooltip_height - \
                10  # Offset from the bottom edge
        else:
            tooltip_x, tooltip_y = position  # Use the provided position

        # Render tooltip background
        tooltip_surface = pygame.Surface((tooltip_width, tooltip_height))
        tooltip_surface.set_alpha(200)  # Transparency
        tooltip_surface.fill((0, 0, 0))  # Black background

        # Render text on the tooltip
        for i, line in enumerate(lines):
            line_surface = font.render(
                line, True, (255, 255, 255))  # White text
            tooltip_surface.blit(
                line_surface, (padding, i * line_height + padding))

        # Blit the tooltip surface onto the screen
        self.screen.blit(tooltip_surface, (tooltip_x, tooltip_y))

    def load_frames(self, sprite_sheet, frame_width, frame_height):
        """
        Extract individual frames from a sprite sheet.

        :param sprite_sheet: The loaded sprite sheet image.
        :param frame_width: The width of each frame in the sprite sheet.
        :param frame_height: The height of each frame in the sprite sheet.
        :return: A list of individual frames.
        """
        frames = []
        sheet_width, sheet_height = sprite_sheet.get_size()
        for i in range(
                sheet_height //
                frame_height):  # Loop through vertical frames
            frame = sprite_sheet.subsurface(
                pygame.Rect(0, i * frame_height, frame_width, frame_height)
            )
            frames.append(frame)
        return frames

    def play_attack_animation(self, position, duration):
        """
        Play an attack animation at the given world position without blocking the main loop.

        :param position: World coordinates (x, y) for the animation.
        :param duration: Duration of the animation in milliseconds.
        """
        print(f"Renderer - Playing attack animation at position: {position}")
        # Convert world position to screen position
        screen_x, screen_y = self.camera.apply(position)
        # print(f"World Position: {position}")
        # print(f"Screen Position: ({screen_x}, {screen_y})")

        # Number of animation frames and duration per frame
        frame_count = len(self.attack_frames)
        frame_duration = duration // frame_count

        # Store the animation state
        self.active_animations.append({
            "screen_position": (screen_x, screen_y),
            "start_time": pygame.time.get_ticks(),
            "frame_duration": frame_duration,
            "current_frame": 0,
            "total_frames": frame_count,
            "duration": duration,
            "z_index": 100  # Add high z-index to ensure animation appears in front
        })

    def update_animations(self):
        """
        Update and render active animations frame by frame.
        """
        current_time = pygame.time.get_ticks()
        updated_animations = []

        for animation in self.active_animations:
            elapsed_time = current_time - animation["start_time"]
            frame_index = elapsed_time // animation["frame_duration"]

            if frame_index < animation["total_frames"]:
                # Render the current frame
                frame = self.attack_frames[frame_index]
                screen_x, screen_y = animation["screen_position"]
                self.screen.blit(frame,
                                 (screen_x - frame.get_width() // 2,
                                  screen_y - frame.get_height() // 2))
                updated_animations.append(animation)

        # Update the active animations
        self.active_animations = updated_animations


#    __  __    _    ___ _   _   __  __ _____ _   _ _   _
#   |  \/  |  / \  |_ _| \ | | |  \/  | ____| \ | | | | |
#   | |\/| | / _ \  | ||  \| | | |\/| |  _| |  \| | | | |
#   | |  | |/ ___ \ | || |\  | | |  | | |___| |\  | |_| |
#   |_|  |_/_/   \_\___|_| \_| |_|  |_|_____|_| \_|\___/
#


class MenuRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(FONT_NAME, 24)
        self.selected_mode = None
        print("MenuRenderer initialised")

    def draw_text(self, surface, text, font, size, colour, x, y):
        font_obj = pygame.font.SysFont(font, size)
        text_surface = font_obj.render(text, True, colour)
        text_rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, text_rect)

    def create_button(
            self,
            surface,
            text,
            font,
            size,
            colour,
            hover_colour,
            click_colour,
            x,
            y,
            width,
            height,
            state='normal',
            icon=None):
        button_rect = pygame.Rect(x, y, width, height)
        if button_rect.collidepoint(pygame.mouse.get_pos()):
            if state == 'normal':
                pygame.draw.rect(surface, hover_colour, button_rect)
            elif state == 'clicked':
                pygame.draw.rect(surface, click_colour, button_rect)
        else:
            pygame.draw.rect(surface, colour, button_rect)

        self.draw_text(surface, text, font, size, WHITE,
                       x + width / 2, y + height / 2)

        if icon:
            icon_size = 40
            icon_x = x + width - icon_size - 10
            icon_y = y + (height // 2) - (icon_size // 2)
            surface.blit(icon, (icon_x, icon_y))

        return button_rect

    def render_menu(
            self,
            ENABLE_TENSORBOARD,
            auto_ENABLE_TENSORBOARD,
            mode,
            start_game_callback):
        self.screen.fill(BLACK)
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT

        button_width = 250
        button_height = 40
        button_font_size = 22
        button_spacing = 20

        center_x = SCREEN_WIDTH // 2 - button_width // 2
        base_y = 250
        self.draw_text(
            self.screen,
            "Welcome to the Multi-agent competitive and cooperative strategy (MACCS) Simulation",
            FONT_NAME,
            28,
            WHITE,
            SCREEN_WIDTH //
            2,
            base_y -
            100)
        self.draw_text(
            self.screen,
            "Created as part of my BSC Computer Science final year project",
            FONT_NAME,
            20,
            WHITE,
            SCREEN_WIDTH // 2,
            base_y - 70)
        self.draw_text(self.screen, "Choose Mode", FONT_NAME,
                       28, WHITE, SCREEN_WIDTH // 2, base_y - 40)

        check_icon = pygame.font.SysFont(
            FONT_NAME, 40).render('✔', True, GREEN)
        cross_icon = pygame.font.SysFont(FONT_NAME, 40).render('❌', True, RED)

        half_width = button_width // 2 + 10

        train_button_rect = self.create_button(
            self.screen,
            "Training",
            FONT_NAME,
            button_font_size,
            GREEN,
            (0,
             200,
             0),
            (0,
             100,
             0),
            SCREEN_WIDTH //
            2 -
            half_width -
            150,
            base_y,
            button_width,
            button_height)
        evaluate_button_rect = self.create_button(
            self.screen, "Evaluation", FONT_NAME, button_font_size, BLUE, (
                0, 0, 255), (0, 0, 200),
            SCREEN_WIDTH // 2 + 50, base_y, button_width, button_height
        )
        if self.selected_mode == 'train':
            start_text = "Start Training Simulation"
            base_color = GREEN
            hover_color = (0, 200, 0)
            click_color = (0, 100, 0)
        elif self.selected_mode == 'evaluate':
            start_text = "Start Evaluation Simulation"
            base_color = BLUE
            hover_color = (0, 0, 255)
            click_color = (0, 0, 200)
        else:
            start_text = "Mode Required"
            base_color = GREY
            hover_color = (100, 100, 100)
            click_color = (70, 70, 70)

        start_button_rect = self.create_button(
            self.screen,
            start_text,
            FONT_NAME,
            button_font_size,
            base_color,
            hover_color,
            click_color,
            center_x,
            base_y + (button_height + button_spacing) * 2,
            button_width,
            button_height
        )

        settings_button_rect = self.create_button(
            self.screen, "Settings", FONT_NAME, button_font_size, GREY, (
                180, 180, 180), (100, 100, 100),
            center_x, base_y + (button_height + button_spacing) *
            3, button_width, button_height
        )

        credits_button_rect = self.create_button(
            self.screen, "Credits", FONT_NAME, button_font_size, DARK_GREY, (
                160, 160, 160), (90, 90, 90),
            center_x, base_y + (button_height + button_spacing) *
            4, button_width, button_height
        )

        exit_button_rect = self.create_button(
            self.screen, "Exit", FONT_NAME, button_font_size, RED, (
                150, 0, 0), (200, 0, 0),
            center_x, base_y + (button_height + button_spacing) *
            5, button_width, button_height
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if train_button_rect.collidepoint(event.pos):
                    self.selected_mode = 'train'
                    print("[INFO] Training mode selected.")

                elif evaluate_button_rect.collidepoint(event.pos):
                    self.selected_mode = 'evaluate'
                    print("[INFO] Evaluation mode selected.")

                elif start_button_rect.collidepoint(event.pos) and self.selected_mode:
                    print("[INFO] Starting game in mode:", self.selected_mode)
                    start_game_callback(
                        self.selected_mode,
                        utils_config.ENABLE_TENSORBOARD,
                        utils_config.ENABLE_TENSORBOARD)
                    return False

                elif settings_button_rect.collidepoint(event.pos):
                    settings_menu = SettingsMenuRenderer(self.screen)
                    while settings_menu.render():
                        pass

                    if settings_menu.saved:
                        updated = settings_menu.get_settings()
                        for key, value in updated.items():
                            if hasattr(utils_config, key):
                                setattr(utils_config, key, value)
                        print("[INFO] Updated settings:", updated)

                elif credits_button_rect.collidepoint(event.pos):
                    # Make sure this import exists at the top
                    credits = CreditsRenderer(self.screen)
                    credits.run()

                elif exit_button_rect.collidepoint(event.pos):
                    print("[INFO] Exiting game...")
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()
        return True


#    ____       _   _   _                   __  __
#   / ___|  ___| |_| |_(_)_ __   __ _ ___  |  \/  | ___ _ __  _   _
#   \___ \ / _ \ __| __| | '_ \ / _` / __| | |\/| |/ _ \ '_ \| | | |
#    ___) |  __/ |_| |_| | | | | (_| \__ \ | |  | |  __/ | | | |_| |
#   |____/ \___|\__|\__|_|_| |_|\__, |___/ |_|  |_|\___|_| |_|\__,_|
#                               |___/


class SettingsMenuRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(FONT_NAME, 24)
        self.selected_category = "debugging"
        self.saved = False
        self.input_mode = False
        self.input_field = None
        self.input_text = ""
        self.cursor_visible = True
        self.cursor_timer = 0

        # Define category labels
        self.sidebar_items = [
            "debugging", "episode settings", "screen",
            "world", "resources", "agent", "faction"
        ]

        # Store config state and original defaults
        self.settings_by_category = {}
        self.defaults = {}

        raw_settings = {
            "debugging": [
                ("TensorBoard", "ENABLE_TENSORBOARD"),
                ("Logging", "ENABLE_LOGGING"),
                ("Profiling", "ENABLE_PROFILE_BOOL"),

            ],
            "episode settings": [
                ("Episodes", "EPISODES_LIMIT", 5),
                ("Steps per Episode", "STEPS_PER_EPISODE", 100),
            ],
            "screen": [
                ("FPS", "FPS", 5),
                ("Width", "SCREEN_WIDTH", 50),
                ("Height", "SCREEN_HEIGHT", 50),
                ("Cell Size", "CELL_SIZE", 1),
            ],
            "world": [
                ("World Width", "WORLD_WIDTH", 50),
                ("World Height", "WORLD_HEIGHT", 50),
                ("Terrain Seed", "Terrain_Seed", 1),
                ("Noise Scale", "NOISE_SCALE", 0.1),
                ("Octaves", "NOISE_OCTAVES", 1),
                ("Persistence", "NOISE_PERSISTENCE", 0.1),
                ("Lacunarity", "NOISE_LACUNARITY", 0.1),
                ("Water Coverage", "WATER_COVERAGE", 0.01),
            ],
            "resources": [
                ("Tree Density", "TREE_DENSITY", 0.01),
                ("Apple Regen Time", "APPLE_REGEN_TIME", 1),
                ("Gold Zone Probability", "GOLD_ZONE_PROBABILITY", 0.01),
                ("Gold Spawn Density", "GOLD_SPAWN_DENSITY", 0.01),
            ],
            "agent": [
                ("Field of View", "Agent_field_of_view", 1),
                ("Interact Range", "Agent_Interact_Range", 1),
                ("Gold Cost for Agent", "Gold_Cost_for_Agent", 1),
            ],
            "faction": [
                ("Spawn Radius", "HQ_SPAWN_RADIUS", 5),
                ("Agent Spawn Radius", "HQ_Agent_Spawn_Radius", 2),
                ("Faction Count", "FACTON_COUNT", 1),
                ("Initial Gatherers", "INITAL_GATHERER_COUNT", 1),
                ("Initial Peacekeepers", "INITAL_PEACEKEEPER_COUNT", 1),
            ]
        }

        for category, fields in raw_settings.items():
            self.settings_by_category[category] = []
            for item in fields:
                label, key = item[:2]
                value = getattr(utils_config, key)
                self.defaults[key] = value
                setting = {"label": label, "key": key, "value": value}
                if len(item) == 3:
                    setting["step"] = item[2]
                else:
                    setting["options"] = [True, False]
                self.settings_by_category[category].append(setting)

        self.check_icon = pygame.font.SysFont(
            FONT_NAME, 40).render('✔', True, GREEN)
        self.cross_icon = pygame.font.SysFont(
            FONT_NAME, 40).render('❌', True, RED)

    def draw_text(self, text, size, colour, x, y):
        font_obj = pygame.font.SysFont(FONT_NAME, size)
        text_surface = font_obj.render(text, True, colour)
        text_rect = text_surface.get_rect(topleft=(x, y))
        self.screen.blit(text_surface, text_rect)

    def create_button(
            self,
            text,
            font,
            size,
            colour,
            hover_colour,
            click_colour,
            x,
            y,
            width,
            height,
            icon=None):
        button_rect = pygame.Rect(x, y, width, height)
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(
                self.screen,
                click_colour if mouse_pressed else hover_colour,
                button_rect)
        else:
            pygame.draw.rect(self.screen, colour, button_rect)

        self.draw_text(text, size, WHITE, x + 10, y + 10)

        if icon:
            icon_size = 40
            icon_x = x + width - icon_size - 10
            icon_y = y + (height // 2) - (icon_size // 2)
            self.screen.blit(icon, (icon_x, icon_y))

        return button_rect

    def render(self):
        self.screen.fill(BLACK)
        self.cursor_timer += 1
        if self.cursor_timer >= 30:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

        for idx, label in enumerate(self.sidebar_items):
            y = 80 + idx * 60
            selected = (label == self.selected_category)
            color = GREY if selected else DARK_GREY
            btn = self.create_button(
                label.upper(),
                FONT_NAME,
                20,
                color,
                (120,
                 120,
                 120),
                (180,
                 180,
                 180),
                20,
                y,
                200,
                40)
            if pygame.mouse.get_pressed()[0] and btn.collidepoint(
                    pygame.mouse.get_pos()):
                self.selected_category = label

        settings = self.settings_by_category.get(self.selected_category, [])
        step_buttons = []
        for i, setting in enumerate(settings):
            y = 80 + i * 60
            label = setting["label"]
            val = setting["value"]
            self.draw_text(f"{label}:", 20, WHITE, 250, y)
            if self.input_mode and self.input_field == setting:
                display_text = self.input_text + \
                    ("|" if self.cursor_visible else "")
                self.draw_text(display_text, 20, WHITE, 600, y)
            else:
                self.draw_text(str(val), 20, GREEN if val else RED, 600, y)

            if "options" in setting:
                toggle_btn = pygame.Rect(700, y, 40, 30)
                pygame.draw.rect(self.screen, DARK_GREY, toggle_btn)
                if setting["value"]:
                    pygame.draw.rect(self.screen, GREEN, toggle_btn)
                    self.draw_text("ON", 18, BLACK,
                                   toggle_btn.x + 5, toggle_btn.y + 5)
                else:
                    pygame.draw.rect(self.screen, RED, toggle_btn)
                    self.draw_text("OFF", 18, BLACK,
                                   toggle_btn.x + 2, toggle_btn.y + 5)

                step_buttons.append(("toggle", toggle_btn, setting))

            elif "step" in setting:
                minus_btn = pygame.Rect(700, y, 30, 30)
                plus_btn = pygame.Rect(740, y, 30, 30)
                default_btn = pygame.Rect(780, y, 80, 30)
                pygame.draw.rect(self.screen, RED, minus_btn)
                pygame.draw.rect(self.screen, GREEN, plus_btn)
                pygame.draw.rect(self.screen, GREY, default_btn)
                self.draw_text("-", 20, WHITE, minus_btn.x +
                               10, minus_btn.y + 5)
                self.draw_text("+", 20, WHITE, plus_btn.x + 10, plus_btn.y + 5)
                self.draw_text("Reset", 18, WHITE,
                               default_btn.x + 10, default_btn.y + 5)
                step_buttons.append(("minus", minus_btn, setting))
                step_buttons.append(("plus", plus_btn, setting))
                step_buttons.append(("reset", default_btn, setting))

        back_btn = self.create_button(
            "Back", FONT_NAME, 20, GREY, (180, 180, 180), (120, 120, 120), 250, 500, 150, 50)
        save_return_btn = self.create_button(
            "Save and Return",
            FONT_NAME,
            20,
            BLUE,
            (80,
             80,
             255),
            (50,
             50,
             200),
            450,
            500,
            250,
            50)
        reset_all_btn = self.create_button(
            "Reset All",
            FONT_NAME,
            20,
            GREY,
            (180,
             180,
             180),
            (120,
             120,
             120),
            20,
            500,
            200,
            50)

        note_text = "Tip: You can click on a value to input in a custom number. Type and press Enter to confirm"
        screen_width = self.screen.get_width()
        self.draw_text(note_text, 24, GREY, 50, 560)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, setting in enumerate(settings):
                    y = 80 + i * 60
                    # matches where you draw the values
                    value_rect = pygame.Rect(600, y, 80, 30)
                    if value_rect.collidepoint(
                            mouse_x, mouse_y) and "step" in setting:
                        self.input_mode = True
                        self.input_field = setting
                        self.input_text = str(setting["value"])
                        clicked_input = True

                if back_btn.collidepoint(event.pos):
                    return False
                if save_return_btn.collidepoint(event.pos):
                    self.saved = True
                    return False
                if reset_all_btn.collidepoint(event.pos):
                    for settings in self.settings_by_category.values():
                        for setting in settings:
                            setting["value"] = self.defaults.get(
                                setting["key"], setting["value"])
                for action, rect, setting in step_buttons:
                    if rect.collidepoint(event.pos):
                        if action == "toggle":
                            options = setting["options"]
                            current_index = options.index(setting["value"])
                            setting["value"] = options[(
                                current_index + 1) % len(options)]
                        elif action == "minus":
                            setting["value"] = round(
                                setting["value"] - setting["step"], 3)
                        elif action == "plus":
                            setting["value"] = round(
                                setting["value"] + setting["step"], 3)
                        elif action == "reset":
                            default_val = self.defaults.get(
                                setting["key"], setting["value"])
                            setting["value"] = default_val
                clicked_input = False
                for i, setting in enumerate(settings):
                    y = 80 + i * 60
                    value_rect = pygame.Rect(600, y, 80, 30)
                    if value_rect.collidepoint(
                            mouse_x, mouse_y) and "step" in setting:
                        self.input_mode = True
                        self.input_field = setting
                        self.input_text = str(setting["value"])
                        clicked_input = True

                if not clicked_input:
                    self.input_mode = False
                    self.input_field = None

            if event.type == pygame.KEYDOWN and self.input_mode:
                if event.key == pygame.K_RETURN:
                    try:
                        value = float(
                            self.input_text) if '.' in self.input_text else int(
                            self.input_text)
                        self.input_field["value"] = value
                    except ValueError:
                        pass  # optionally show a warning or revert to old value
                    self.input_mode = False
                    self.input_field = None
                    self.input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    if event.unicode.isdigit() or event.unicode in ['.', '-']:
                        self.input_text += event.unicode

        return True

    def get_settings(self):
        settings = {}
        for cat in self.settings_by_category.values():
            for setting in cat:
                settings[setting["key"]] = setting["value"]
        return settings


class CreditsRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("arial", 28)
        self.title_font = pygame.font.SysFont("arial", 36, bold=True)
        self.text_colour = (255, 255, 255)
        self.bg_colour = (0, 0, 0)

        self.lines = [
            "Multi-Agent Competitive and Cooperative Strategy (MACCS)",
            "Developed by Arunpreet Garcha",
            "For my",
            "Final Year Project",
            "At",
            "University College Birmingham",
            "April 2025",
            "",
            "Press ESC to return to the menu"
        ]

    def draw(self):
        self.screen.fill(self.bg_colour)

        # Title
        title = self.title_font.render("CREDITS", True, self.text_colour)
        title_rect = title.get_rect(center=(self.screen.get_width() // 2, 80))
        self.screen.blit(title, title_rect)

        # Credit lines
        for i, line in enumerate(self.lines):
            rendered = self.font.render(line, True, self.text_colour)
            rect = rendered.get_rect(
                center=(self.screen.get_width() // 2, 150 + i * 40))
            self.screen.blit(rendered, rect)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
