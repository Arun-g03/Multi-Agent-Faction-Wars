"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from ENVIRONMENT.env_resources import AppleTree, GoldLump
import UTILITIES.utils_config as utils_config
from RENDER.Common import GAME_FONT, get_font, get_text_surface


logger = Logger(log_file="Renderer.txt", log_level=logging.DEBUG)



#    ____                _
#   |  _ \ ___ _ __   __| | ___ _ __ ___ _ __
#   | |_) / _ \ '_ \ / _` |/ _ \ '__/ _ \ '__|
#   |  _ <  __/ | | | (_| |  __/ | |  __/ |
#   |_| \_\___|_| |_|\__,_|\___|_|  \___|_|
#

pygame.init()


Game_Title = 'Multi-agent competitive and cooperative strategy (MACCS) - Simulation'



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
        pygame.display.set_caption(Game_Title)
        # Use cached font for displaying faction IDs on bases
        self.font = get_font(24)
        if not utils_config.HEADLESS_MODE:
            self.attack_sprite_sheet = pygame.image.load(
                "RENDER\IMAGES\Attack_Animation.png").convert_alpha()
            self.faction_base_image = pygame.image.load(
                "RENDER\IMAGES\castle-7440761_1280.png").convert_alpha()
            self.attack_frames = self.load_frames(
                self.attack_sprite_sheet, frame_width=64, frame_height=64)
        self.camera = camera  # Set camera if passed, or use None by default
        self.active_animations = []
        

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
        if utils_config.HEADLESS_MODE:
            return  
        else:
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

        if utils_config.HEADLESS_MODE:
                return  
        else:
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
                    text = get_text_surface(str(faction.id), "Arial", 24, (0, 0, 0))                
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
                            f"Gatherers: {gatherer_count}\n"
                            f"Current Strategy: {faction.current_strategy}\n"
                            f"HQ Location: {x, y}"
                        )
                        self.render_tooltip(overlay_text)


#    _   _ _   _ ____
#   | | | | | | |  _ \
#   | |_| | | | | | | |
#   |  _  | |_| | |_| |
#   |_| |_|\___/|____/
#


    def render_hud(self, screen, episode, step, factions, resource_counts):
        font_size = 20
        lines = []

        # Episode and Step
        lines.append(get_text_surface(
            f"Episode: {episode}/{utils_config.EPISODES_LIMIT} | Step: {step} /{utils_config.STEPS_PER_EPISODE}",
            GAME_FONT, font_size, (255, 255, 255)
        ))

        # Time
        start_time = getattr(self, 'start_time', pygame.time.get_ticks())
        if not hasattr(self, 'start_time'):
            self.start_time = start_time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000
        hours, minutes, seconds = elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60
        lines.append(get_text_surface(
            f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}",
            GAME_FONT, font_size, (255, 255, 255)
        ))
        # Resources
        lines.append(get_text_surface("Resources available:", GAME_FONT, font_size, (255, 255, 255)))
        lines.append(get_text_surface(
            f"Gold Lumps: {resource_counts.get('gold_lumps', 0)}, Gold: {resource_counts.get('gold_quantity', 0)}",
            GAME_FONT, font_size, (255, 255, 255)
        ))
        lines.append(get_text_surface(
            f"Apple Trees: {resource_counts.get('apple_trees', 0)}, Apples: {resource_counts.get('apple_quantity', 0)}",
            GAME_FONT, font_size, (255, 255, 255)
        ))

        # Leaderboard
        lines.append(get_text_surface("Faction Leaderboard:", GAME_FONT, font_size, (255, 255, 255)))
        sorted_factions = sorted(factions, key=lambda f: f.gold_balance, reverse=True)
        for faction in sorted_factions[:4]:
            lines.append(get_text_surface(
                f"Faction {faction.id} - Gold: {faction.gold_balance}, Food: {faction.food_balance}, Agents: {len(faction.agents)}",
                GAME_FONT, font_size, (255, 255, 255)
            ))

        # Compute background size dynamically
        padding = 10
        line_height = get_font(font_size).get_height() + 4
        max_width = max(line.get_width() for line in lines)
        total_height = len(lines) * line_height

        # Create dynamic HUD background
        hud_background = pygame.Surface((max_width + 2 * padding, total_height + 2 * padding))
        hud_background.set_alpha(128)
        hud_background.fill((0, 0, 0))
        screen.blit(hud_background, (10, 10))

        # Draw text lines
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


                    # Highlight task target
                    if isinstance(agent.current_task, dict):
                        task_type = agent.current_task.get("type", "unknown")
                        target = agent.current_task.get("target")

                        if target:
                            if utils_config.ENABLE_LOGGING:
                                logger.log_msg(
                                    f"[HUD TARGET] Agent {agent.agent_id} Task -> Type: {task_type}, Target: {target}",
                                    level=logging.DEBUG)

                            target_position = target.get("position", None)  # Get position, or None if not found

                            # Handle None case
                            if target_position is None:
                                target_position = (0, 0)  # Default position if None
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

                    # Tooltip Info
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
                            if isinstance(agent.current_action, int) and 0 <= agent.current_action < len(agent.role_actions)
                            else "None"
                        )


                    if utils_config.SUB_TILE_PRECISION:
                        pos_string = f"({round(agent.x)}, {round(agent.y)})"
                    else:
                        grid_x = int(agent.x // utils_config.CELL_SIZE)
                        grid_y = int(agent.y // utils_config.CELL_SIZE)
                        pos_string = f"Grid: ({grid_x}, {grid_y})"

                    self.render_tooltip(
                        f"ID: {agent.agent_id}\n"
                        f"Role: {agent.role}\n"
                        f"{task_info}\n"
                        f"Action: {action}\n"
                        f"Health: {agent.Health}\n"
                        f"Position: {pos_string}"
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
            line_surface = get_text_surface(line, "Arial", 16, (255, 255, 255))            
            tooltip_surface.blit(
                line_surface, (padding, i * line_height + padding))

        # Blit the tooltip surface onto the screen
        self.screen.blit(tooltip_surface, (tooltip_x, tooltip_y))

    def load_frames(self, sprite_sheet, frame_width, frame_height):
        """
        Extract individual frames from a sprite sheet.
        """
        if utils_config.HEADLESS_MODE:
            return []  # Skip loading frames entirely

        frames = []
        sheet_width, sheet_height = sprite_sheet.get_size()
        for i in range(sheet_height // frame_height):
            frame = sprite_sheet.subsurface(
                pygame.Rect(0, i * frame_height, frame_width, frame_height)
            )
            frames.append(frame)
        return frames

    def play_attack_animation(self, position, duration):
        """
        Queue an attack animation at the given world position.
        """
        if utils_config.HEADLESS_MODE:
            return  # Skip in headless mode

        print(f"Renderer - Playing attack animation at position: {position}")
        screen_x, screen_y = self.camera.apply(position)

        frame_count = len(self.attack_frames)
        if frame_count == 0:
            return  # Avoid div-by-zero if load_frames was skipped

        frame_duration = duration // frame_count

        self.active_animations.append({
            "screen_position": (screen_x, screen_y),
            "start_time": pygame.time.get_ticks(),
            "frame_duration": frame_duration,
            "current_frame": 0,
            "total_frames": frame_count,
            "duration": duration,
            "z_index": 100
        })


    def update_animations(self):
        """
        Update and render active animations frame by frame.
        """
        if utils_config.HEADLESS_MODE:
            return  # Skip animation rendering in headless mode

        current_time = pygame.time.get_ticks()
        updated_animations = []

        for animation in self.active_animations:
            elapsed_time = current_time - animation["start_time"]
            frame_index = elapsed_time // animation["frame_duration"]

            if frame_index < animation["total_frames"]:
                frame = self.attack_frames[frame_index]
                screen_x, screen_y = animation["screen_position"]
                self.screen.blit(
                    frame,
                    (screen_x - frame.get_width() // 2, screen_y - frame.get_height() // 2)
                )
                updated_animations.append(animation)

        self.active_animations = updated_animations

