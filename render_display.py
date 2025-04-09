import pygame
from utils_config import (SCREEN_WIDTH, 
                          SCREEN_HEIGHT, 
                          SCALING_FACTOR, 
                          Tree_Scale_Img, 
                          GoldLump_Scale_Img, 
                          CELL_SIZE, 
                          Agent_field_of_view, 
                          Agent_Interact_Range,
                          LOGGING_ENABLED)
from env_resources import AppleTree, GoldLump
import sys
import traceback
import logging
from utils_logger import Logger
logger = Logger(log_file="Renderer.txt", log_level=logging.DEBUG)

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
    def __init__(self, screen, terrain, resources, factions, agents, camera=None):
        pygame.font.init()  # Initialise the font module
        self.screen = screen  # Use the screen passed as argument
        pygame.display.set_caption('Multi-agent competitive and cooperative strategy (MACCS) - Simulation')
        self.font = get_font(24)  # Use cached font for displaying faction IDs on bases
        self.attack_sprite_sheet = pygame.image.load("images/Attack_Animation.png").convert_alpha()
        self.attack_frames = self.load_frames(self.attack_sprite_sheet, frame_width=64, frame_height=64)
        self.camera = camera  # Set camera if passed, or use None by default
        self.active_animations = []
        self.faction_base_image = pygame.image.load("images/castle-7440761_1280.png").convert_alpha()


    def render(self, camera, terrain, resources, factions, agents, episode, step, resource_counts, enable_cell_tooltip=True):
        """
        Render the entire game state, including terrain, resources, home bases, and agents.
        Display tooltips for resources or agents if the mouse hovers over them.
        Optionally, show the grid cell under the mouse cursor relative to the world coordinates.
        """
        try:
            self.screen.fill((0, 0, 0))  # Clear the screen with black background
            self.update_animations()

            # Render terrain, passing factions for territory visualisation
            terrain.draw(self.screen, camera, factions)

            # Get mouse position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            tooltip_text = None  # To hold tooltip text if hovering over a resource or agent

            # Render resources and check for hover
            for resource in resources:
                if not hasattr(resource, 'x') or not hasattr(resource, 'y'):
                    print(f"Warning: Resource {resource} does not have 'x' and 'y' attributes. Skipping render.")
                    continue  # Skip this resource if invalid
                resource.render(self.screen, camera)  # Render the resource

                # Recalculate the resource's screen position based on the camera
                screen_x = (resource.x - camera.x) * camera.zoom
                screen_y = (resource.y - camera.y) * camera.zoom

                # Calculate the size of the resource for collision detection
                if isinstance(resource, AppleTree):
                    final_size = int(CELL_SIZE * SCALING_FACTOR * camera.zoom * Tree_Scale_Img)
                elif isinstance(resource, GoldLump):
                    final_size = int(CELL_SIZE * SCALING_FACTOR * camera.zoom * GoldLump_Scale_Img)
                else:
                    final_size = CELL_SIZE  # Fallback for unknown resources

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
                mouse_grid_x = int(world_mouse_x // CELL_SIZE)
                mouse_grid_y = int(world_mouse_y // CELL_SIZE)

                # Create the tooltip text for the world-relative cell and screen position
                cell_tooltip_text = (
                    f"Screen Pos: ({mouse_x}, {mouse_y})\n"  # Screen position
                    f"World Cell: ({mouse_grid_x}, {mouse_grid_y})"  # Grid coordinates
                )

                # Calculate bottom-left position for the tooltip
                tooltip_x = 10  # Offset from the left edge
                tooltip_y = self.screen.get_height() - 50  # 50px offset from the bottom edge (adjust based on tooltip size)

                # Render the tooltip using the existing tooltip function
                self.render_tooltip(cell_tooltip_text, position=(tooltip_x, tooltip_y))

            #  Render the HUD with episode and step data
            self.render_hud(self.screen, episode, step, factions, resource_counts)

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
                base_image_scaled = pygame.transform.scale(self.faction_base_image, (int(adjusted_size), int(adjusted_size)))
                # Blit the base image instead of drawing a square
                self.screen.blit(base_image_scaled, (screen_x, screen_y))


                # Display the faction ID at the center of the base
                text = self.font.render(str(faction.id), True, (0, 0, 0))
                text_rect = text.get_rect(center=(screen_x + adjusted_size // 2 - 12, screen_y + adjusted_size // 2))
                self.screen.blit(text, text_rect)
                # Highlight resources and threats known to the faction if hovering over the base
                base_rect = pygame.Rect(screen_x, screen_y, adjusted_size, adjusted_size)
                if base_rect.collidepoint(mouse_x, mouse_y):
                    
                        
                    # Highlight resources
                    for resource in faction.global_state["resources"]:
                        resource_screen_x = (resource["location"][0] * CELL_SIZE - camera.x) * camera.zoom
                        resource_screen_y = (resource["location"][1] * CELL_SIZE - camera.y) * camera.zoom

                        pygame.draw.rect(
                            self.screen,
                            (0, 255, 0),  # Green colour for highlighting resources
                            pygame.Rect(
                                resource_screen_x - CELL_SIZE * camera.zoom // 2,
                                resource_screen_y - CELL_SIZE * camera.zoom // 2,
                                CELL_SIZE * camera.zoom,
                                CELL_SIZE * camera.zoom
                            ),
                            width=2
                        )

                    # Highlight threats
                    for threat in faction.global_state["threats"]:
                        try:
                            if isinstance(threat, dict) and "location" in threat:
                                threat_x, threat_y = threat["location"]
                            else:
                                continue  # Skip invalid threats

                            # Convert grid coordinates to screen coordinates
                            threat_screen_x = (threat_x - camera.x) * camera.zoom
                            threat_screen_y = (threat_y - camera.y) * camera.zoom

                            # Draw the threat box
                            box_size = CELL_SIZE * camera.zoom
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
                        "apple_trees": sum(1 for res in faction.global_state["resources"] if res.get("type") == "AppleTree"),
                        "gold_lumps": sum(1 for res in faction.global_state["resources"] if res.get("type") == "GoldLump"),
                    }
                    threat_count = len(faction.global_state["threats"])
                    peacekeeper_count = sum(1 for agent in faction.agents if agent.role == "peacekeeper")
                    gatherer_count = sum(1 for agent in faction.agents if agent.role == "gatherer")

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
        """
        Render the HUD with game statistics.
        :param screen: The Pygame display surface.
        :param episode: Current episode number.
        :param step: Current step in the episode.
        :param factions: List of factions for leaderboard stats.
        :param resource_counts: Dictionary with counts of resources.
        """
        # Create a transparent background surface
        hud_background = pygame.Surface((300, 320))  # Increased height to accommodate all elements
        hud_background.set_alpha(128)  # Transparency (0-255)
        hud_background.fill((0, 0, 0))  # Fill with black colour

        # Blit the transparent background onto the main screen
        screen.blit(hud_background, (10, 10))  # Slight offset from corner for better visibility

        # Font for HUD text
        font = get_font(20)        

        # Display Episode and Step
        episode_text = font.render(f"Episode: {episode} | Step: {step}", True, (255, 255, 255))
        screen.blit(episode_text, (20, 20))

        # Display elapsed time
        elapsed_time = pygame.time.get_ticks() // 1000  # Convert milliseconds to seconds
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        time_text = font.render(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}", True, (255, 255, 255))
        screen.blit(time_text, (20, 50))

        # Display Resource Counts
        gold_lumps = resource_counts.get("gold_lumps", 0)
        gold_quantity = resource_counts.get("gold_quantity", 0)
        tree_count = resource_counts.get("apple_trees", 0)
        apple_count = resource_counts.get("apple_quantity", 0)
        resources_header = font.render("Resources available:", True, (255, 255, 255))
        screen.blit(resources_header, (20, 80))
        resource_text_gold = font.render(f"Gold Lumps: {gold_lumps}, Gold: {gold_quantity}", True, (255, 255, 255))
        resource_text_apple = font.render(f"Apple Trees: {tree_count}, Apples: {apple_count}", True, (255, 255, 255))
        screen.blit(resource_text_gold, (20, 110))
        screen.blit(resource_text_apple, (20, 140))

        # Display Faction Leaderboard
        leaderboard_text = font.render("Faction Leaderboard:", True, (255, 255, 255))
        screen.blit(leaderboard_text, (20, 180))

        # Sort factions by gold balance and display the top 4
        sorted_factions = sorted(factions, key=lambda f: f.gold_balance, reverse=True)
        for i, faction in enumerate(sorted_factions[:4]):
            faction_text = font.render(
                f"{i + 1}. Faction {faction.id} - Gold: {faction.gold_balance}, Agents: {len(faction.agents)}",
                True,
                (255, 255, 255),
            )
            screen.blit(faction_text, (20, 210 + i * 30))

        # Update the display
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
                print(f"Warning: Agent {agent} does not have 'x' and 'y' attributes. Skipping render.")
                continue

            screen_x, screen_y = camera.apply((agent.x, agent.y))
            agent_sprite = agent.sprite

            if agent_sprite:
                agent_rect = agent_sprite.get_rect(center=(int(screen_x), int(screen_y)))
                self.screen.blit(agent_sprite, agent_rect)

                if agent_rect.collidepoint(mouse_x, mouse_y):

                    
                    #Field of View
                    pygame.draw.circle(self.screen, (255, 255, 0), (screen_x, screen_y), Agent_field_of_view*CELL_SIZE * camera.zoom, width=2)

                    #Interact Range
                    pygame.draw.circle(self.screen, (173, 216, 230), (screen_x, screen_y), Agent_Interact_Range*CELL_SIZE * camera.zoom, width=2)

                    # 🎯 Highlight task target
                    if isinstance(agent.current_task, dict):
                        task_type = agent.current_task.get("type", "unknown")
                        target = agent.current_task.get("target")

                        if target:
                            if LOGGING_ENABLED: logger.log_msg(
                                f"[HUD TARGET] Agent {agent.agent_id} Task -> Type: {task_type}, Target: {target}",
                                level=logging.DEBUG
                            )

                            target_position = target.get("position")
                            target_type = target.get("type")
                            target_x = target_y = None

                            if task_type == "eliminate" and target_type == "agent" and "id" in target:
                                target_id = target["id"]
                                target_agent = next((a for a in agents if a.agent_id == target_id), None)

                                if target_agent:
                                    target_x, target_y = target_agent.x, target_agent.y
                                else:
                                    if LOGGING_ENABLED: logger.log_msg(f"[WARN] Target agent {target_id} not found.", level=logging.WARNING)

                            elif target_position:
                                target_x, target_y = target_position
                                target_x *= CELL_SIZE
                                target_y *= CELL_SIZE

                            if target_x is not None and target_y is not None:
                                target_screen_x, target_screen_y = camera.apply((target_x, target_y))
                                if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Target Screen Coordinates: ({target_screen_x}, {target_screen_y})", level=logging.DEBUG)

                                if task_type == "eliminate":
                                    box_colour = (255, 0, 0)
                                elif task_type == "gather":
                                    box_colour = (0, 255, 0)
                                else:
                                    box_colour = (255, 255, 0)

                                box_rect = pygame.Rect(
                                    target_screen_x - (CELL_SIZE * camera.zoom) // 2,
                                    target_screen_y - (CELL_SIZE * camera.zoom) // 2,
                                    CELL_SIZE * camera.zoom,
                                    CELL_SIZE * camera.zoom
                                )

                                if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Drawing Box: {box_rect}", level=logging.DEBUG)
                                pygame.draw.rect(self.screen, box_colour, box_rect, width=2)
                            else:
                                if LOGGING_ENABLED: logger.log_msg(f"[ERROR] No valid target coordinates for Agent {agent.agent_id}", level=logging.ERROR)

                    # 📝 Tooltip Info
                    if isinstance(agent.current_task, dict):
                        task_type = agent.current_task.get("type", "Unknown")
                        target = agent.current_task.get("target", {})
                        target_type = target.get("type", "Unknown") if isinstance(target, dict) else "Unknown"
                        target_position = target.get("position", "Unknown") if isinstance(target, dict) else "Unknown"
                        target_quantity = target.get("quantity", "Unknown") if isinstance(target, dict) else "N/A"
                        task_info = (
                            f"Task: {task_type}\n"
                            f"Target: {target_type} at {target_position}\n"
                            f"Quantity: {target_quantity}"
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
            tooltip_x = self.screen.get_width() - tooltip_width - 10  # Offset from the right edge
            tooltip_y = self.screen.get_height() - tooltip_height - 10  # Offset from the bottom edge
        else:
            tooltip_x, tooltip_y = position  # Use the provided position

        # Render tooltip background
        tooltip_surface = pygame.Surface((tooltip_width, tooltip_height))
        tooltip_surface.set_alpha(200)  # Transparency
        tooltip_surface.fill((0, 0, 0))  # Black background

        # Render text on the tooltip
        for i, line in enumerate(lines):
            line_surface = font.render(line, True, (255, 255, 255))  # White text
            tooltip_surface.blit(line_surface, (padding, i * line_height + padding))

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
        for i in range(sheet_height // frame_height):  # Loop through vertical frames
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
        #print(f"World Position: {position}")
        #print(f"Screen Position: ({screen_x}, {screen_y})")

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
            "duration": duration
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
                self.screen.blit(frame, (screen_x - frame.get_width() // 2, screen_y - frame.get_height() // 2))
                updated_animations.append(animation)

        # Update the active animations
        self.active_animations = updated_animations






# Constants for the menu screen
FONT_NAME = "Arial"
TITLE = "Simulation Game"
WELCOME_TEXT = "Welcome to the Multi-Agent Faction Wars Simulation Game!"
HYPERPARAMS_TEXT = "Please set your hyperparameters and preferences below."
TRAIN_EVALUATE_TEXT = "Choose Mode (Training or Evaluation):"
TENSORBOARD_TEXT = "Run TensorBoard after Simulation?"
AUTOMATIC_TB_TEXT = "Automatically open TensorBoard after simulation?"

# Define some colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (50, 50, 50)  # Dark grey background for buttons
BLUE = (50, 100, 255)
GREEN = (50, 255, 50)
RED = (255, 50, 50)

# Define icons










#    __  __    _    ___ _   _   __  __ _____ _   _ _   _ 
#   |  \/  |  / \  |_ _| \ | | |  \/  | ____| \ | | | | |
#   | |\/| | / _ \  | ||  \| | | |\/| |  _| |  \| | | | |
#   | |  | |/ ___ \ | || |\  | | |  | | |___| |\  | |_| |
#   |_|  |_/_/   \_\___|_| \_| |_|  |_|_____|_| \_|\___/ 
#                                                        
check_icon = pygame.font.SysFont(None, 40).render('✔', True, GREEN)
cross_icon = pygame.font.SysFont(None, 40).render('❌', True, RED)

class MenuRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(FONT_NAME, 24)
        self.selected_mode = None  #  Store mode persistently
        print("MenuRenderer initialised")

    def draw_text(self, surface, text, font, size, colour, x, y):
        """Helper function to draw text on the screen"""
        font_obj = pygame.font.SysFont(font, size)
        text_surface = font_obj.render(text, True, colour)
        text_rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, text_rect)

    def create_button(self, surface, text, font, size, colour, hover_colour, click_colour, x, y, width, height, state='normal', icon=None):
        """Helper function to create buttons with hover and click feedback, and an icon (check or cross)"""
        button_rect = pygame.Rect(x, y, width, height)

        # Check for mouse hover
        if button_rect.collidepoint(pygame.mouse.get_pos()):
            if state == 'normal':
                pygame.draw.rect(surface, hover_colour, button_rect)  # Hover state
            elif state == 'clicked':
                pygame.draw.rect(surface, click_colour, button_rect)  # Clicked state
        else:
            pygame.draw.rect(surface, colour, button_rect)  # Normal state

        # Draw the main text
        self.draw_text(surface, text, font, size, WHITE, x + width / 2, y + height / 2)

        # If there's an icon, display it with padding
        if icon:
            icon_size = 40  # Increase icon size
            icon_x = x + width - icon_size - 10  # Add padding on the right
            icon_y = y + (height // 2) - (icon_size // 2)  # Centre the icon vertically
            surface.blit(icon, (icon_x, icon_y))

        return button_rect

    def render_menu(self, tensorboard_enabled, auto_tensorboard_enabled, mode, start_game_callback):
        """Renders the menu and handles button states."""
        self.screen.fill(BLACK)  # Set the background to black

        # Initialise icons for buttons
        check_icon = pygame.font.SysFont(FONT_NAME, 40).render('✔', True, GREEN)
        cross_icon = pygame.font.SysFont(FONT_NAME, 40).render('❌', True, RED)

        # Training vs Evaluation Mode
        train_button_rect = self.create_button(
            self.screen, "Training", FONT_NAME, 20, GREEN, (0, 255, 0), (0, 200, 0),
            SCREEN_WIDTH // 4, 330, 150, 50
        )
        evaluate_button_rect = self.create_button(
            self.screen, "Evaluation", FONT_NAME, 20, BLUE, (0, 0, 255), (0, 0, 200),
            SCREEN_WIDTH // 2, 330, 150, 50
        )
        GRAY = (100, 100, 100)  # Light grey
        # Start Simulation Button (Disabled if no mode selected)
        start_button_rect = self.create_button(
            self.screen, "Start Simulation", FONT_NAME, 25, BLUE if self.selected_mode else GRAY, 
            (50, 100, 255) if self.selected_mode else (100, 100, 100), 
            (30, 70, 200) if self.selected_mode else (70, 70, 70),
            SCREEN_WIDTH // 3, 600, 200, 60
        )

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if train_button_rect.collidepoint(event.pos):
                    self.selected_mode = 'train'  #  Store mode 
                    print("[INFO] Training mode selected.")

                elif evaluate_button_rect.collidepoint(event.pos):
                    self.selected_mode = 'evaluate'  #  Store mode 
                    print("[INFO] Evaluation mode selected.")

                elif start_button_rect.collidepoint(event.pos) and self.selected_mode:
                    print("[INFO] Starting game in mode:", self.selected_mode)
                    start_game_callback(self.selected_mode, tensorboard_enabled, auto_tensorboard_enabled)
                    return False  #  Exit menu and start game

        # Update the display
        pygame.display.flip()
        return True  #  Stay in the menu until "Start Simulation" is clicked