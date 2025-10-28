"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
from RENDER.Common import (
    SETTINGS_FONT,
    WHITE,
    BLACK,
    BLUE,
    GREEN,
    RED,
    GREY,
    DARK_GREY,
    get_font,
)
import UTILITIES.utils_config as utils_config
from UTILITIES.settings_manager import settings_manager
import traceback
import sys


#    ____       _   _   _                   __  __
#   / ___|  ___| |_| |_(_)_ __   __ _ ___  |  \/  | ___ _ __  _   _
#   \___ \ / _ \ __| __| | '_ \ / _` / __| | |\/| |/ _ \ '_ \| | | |
#    ___) |  __/ |_| |_| | | | | (_| \__ \ | |  | |  __/ | | | |_| |
#   |____/ \___|\__|\__|_|_| |_|\__, |___/ |_|  |_|\___|_| |_|\__,_|
#                               |___/

SettingsTITLE = "Settings Menu"


class SettingsMenuRenderer:
    """Class for rendering the settings menu."""

    def __init__(self, screen):
        try:
            self.screen = screen

            # Get screen dimensions for responsive scaling
            self.screen_width = screen.get_width()
            self.screen_height = screen.get_height()

            # Define layout parameters (scaled based on screen size)
            self.sidebar_width = int(self.screen_width * 0.175)  # ~175px at default
            self.content_start_x = int(self.screen_width * 0.22)  # ~210px at default
            self.value_x = int(self.screen_width * 0.55)  # ~550px at default
            self.button_x = int(self.screen_width * 0.68)  # ~700px at default
            self.help_button_x = int(self.screen_width * 0.75)  # ~750px at default

            # Vertical layout
            self.title_y = 20
            self.sidebar_start_y = 80
            self.item_height = int(self.screen_height * 0.055)  # ~60px at default
            self.buttons_y = int(self.screen_height * 0.87)  # ~500px at 576px height
            self.buttons_spacing = int(self.screen_width * 0.15)  # ~150px at default

            self.font = get_font(24, SETTINGS_FONT, False)
            self.selected_category = "system"
            self.saved = False
            self.input_mode = False
            self.input_field = None
            self.input_text = ""
            self.cursor_visible = True
            self.cursor_timer = 0
            self.show_help = False
            self.help_timer = 0
            self.help_scroll_offset = 0
            self.max_help_scroll = 0

            # Scrolling variables
            self.scroll_offset = 0
            self.max_scroll = 0
            self.scroll_speed = 60  # pixels per scroll
            self.scroll_bar_width = 20
            self.is_scrolling = False
            self.scroll_bar_rect = None

            pygame.display.set_caption(SettingsTITLE)

            # Define category labels
            self.sidebar_items = [
                "system",
                "debugging",
                "episode settings",
                "screen",
                "world",
                "resources",
                "agent",
                "faction",
                "ai training",
                "curriculum learning",
                "advanced loss",
            ]

            # Store config state and original defaults
            self.settings_by_category = {}
            self.defaults = {}

            raw_settings = {
                "system": [
                    ("Headless Mode", "HEADLESS_MODE"),
                    ("Force Dependency Check", "FORCE_DEPENDENCY_CHECK"),
                ],
                "debugging": [
                    ("TensorBoard", "ENABLE_TENSORBOARD"),
                    ("Logging", "ENABLE_LOGGING"),
                    ("Plots & CSVs", "ENABLE_PLOTS"),
                    ("Performance Profiling", "ENABLE_PROFILE_BOOL"),
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
                    ("Agent Spawn Radius", "HQ_Agent_Spawn_RADIUS", 2),
                    ("Faction Count", "FACTON_COUNT", 1),
                    ("Initial Gatherers", "INITAL_GATHERER_COUNT", 1),
                    ("Initial Peacekeepers", "INITAL_PEACEKEEPER_COUNT", 1),
                ],
                "ai training": [
                    ("PPO Learning Rate", "INITIAL_LEARNING_RATE_PPO", 0.0001),
                    ("HQ Learning Rate", "INITIAL_LEARNING_RATE_HQ", 0.0001),
                    ("Min Learning Rate", "MIN_LEARNING_RATE", 0.000001),
                    ("Learning Rate Decay", "LEARNING_RATE_DECAY", 0.01),
                    ("LR Step Size", "LEARNING_RATE_STEP_SIZE", 100),
                    ("Batch Size", "BATCH_SIZE", 10),
                    ("Min Memory Size", "MIN_MEMORY_SIZE", 10),
                    ("GAE Lambda", "GAE_LAMBDA", 0.01),
                    ("Value Loss Coeff", "VALUE_LOSS_COEFF", 0.1),
                    ("Entropy Coeff", "ENTROPY_COEFF", 0.001),
                    ("Gradient Clip Norm", "GRADIENT_CLIP_NORM", 0.1),
                    ("PPO Clip Ratio", "PPO_CLIP_RATIO", 0.01),
                ],
                "curriculum learning": [
                    ("Enable Curriculum", "ENABLE_CURRICULUM_LEARNING"),
                    ("Initial Resource Rate", "INITIAL_RESOURCE_SPAWN_RATE", 0.01),
                    ("Final Resource Rate", "FINAL_RESOURCE_SPAWN_RATE", 0.01),
                ],
                "advanced loss": [
                    ("Use Huber Loss", "USE_HUBER_LOSS"),
                ],
            }

            # Initialize settings with error handling
            try:
                for category, fields in raw_settings.items():
                    self.settings_by_category[category] = []
                    for item in fields:
                        try:
                            label, key = item[:2]
                            value = getattr(utils_config, key)
                            self.defaults[key] = value
                            setting = {"label": label, "key": key, "value": value}
                            if len(item) == 3:
                                setting["step"] = item[2]
                            else:
                                setting["options"] = [True, False]
                            self.settings_by_category[category].append(setting)
                        except AttributeError as e:
                            print(
                                f"[WARNING] Config key '{key}' not found in utils_config, using default value"
                            )
                            # Use sensible defaults for missing config keys
                            if len(item) == 3:
                                setting = {
                                    "label": label,
                                    "key": key,
                                    "value": item[2],
                                    "step": item[2],
                                }
                            else:
                                setting = {
                                    "label": label,
                                    "key": key,
                                    "value": False,
                                    "options": [True, False],
                                }
                            self.settings_by_category[category].append(setting)
                        except Exception as e:
                            print(f"[ERROR] Failed to initialize setting {item}: {e}")
                            continue
            except Exception as e:
                print(f"[ERROR] Failed to initialize settings: {e}")
                traceback.print_exc()

            # Add custom step sizes for AI training parameters
            try:
                self.add_custom_step_sizes()
            except Exception as e:
                print(f"[ERROR] Failed to add custom step sizes: {e}")

            # Calculate max scroll based on number of categories
            self.calculate_scroll_bounds()

        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to initialize SettingsMenuRenderer: {e}")
            traceback.print_exc()
            # Set minimal defaults to prevent complete failure
            self.screen = screen
            self.selected_category = "debugging"
            self.settings_by_category = {"debugging": []}
            self.defaults = {}
            self.scroll_offset = 0
            self.max_scroll = 0

    def calculate_scroll_bounds(self):
        """Calculate the maximum scroll offset based on screen height and number of categories."""
        try:
            # Screen height available for categories
            available_height = self.buttons_y - self.sidebar_start_y - 40
            # Height needed for all categories
            total_height = len(self.sidebar_items) * self.item_height
            # Maximum scroll offset
            self.max_scroll = max(0, total_height - available_height)
            # Ensure scroll offset is within bounds
            self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
        except Exception as e:
            print(f"[ERROR] Failed to calculate scroll bounds: {e}")
            self.max_scroll = 0
            self.scroll_offset = 0

    def handle_scroll_input(self, direction):
        """Handle scroll input (mouse wheel or keyboard)."""
        try:
            if direction > 0:  # Scroll up
                self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)
            else:  # Scroll down
                self.scroll_offset = min(
                    self.max_scroll, self.scroll_offset + self.scroll_speed
                )
        except Exception as e:
            print(f"[ERROR] Failed to handle scroll input: {e}")

    def is_category_visible(self, index):
        """Check if a category at the given index is currently visible on screen."""
        try:
            category_y = (
                self.sidebar_start_y + index * self.item_height - self.scroll_offset
            )
            visible_top = self.sidebar_start_y
            visible_bottom = self.buttons_y - 40
            return visible_top <= category_y <= visible_bottom
        except Exception as e:
            print(f"[ERROR] Failed to check category visibility: {e}")
            return True

    def draw_scroll_bar(self):
        """Draw the scroll bar on the right side of the sidebar."""
        try:
            if self.max_scroll <= 0:
                return  # No scrolling needed

            # Calculate scroll bar dimensions
            sidebar_height = self.buttons_y - self.sidebar_start_y - 40
            scroll_bar_height = max(
                30,
                (sidebar_height / (self.max_scroll + sidebar_height)) * sidebar_height,
            )

            # Calculate scroll bar position
            scroll_ratio = (
                self.scroll_offset / self.max_scroll if self.max_scroll > 0 else 0
            )
            scroll_bar_y = self.sidebar_start_y + scroll_ratio * (
                sidebar_height - scroll_bar_height
            )

            # Draw scroll bar background
            scroll_bg_rect = pygame.Rect(
                20 + self.sidebar_width - self.scroll_bar_width,
                self.sidebar_start_y,
                self.scroll_bar_width,
                sidebar_height,
            )
            pygame.draw.rect(self.screen, DARK_GREY, scroll_bg_rect)

            # Draw scroll bar handle
            self.scroll_bar_rect = pygame.Rect(
                20 + self.sidebar_width - self.scroll_bar_width,
                scroll_bar_y,
                self.scroll_bar_width,
                scroll_bar_height,
            )
            pygame.draw.rect(self.screen, GREY, self.scroll_bar_rect)
            pygame.draw.rect(self.screen, WHITE, self.scroll_bar_rect, 2)

            # Draw scroll indicators
            if self.scroll_offset > 0:
                # Up arrow
                up_arrow_rect = pygame.Rect(
                    20 + self.sidebar_width - self.scroll_bar_width,
                    self.sidebar_start_y - 5,
                    self.scroll_bar_width,
                    10,
                )
                pygame.draw.rect(self.screen, WHITE, up_arrow_rect)
                self.draw_text(
                    "▲",
                    12,
                    BLACK,
                    20 + self.sidebar_width - self.scroll_bar_width + 5,
                    self.sidebar_start_y - 3,
                )

            if self.scroll_offset < self.max_scroll:
                # Down arrow
                down_arrow_rect = pygame.Rect(
                    20 + self.sidebar_width - self.scroll_bar_width,
                    self.buttons_y - 30,
                    self.scroll_bar_width,
                    10,
                )
                pygame.draw.rect(self.screen, WHITE, down_arrow_rect)
                self.draw_text(
                    "▼",
                    12,
                    BLACK,
                    20 + self.sidebar_width - self.scroll_bar_width + 5,
                    self.buttons_y - 28,
                )

        except Exception as e:
            print(f"[ERROR] Failed to draw scroll bar: {e}")

    def handle_scroll_bar_click(self, mouse_pos):
        """Handle clicks on the scroll bar."""
        try:
            if not self.scroll_bar_rect or self.max_scroll <= 0:
                return False

            # Check if click is on scroll bar background
            sidebar_height = self.buttons_y - self.sidebar_start_y - 40
            scroll_bg_rect = pygame.Rect(
                20 + self.sidebar_width - self.scroll_bar_width,
                self.sidebar_start_y,
                self.scroll_bar_width,
                sidebar_height,
            )

            if scroll_bg_rect.collidepoint(mouse_pos):
                # Calculate new scroll position based on click
                click_y = mouse_pos[1] - self.sidebar_start_y
                new_scroll_ratio = click_y / sidebar_height
                self.scroll_offset = int(new_scroll_ratio * self.max_scroll)
                self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
                return True

            return False
        except Exception as e:
            print(f"[ERROR] Failed to handle scroll bar click: {e}")
            return False

    def add_custom_step_sizes(self):
        """Add custom step sizes for AI training parameters that need finer control."""
        try:
            custom_steps = {
                "INITIAL_LEARNING_RATE_PPO": 0.0001,
                "INITIAL_LEARNING_RATE_HQ": 0.0001,
                "MIN_LEARNING_RATE": 0.000001,
                "LEARNING_RATE_DECAY": 0.01,
                "LEARNING_RATE_STEP_SIZE": 50,
                "BATCH_SIZE": 8,
                "MIN_MEMORY_SIZE": 16,
                "GAE_LAMBDA": 0.01,
                "VALUE_LOSS_COEFF": 0.05,
                "ENTROPY_COEFF": 0.001,
                "GRADIENT_CLIP_NORM": 0.05,
                "PPO_CLIP_RATIO": 0.01,
                "INITIAL_RESOURCE_SPAWN_RATE": 0.01,
                "FINAL_RESOURCE_SPAWN_RATE": 0.01,
            }

            # Update step sizes for existing settings
            for category in self.settings_by_category.values():
                for setting in category:
                    try:
                        if setting["key"] in custom_steps:
                            setting["step"] = custom_steps[setting["key"]]
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to set step size for {setting.get('key', 'unknown')}: {e}"
                        )
                        continue
        except Exception as e:
            print(f"[ERROR] Failed to add custom step sizes: {e}")
            traceback.print_exc()

    def get_parameter_description(self, key):
        """Get helpful description for AI training parameters."""
        try:
            descriptions = {
                "INITIAL_LEARNING_RATE_PPO": "Learning rate for PPO agents (0.0001-0.001 recommended)",
                "INITIAL_LEARNING_RATE_HQ": "Learning rate for HQ strategy networks (0.0001-0.001 recommended)",
                "MIN_LEARNING_RATE": "Minimum learning rate to prevent overfitting",
                "LEARNING_RATE_DECAY": "How quickly learning rate decreases (0.95-0.99 recommended)",
                "LEARNING_RATE_STEP_SIZE": "Steps between learning rate updates (500-1000 recommended)",
                "BATCH_SIZE": "Training batch size (32-128 recommended)",
                "MIN_MEMORY_SIZE": "Minimum experiences before training (128+ recommended)",
                "GAE_LAMBDA": "Generalized Advantage Estimation parameter (0.95-0.99 recommended)",
                "VALUE_LOSS_COEFF": "Value function loss weight (0.5 recommended)",
                "ENTROPY_COEFF": "Exploration bonus weight (0.01-0.1 recommended)",
                "GRADIENT_CLIP_NORM": "Gradient clipping for stability (0.5 recommended)",
                "PPO_CLIP_RATIO": "PPO policy clipping (0.1-0.2 recommended)",
                "ENABLE_CURRICULUM_LEARNING": "Gradually increase difficulty during training",
                "INITIAL_RESOURCE_SPAWN_RATE": "Starting resource density (0.3 recommended)",
                "FINAL_RESOURCE_SPAWN_RATE": "Final resource density (1.0 recommended)",
                "USE_HUBER_LOSS": "Use Huber loss for value function (more robust than MSE)",
            }
            return descriptions.get(key, "")
        except Exception as e:
            print(f"[ERROR] Failed to get parameter description for {key}: {e}")
            return ""

    def show_parameter_tooltips(self):
        """Show tooltips for AI training parameters when hovering."""
        try:
            mouse_pos = pygame.mouse.get_pos()
            settings = self.settings_by_category.get(self.selected_category, [])

            for i, setting in enumerate(settings):
                try:
                    y = self.sidebar_start_y + i * self.item_height
                    # Check if mouse is hovering over the parameter label
                    label_rect = pygame.Rect(self.content_start_x, y, 350, 30)
                    if label_rect.collidepoint(mouse_pos):
                        description = self.get_parameter_description(setting["key"])
                        if description:
                            # Show tooltip
                            self.draw_tooltip(
                                description, mouse_pos[0], mouse_pos[1] - 40
                            )
                except Exception as e:
                    print(f"[WARNING] Failed to show tooltip for setting {i}: {e}")
                    continue
        except Exception as e:
            print(f"[ERROR] Failed to show parameter tooltips: {e}")

    def draw_tooltip(self, text, x, y):
        """Draw a tooltip with the given text at the specified position."""
        try:
            # Split text into lines if too long
            max_width = 300
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) * 8 > max_width:  # Approximate character width
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line

            if current_line:
                lines.append(current_line)

            # Calculate tooltip dimensions
            line_height = 20
            tooltip_height = len(lines) * line_height + 10
            tooltip_width = max(len(line) * 8 for line in lines) + 20

            # Ensure tooltip stays on screen
            if x + tooltip_width > 800:
                x = 800 - tooltip_width
            if y < 0:
                y = 0

            # Draw tooltip background
            tooltip_rect = pygame.Rect(x, y, tooltip_width, tooltip_height)
            pygame.draw.rect(self.screen, (50, 50, 50), tooltip_rect)
            pygame.draw.rect(self.screen, WHITE, tooltip_rect, 2)

            # Draw tooltip text
            for i, line in enumerate(lines):
                try:
                    self.draw_text(line, 16, WHITE, x + 10, y + 5 + i * line_height)
                except Exception as e:
                    print(f"[WARNING] Failed to draw tooltip line {i}: {e}")
                    continue
        except Exception as e:
            print(f"[ERROR] Failed to draw tooltip: {e}")

    def draw_help_overlay(self):
        """Draw a comprehensive help overlay for AI training parameters with scrolling."""
        try:
            # Semi-transparent background
            overlay_surface = pygame.Surface((self.screen_width, self.screen_height))
            overlay_surface.set_alpha(200)
            overlay_surface.fill((0, 0, 0))
            self.screen.blit(overlay_surface, (0, 0))

            # Help content
            help_content = [
                "AI Training Parameters Help",
                "",
                "Learning Rates:",
                "• PPO Learning Rate: Start with 0.0001 for stability",
                "• HQ Learning Rate: Use 0.0001 for strategic learning",
                "• Min Learning Rate: Minimum LR to prevent stagnation",
                "• Learning Rate Decay: 0.95-0.99 for gradual reduction",
                "• LR Step Size: How often to decay the learning rate",
                "",
                "Training Parameters:",
                "• Batch Size: 32-128 for optimal training, larger = more stable",
                "• Memory Size: Minimum experiences before training starts",
                "• GAE Lambda: 0.95-0.99 for Generalized Advantage Estimation",
                "",
                "Loss Coefficients:",
                "• Value Loss Coeff: Weight of value function loss (0.5 recommended)",
                "• Entropy Coeff: Exploration bonus (0.01-0.1)",
                "• Gradient Clip Norm: Clips gradients for stability (0.5)",
                "• PPO Clip Ratio: Policy clipping range (0.1-0.2)",
                "",
                "Curriculum Learning:",
                "• Enable Curriculum: Gradually increase difficulty over time",
                "• Initial Resource Rate: Starting resource density (0.3 = 30%)",
                "• Final Resource Rate: Ending resource density (1.0 = 100%)",
                "",
                "Advanced Loss:",
                "• Use Huber Loss: More robust than MSE for value learning",
                "  Reduces impact of outliers during training",
                "  Better for noisy value estimates",
                "",
                "Keyboard Shortcuts:",
                "• ESC: Close help / Toggle pause in game",
                "• Mouse Wheel: Scroll through settings",
                "• Home/End: Jump to top/bottom of category list",
                "",
                "Press ESC or click Close to exit",
            ]

            # Calculate scroll bounds
            total_height = len(help_content) * 25
            visible_height = self.screen_height - 100  # Space for title and buttons
            self.max_help_scroll = max(0, total_height - visible_height)

            # Clamp scroll offset
            self.help_scroll_offset = max(
                0, min(self.help_scroll_offset, self.max_help_scroll)
            )

            # Draw help content with scroll offset
            y_offset = 50 - self.help_scroll_offset
            for line in help_content:
                try:
                    if line.startswith("•"):
                        color = GREEN
                        size = 16
                    elif line in [
                        "AI Training Parameters Help",
                        "Learning Rates:",
                        "Training Parameters:",
                        "Loss Coefficients:",
                        "Curriculum Learning:",
                        "Advanced Loss:",
                    ]:
                        color = WHITE
                        size = 20
                        if line == "AI Training Parameters Help":
                            size = 24
                    else:
                        color = GREY
                        size = 16

                    # Only draw if visible
                    if 0 <= y_offset <= self.screen_height:
                        self.draw_text(line, size, color, 50, y_offset)
                    y_offset += 25
                except Exception as e:
                    print(f"[WARNING] Failed to draw help line: {e}")
                    y_offset += 25
                    continue

            # Draw scroll bar if needed
            if self.max_help_scroll > 0:
                self.draw_help_scroll_bar()

            # Close button
            try:
                close_btn = pygame.Rect(int(self.screen_width * 0.73), 20, 80, 30)
                pygame.draw.rect(self.screen, RED, close_btn)
                self.draw_text(
                    "Close", 16, WHITE, int(self.screen_width * 0.73) + 20, 25
                )

                # Handle close button click
                mouse_pos = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed()[0] and close_btn.collidepoint(mouse_pos):
                    self.show_help = False
            except Exception as e:
                print(f"[ERROR] Failed to draw close button: {e}")

        except Exception as e:
            print(f"[ERROR] Failed to draw help overlay: {e}")
            traceback.print_exc()

    def draw_help_scroll_bar(self):
        """Draw scroll bar for help overlay."""
        try:
            scroll_bar_x = self.screen_width - 30
            scroll_bar_width = 20
            scroll_bar_height = self.screen_height - 100
            scroll_bar_bg_y = 50

            # Calculate handle size and position
            handle_height = max(
                30,
                (scroll_bar_height / (self.max_help_scroll + scroll_bar_height))
                * scroll_bar_height,
            )
            scroll_ratio = (
                self.help_scroll_offset / self.max_help_scroll
                if self.max_help_scroll > 0
                else 0
            )
            handle_y = scroll_bar_bg_y + scroll_ratio * (
                scroll_bar_height - handle_height
            )

            # Draw scroll bar background
            pygame.draw.rect(
                self.screen,
                DARK_GREY,
                (scroll_bar_x, scroll_bar_bg_y, scroll_bar_width, scroll_bar_height),
            )

            # Draw scroll bar handle
            pygame.draw.rect(
                self.screen,
                GREY,
                (scroll_bar_x, handle_y, scroll_bar_width, handle_height),
            )
            pygame.draw.rect(
                self.screen,
                WHITE,
                (scroll_bar_x, handle_y, scroll_bar_width, handle_height),
                2,
            )

        except Exception as e:
            print(f"[ERROR] Failed to draw help scroll bar: {e}")

    def draw_text(self, text, size, colour, x, y, bold=False):
        """Draw text with error handling."""
        try:
            font_obj = get_font(size, SETTINGS_FONT, bold)
            text_surface = font_obj.render(text, True, colour)
            text_rect = text_surface.get_rect(topleft=(x, y))
            self.screen.blit(text_surface, text_rect)
        except Exception as e:
            print(f"[ERROR] Failed to draw text '{text}' at ({x}, {y}): {e}")
            # Fallback: try to draw with default font
            try:
                fallback_font = pygame.font.Font(None, size)
                text_surface = fallback_font.render(text, True, colour)
                text_rect = text_surface.get_rect(topleft=(x, y))
                self.screen.blit(text_surface, text_rect)
            except Exception as fallback_e:
                print(
                    f"[CRITICAL] Failed to draw text even with fallback font: {fallback_e}"
                )

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
        icon=None,
    ):
        try:
            button_rect = pygame.Rect(x, y, width, height)
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]

            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(
                    self.screen,
                    click_colour if mouse_pressed else hover_colour,
                    button_rect,
                )
            else:
                pygame.draw.rect(self.screen, colour, button_rect)

            try:
                self.draw_text(text, size, WHITE, x + 10, y + 10)
            except Exception as e:
                print(f"[WARNING] Failed to draw button text '{text}': {e}")

            if icon:
                try:
                    icon_size = 40
                    icon_x = x + width - icon_size - 10
                    icon_y = y + (height // 2) - (icon_size // 2)
                    self.screen.blit(icon, (icon_x, icon_y))
                except Exception as e:
                    print(f"[WARNING] Failed to draw button icon: {e}")

            return button_rect
        except Exception as e:
            print(f"[ERROR] Failed to create button '{text}': {e}")
            # Return a minimal button rect as fallback
            return pygame.Rect(x, y, width, height)

    def render(self):
        try:
            self.screen.fill(BLACK)
            self.draw_text(
                "SETTINGS", 36, WHITE, 20, 20, bold=True
            )  # Larger font size for the title

            try:
                self.cursor_timer += 1
                if self.cursor_timer >= 30:
                    self.cursor_visible = not self.cursor_visible
                    self.cursor_timer = 0
            except Exception as e:
                print(f"[WARNING] Failed to update cursor timer: {e}")

            # Draw sidebar
            try:
                for idx, label in enumerate(self.sidebar_items):
                    try:
                        # Calculate position with scroll offset
                        y = (
                            self.sidebar_start_y
                            + idx * self.item_height
                            - self.scroll_offset
                        )

                        # Skip if category is not visible
                        if not self.is_category_visible(idx):
                            continue

                        selected = label == self.selected_category
                        color = GREY if selected else DARK_GREY
                        btn = self.create_button(
                            label.upper(),
                            SETTINGS_FONT,
                            20,
                            color,
                            (120, 120, 120),
                            (180, 180, 180),
                            20,
                            y,
                            self.sidebar_width,
                            40,
                        )
                        if pygame.mouse.get_pressed()[0] and btn.collidepoint(
                            pygame.mouse.get_pos()
                        ):
                            self.selected_category = label
                    except Exception as e:
                        print(f"[WARNING] Failed to draw sidebar item {label}: {e}")
                        continue

                # Draw scroll bar
                self.draw_scroll_bar()

            except Exception as e:
                print(f"[ERROR] Failed to draw sidebar: {e}")

            # Draw settings
            try:
                settings = self.settings_by_category.get(self.selected_category, [])
                step_buttons = []
                for i, setting in enumerate(settings):
                    try:
                        y = self.sidebar_start_y + i * self.item_height
                        label = setting["label"]
                        val = setting["value"]
                        self.draw_text(f"{label}:", 20, WHITE, self.content_start_x, y)

                        if self.input_mode and self.input_field == setting:
                            display_text = self.input_text + (
                                "|" if self.cursor_visible else ""
                            )
                            self.draw_text(display_text, 20, WHITE, self.value_x, y)
                        else:
                            self.draw_text(
                                str(val), 20, GREEN if val else RED, self.value_x, y
                            )

                        if "options" in setting:
                            try:
                                button_size = int(self.screen_width * 0.04)
                                toggle_btn = pygame.Rect(
                                    self.button_x, y, button_size, 30
                                )
                                pygame.draw.rect(self.screen, DARK_GREY, toggle_btn)
                                if setting["value"]:
                                    pygame.draw.rect(self.screen, GREEN, toggle_btn)
                                    self.draw_text(
                                        "ON",
                                        18,
                                        BLACK,
                                        toggle_btn.x + 5,
                                        toggle_btn.y + 5,
                                    )
                                else:
                                    pygame.draw.rect(self.screen, RED, toggle_btn)
                                    self.draw_text(
                                        "OFF",
                                        18,
                                        BLACK,
                                        toggle_btn.x + 2,
                                        toggle_btn.y + 5,
                                    )

                                step_buttons.append(("toggle", toggle_btn, setting))
                            except Exception as e:
                                print(
                                    f"[WARNING] Failed to draw toggle button for {label}: {e}"
                                )

                        elif "step" in setting:
                            try:
                                button_size = int(self.screen_width * 0.03)
                                minus_btn = pygame.Rect(
                                    self.button_x, y, button_size, 30
                                )
                                plus_btn = pygame.Rect(
                                    self.button_x + button_size + 5, y, button_size, 30
                                )
                                default_btn = pygame.Rect(
                                    self.button_x + (button_size * 2) + 10,
                                    y,
                                    int(self.screen_width * 0.09),
                                    30,
                                )
                                pygame.draw.rect(self.screen, RED, minus_btn)
                                pygame.draw.rect(self.screen, GREEN, plus_btn)
                                pygame.draw.rect(self.screen, GREY, default_btn)
                                self.draw_text(
                                    "-", 20, WHITE, minus_btn.x + 10, minus_btn.y + 5
                                )
                                self.draw_text(
                                    "+", 20, WHITE, plus_btn.x + 10, plus_btn.y + 5
                                )
                                self.draw_text(
                                    "Reset",
                                    18,
                                    WHITE,
                                    default_btn.x + 10,
                                    default_btn.y + 5,
                                )
                                step_buttons.append(("minus", minus_btn, setting))
                                step_buttons.append(("plus", plus_btn, setting))
                                step_buttons.append(("reset", default_btn, setting))
                            except Exception as e:
                                print(
                                    f"[WARNING] Failed to draw step buttons for {label}: {e}"
                                )
                    except Exception as e:
                        print(f"[WARNING] Failed to draw setting {i}: {e}")
                        continue
            except Exception as e:
                print(f"[ERROR] Failed to draw settings: {e}")
                settings = []
                step_buttons = []

            # Draw main buttons
            try:
                button_y = self.buttons_y
                button_height = 50
                button_w1 = int(self.screen_width * 0.15)
                button_w2 = int(self.screen_width * 0.25)
                button_w3 = int(self.screen_width * 0.17)

                back_btn = self.create_button(
                    "Back",
                    SETTINGS_FONT,
                    20,
                    GREY,
                    (180, 180, 180),
                    (120, 120, 120),
                    self.content_start_x,
                    button_y,
                    button_w1,
                    button_height,
                )
                save_return_btn = self.create_button(
                    "Save and Return",
                    SETTINGS_FONT,
                    20,
                    BLUE,
                    (80, 80, 255),
                    (50, 50, 200),
                    self.content_start_x + button_w1 + 10,
                    button_y,
                    button_w2,
                    button_height,
                )
                reset_all_btn = self.create_button(
                    "Reset All",
                    SETTINGS_FONT,
                    20,
                    GREY,
                    (180, 180, 180),
                    (120, 120, 120),
                    self.content_start_x + button_w1 + button_w2 + 20,
                    button_y,
                    button_w3,
                    button_height,
                )

                # Add help button for AI training categories
                help_btn = None
                if self.selected_category in [
                    "ai training",
                    "curriculum learning",
                    "advanced loss",
                ]:
                    try:
                        help_btn = self.create_button(
                            "Help",
                            SETTINGS_FONT,
                            18,
                            GREEN,
                            (100, 200, 100),
                            (80, 160, 80),
                            self.content_start_x
                            + button_w1
                            + button_w2
                            + button_w3
                            + 30,
                            button_y,
                            80,
                            50,
                        )
                    except Exception as e:
                        print(f"[WARNING] Failed to create help button: {e}")
            except Exception as e:
                print(f"[ERROR] Failed to draw main buttons: {e}")
                # Create minimal fallback buttons
                back_btn = pygame.Rect(
                    self.content_start_x,
                    self.buttons_y,
                    int(self.screen_width * 0.15),
                    50,
                )
                save_return_btn = pygame.Rect(
                    self.content_start_x + int(self.screen_width * 0.17),
                    self.buttons_y,
                    int(self.screen_width * 0.25),
                    50,
                )
                reset_all_btn = pygame.Rect(
                    self.content_start_x + int(self.screen_width * 0.44),
                    self.buttons_y,
                    int(self.screen_width * 0.17),
                    50,
                )
                help_btn = None

            # Show tooltips for AI training parameters
            try:
                if self.selected_category in [
                    "ai training",
                    "curriculum learning",
                    "advanced loss",
                ]:
                    self.show_parameter_tooltips()
            except Exception as e:
                print(f"[WARNING] Failed to show parameter tooltips: {e}")

            # Show help overlay if help button was clicked
            try:
                if self.show_help:
                    self.draw_help_overlay()
            except Exception as e:
                print(f"[ERROR] Failed to show help overlay: {e}")
                self.show_help = False  # Reset if it fails

            # Hover tooltips are handled elsewhere, no static tips needed

            pygame.display.update()

            # Event handling
            try:
                for event in pygame.event.get():
                    try:
                        if event.type == pygame.QUIT:
                            cleanup(QUIT=True)

                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            try:
                                mouse_x, mouse_y = pygame.mouse.get_pos()

                                # Check for scroll bar clicks first
                                if self.handle_scroll_bar_click((mouse_x, mouse_y)):
                                    continue  # Skip other click handling if scroll bar was clicked

                                for i, setting in enumerate(settings):
                                    try:
                                        y = self.sidebar_start_y + i * self.item_height
                                        # matches where you draw the values
                                        value_rect = pygame.Rect(
                                            self.value_x, y, 80, 30
                                        )
                                        if (
                                            value_rect.collidepoint(mouse_x, mouse_y)
                                            and "step" in setting
                                        ):
                                            self.input_mode = True
                                            self.input_field = setting
                                            self.input_text = str(setting["value"])
                                            clicked_input = True
                                    except Exception as e:
                                        print(
                                            f"[WARNING] Failed to handle setting click {i}: {e}"
                                        )
                                        continue

                                if back_btn.collidepoint(event.pos):
                                    return False
                                if save_return_btn.collidepoint(event.pos):
                                    # Apply any pending input before saving
                                    try:
                                        if self.input_mode and self.input_field:
                                            try:
                                                value = (
                                                    float(self.input_text)
                                                    if "." in self.input_text
                                                    else int(self.input_text)
                                                )
                                                self.input_field["value"] = value
                                            except ValueError:
                                                print(
                                                    f"Invalid input for {self.input_field['key']}. Invalid input: {self.input_text}"
                                                )
                                                pass  # Invalid input, keep the old value
                                    except Exception as e:
                                        print(
                                            f"[ERROR] Failed to process input before saving: {e}"
                                        )

                                    # Save all settings to persistent storage
                                    try:
                                        settings_dict = {}
                                        for (
                                            category_settings
                                        ) in self.settings_by_category.values():
                                            for setting in category_settings:
                                                try:
                                                    settings_dict[setting["key"]] = (
                                                        setting["value"]
                                                    )
                                                except Exception as e:
                                                    print(
                                                        f"[WARNING] Failed to save setting {setting.get('key', 'unknown')}: {e}"
                                                    )

                                        settings_manager.update_config_settings(
                                            settings_dict
                                        )
                                        print(
                                            f"[INFO] Saved {len(settings_dict)} settings to persistent storage"
                                        )
                                    except Exception as e:
                                        print(
                                            f"[ERROR] Failed to save settings to persistent storage: {e}"
                                        )

                                    self.saved = True
                                    return False
                                if reset_all_btn.collidepoint(event.pos):
                                    try:
                                        for (
                                            settings
                                        ) in self.settings_by_category.values():
                                            for setting in settings:
                                                try:
                                                    setting["value"] = (
                                                        self.defaults.get(
                                                            setting["key"],
                                                            setting["value"],
                                                        )
                                                    )
                                                except Exception as e:
                                                    print(
                                                        f"[WARNING] Failed to reset setting {setting.get('key', 'unknown')}: {e}"
                                                    )
                                                    continue
                                    except Exception as e:
                                        print(
                                            f"[ERROR] Failed to reset all settings: {e}"
                                        )

                                # Handle help button click
                                if self.selected_category in [
                                    "ai training",
                                    "curriculum learning",
                                    "advanced loss",
                                ]:
                                    try:
                                        help_btn_rect = pygame.Rect(
                                            self.content_start_x
                                            + button_w1
                                            + button_w2
                                            + button_w3
                                            + 30,
                                            button_y,
                                            80,
                                            50,
                                        )
                                        if help_btn_rect.collidepoint(event.pos):
                                            self.show_help = not self.show_help
                                            self.help_timer = 0
                                    except Exception as e:
                                        print(
                                            f"[WARNING] Failed to handle help button click: {e}"
                                        )

                                for action, rect, setting in step_buttons:
                                    try:
                                        if rect.collidepoint(event.pos):
                                            if action == "toggle":
                                                try:
                                                    options = setting["options"]
                                                    current_index = options.index(
                                                        setting["value"]
                                                    )
                                                    setting["value"] = options[
                                                        (current_index + 1)
                                                        % len(options)
                                                    ]
                                                except Exception as e:
                                                    print(
                                                        f"[WARNING] Failed to toggle setting: {e}"
                                                    )
                                            elif action == "minus":
                                                try:
                                                    setting["value"] = round(
                                                        setting["value"]
                                                        - setting["step"],
                                                        3,
                                                    )
                                                except Exception as e:
                                                    print(
                                                        f"[WARNING] Failed to decrease setting value: {e}"
                                                    )
                                            elif action == "plus":
                                                try:
                                                    setting["value"] = round(
                                                        setting["value"]
                                                        + setting["step"],
                                                        3,
                                                    )
                                                except Exception as e:
                                                    print(
                                                        f"[WARNING] Failed to increase setting value: {e}"
                                                    )
                                            elif action == "reset":
                                                try:
                                                    default_val = self.defaults.get(
                                                        setting["key"], setting["value"]
                                                    )
                                                    setting["value"] = default_val
                                                except Exception as e:
                                                    print(
                                                        f"[WARNING] Failed to reset setting value: {e}"
                                                    )
                                    except Exception as e:
                                        print(
                                            f"[WARNING] Failed to handle step button action {action}: {e}"
                                        )
                                        continue

                                clicked_input = False
                                for i, setting in enumerate(settings):
                                    try:
                                        y = self.sidebar_start_y + i * self.item_height
                                        value_rect = pygame.Rect(
                                            self.value_x, y, 80, 30
                                        )
                                        if (
                                            value_rect.collidepoint(mouse_x, mouse_y)
                                            and "step" in setting
                                        ):
                                            self.input_mode = True
                                            self.input_field = setting
                                            self.input_text = str(setting["value"])
                                            clicked_input = True
                                    except Exception as e:
                                        print(
                                            f"[WARNING] Failed to handle setting input {i}: {e}"
                                        )
                                        continue

                                if not clicked_input:
                                    self.input_mode = False
                                    self.input_field = None
                            except Exception as e:
                                print(
                                    f"[ERROR] Failed to handle mouse button down: {e}"
                                )

                        # Handle mouse wheel scrolling
                        if event.type == pygame.MOUSEWHEEL:
                            try:
                                if self.show_help:
                                    # Scroll help overlay
                                    if event.y > 0:
                                        self.help_scroll_offset = max(
                                            0, self.help_scroll_offset - 40
                                        )
                                    else:
                                        self.help_scroll_offset = min(
                                            self.max_help_scroll,
                                            self.help_scroll_offset + 40,
                                        )
                                else:
                                    # Scroll category list
                                    self.handle_scroll_input(event.y)
                            except Exception as e:
                                print(f"[ERROR] Failed to handle mouse wheel: {e}")

                        if event.type == pygame.KEYDOWN and self.input_mode:
                            try:
                                if event.key == pygame.K_RETURN:
                                    try:
                                        value = (
                                            float(self.input_text)
                                            if "." in self.input_text
                                            else int(self.input_text)
                                        )
                                        self.input_field["value"] = value
                                    except ValueError:
                                        pass  # optionally show a warning or revert to old value
                                    self.input_mode = False
                                    self.input_field = None
                                    self.input_text = ""
                                elif event.key == pygame.K_BACKSPACE:
                                    self.input_text = self.input_text[:-1]
                                else:
                                    if event.unicode.isdigit() or event.unicode in [
                                        ".",
                                        "-",
                                    ]:
                                        self.input_text += event.unicode
                            except Exception as e:
                                print(f"[ERROR] Failed to handle keyboard input: {e}")
                                # Reset input mode on error
                                self.input_mode = False
                                self.input_field = None
                                self.input_text = ""

                        # Handle ESC key to close help overlay
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            try:
                                if self.show_help:
                                    self.show_help = False
                                elif self.input_mode:
                                    self.input_mode = False
                                    self.input_field = None
                                    self.input_text = ""
                            except Exception as e:
                                print(f"[WARNING] Failed to handle ESC key: {e}")

                        # Handle keyboard scrolling
                        if event.type == pygame.KEYDOWN and not self.input_mode:
                            try:
                                if self.show_help:
                                    # Handle help overlay scrolling
                                    if event.key == pygame.K_PAGEUP:
                                        self.help_scroll_offset = max(
                                            0, self.help_scroll_offset - 200
                                        )
                                    elif event.key == pygame.K_PAGEDOWN:
                                        self.help_scroll_offset = min(
                                            self.max_help_scroll,
                                            self.help_scroll_offset + 200,
                                        )
                                    elif event.key == pygame.K_HOME:
                                        self.help_scroll_offset = 0  # Jump to top
                                    elif event.key == pygame.K_END:
                                        self.help_scroll_offset = (
                                            self.max_help_scroll
                                        )  # Jump to bottom
                                else:
                                    # Handle category list scrolling
                                    if event.key == pygame.K_PAGEUP:
                                        self.handle_scroll_input(1)  # Scroll up
                                    elif event.key == pygame.K_PAGEDOWN:
                                        self.handle_scroll_input(-1)  # Scroll down
                                    elif event.key == pygame.K_HOME:
                                        self.scroll_offset = 0  # Jump to top
                                    elif event.key == pygame.K_END:
                                        self.scroll_offset = (
                                            self.max_scroll
                                        )  # Jump to bottom
                                    elif event.key == pygame.K_UP:
                                        # Find current category and move to previous
                                        try:
                                            current_index = self.sidebar_items.index(
                                                self.selected_category
                                            )
                                            if current_index > 0:
                                                self.selected_category = (
                                                    self.sidebar_items[
                                                        current_index - 1
                                                    ]
                                                )
                                                # Ensure the new category is visible
                                                if not self.is_category_visible(
                                                    current_index - 1
                                                ):
                                                    self.scroll_offset = max(
                                                        0,
                                                        (current_index - 1)
                                                        * self.item_height
                                                        - 200,
                                                    )
                                        except Exception as e:
                                            print(
                                                f"[WARNING] Failed to navigate to previous category: {e}"
                                            )
                                    elif event.key == pygame.K_DOWN:
                                        # Find current category and move to next
                                        try:
                                            current_index = self.sidebar_items.index(
                                                self.selected_category
                                            )
                                            if (
                                                current_index
                                                < len(self.sidebar_items) - 1
                                            ):
                                                self.selected_category = (
                                                    self.sidebar_items[
                                                        current_index + 1
                                                    ]
                                                )
                                                # Ensure the new category is visible
                                                if not self.is_category_visible(
                                                    current_index + 1
                                                ):
                                                    self.scroll_offset = min(
                                                        self.max_scroll,
                                                        (current_index + 1)
                                                        * self.item_height
                                                        - 200,
                                                    )
                                        except Exception as e:
                                            print(
                                                f"[WARNING] Failed to navigate to next category: {e}"
                                            )

                            except Exception as e:
                                print(
                                    f"[ERROR] Failed to handle keyboard scrolling: {e}"
                                )

                    except Exception as e:
                        print(f"[ERROR] Failed to handle event {event.type}: {e}")
                        continue

            except Exception as e:
                print(f"[ERROR] Failed to handle events: {e}")

            return True

        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to render settings menu: {e}")
            traceback.print_exc()
            # Try to show error message on screen
            try:
                self.screen.fill(BLACK)
                error_font = pygame.font.Font(None, 36)
                error_text = error_font.render(
                    "Settings Error - Check Console", True, RED
                )
                self.screen.blit(error_text, (50, 300))
                pygame.display.update()
            except:
                pass
            return False

    def get_settings(self):
        """Get all settings with error handling."""
        try:
            settings = {}
            for cat in self.settings_by_category.values():
                for setting in cat:
                    try:
                        settings[setting["key"]] = setting["value"]
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to get setting {setting.get('key', 'unknown')}: {e}"
                        )
                        continue
            return settings
        except Exception as e:
            print(f"[ERROR] Failed to get settings: {e}")
            return {}
