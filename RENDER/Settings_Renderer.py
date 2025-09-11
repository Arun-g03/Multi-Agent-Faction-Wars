"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from RENDER.Common import SETTINGS_FONT, WHITE, BLACK, BLUE, GREEN, RED, GREY, DARK_GREY, get_font
import UTILITIES.utils_config as utils_config
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
    """ Class for rendering the settings menu. """
    def __init__(self, screen):
        try:
            self.screen = screen
            self.font = get_font(24, SETTINGS_FONT, False)
            self.selected_category = "debugging"
            self.saved = False
            self.input_mode = False
            self.input_field = None
            self.input_text = ""
            self.cursor_visible = True
            self.cursor_timer = 0
            self.show_help = False
            self.help_timer = 0
            
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
                "debugging", "episode settings", "screen",
                "world", "resources", "agent", "faction",
                "ai training", "curriculum learning", "experience replay",
                "multi-agent", "training monitoring", "advanced loss"
            ]

            # Store config state and original defaults
            self.settings_by_category = {}
            self.defaults = {}

            raw_settings = {
                "debugging": [
                    ("TensorBoard", "ENABLE_TENSORBOARD"),
                    ("Logging", "ENABLE_LOGGING"),
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
                "experience replay": [
                    ("Enable Experience Replay", "ENABLE_EXPERIENCE_REPLAY"),
                    ("Replay Buffer Size", "REPLAY_BUFFER_SIZE", 1000),
                    ("Priority Replay Alpha", "PRIORITY_REPLAY_ALPHA", 0.01),
                    ("Priority Replay Beta", "PRIORITY_REPLAY_BETA", 0.01),
                ],
                "multi-agent": [
                    ("Enable Multi-Agent Training", "ENABLE_MULTI_AGENT_TRAINING"),
                    ("Inter-Agent Communication", "INTER_AGENT_COMMUNICATION"),
                    ("Faction Coordination Bonus", "FACTION_COORDINATION_BONUS", 0.01),
                ],
                "training monitoring": [
                    ("Enable Training Monitoring", "ENABLE_TRAINING_MONITORING"),
                    ("Save Checkpoint Every", "SAVE_CHECKPOINT_EVERY", 1),
                    ("Evaluation Frequency", "EVALUATION_FREQUENCY", 1),
                    ("Early Stopping Patience", "EARLY_STOPPING_PATIENCE", 1),
                ],
                "advanced loss": [
                    ("Use Huber Loss", "USE_HUBER_LOSS"),
                    ("Use Focal Loss", "USE_FOCAL_LOSS"),
                    ("Loss Normalization", "LOSS_NORMALIZATION"),
                ]
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
                            print(f"[WARNING] Config key '{key}' not found in utils_config, using default value")
                            # Use sensible defaults for missing config keys
                            if len(item) == 3:
                                setting = {"label": label, "key": key, "value": item[2], "step": item[2]}
                            else:
                                setting = {"label": label, "key": key, "value": False, "options": [True, False]}
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
            # Screen height available for categories (from y=80 to y=500)
            available_height = 500 - 80
            # Height needed for all categories
            total_height = len(self.sidebar_items) * 60
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
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + self.scroll_speed)
        except Exception as e:
            print(f"[ERROR] Failed to handle scroll input: {e}")
    
    def is_category_visible(self, index):
        """Check if a category at the given index is currently visible on screen."""
        try:
            category_y = 80 + index * 60 - self.scroll_offset
            return 80 <= category_y <= 500
        except Exception as e:
            print(f"[ERROR] Failed to check category visibility: {e}")
            return True
    
    def draw_scroll_bar(self):
        """Draw the scroll bar on the right side of the sidebar."""
        try:
            if self.max_scroll <= 0:
                return  # No scrolling needed
            
            # Calculate scroll bar dimensions
            sidebar_width = 210
            sidebar_height = 500 - 80
            scroll_bar_height = max(30, (sidebar_height / (self.max_scroll + sidebar_height)) * sidebar_height)
            
            # Calculate scroll bar position
            scroll_ratio = self.scroll_offset / self.max_scroll if self.max_scroll > 0 else 0
            scroll_bar_y = 80 + scroll_ratio * (sidebar_height - scroll_bar_height)
            
            # Draw scroll bar background
            scroll_bg_rect = pygame.Rect(20 + sidebar_width - self.scroll_bar_width, 80, 
                                       self.scroll_bar_width, sidebar_height)
            pygame.draw.rect(self.screen, DARK_GREY, scroll_bg_rect)
            
            # Draw scroll bar handle
            self.scroll_bar_rect = pygame.Rect(20 + sidebar_width - self.scroll_bar_width, scroll_bar_y,
                                             self.scroll_bar_width, scroll_bar_height)
            pygame.draw.rect(self.screen, GREY, self.scroll_bar_rect)
            pygame.draw.rect(self.screen, WHITE, self.scroll_bar_rect, 2)
            
            # Draw scroll indicators
            if self.scroll_offset > 0:
                # Up arrow
                up_arrow_rect = pygame.Rect(20 + sidebar_width - self.scroll_bar_width, 75, 
                                          self.scroll_bar_width, 10)
                pygame.draw.rect(self.screen, WHITE, up_arrow_rect)
                self.draw_text("▲", 12, BLACK, 20 + sidebar_width - self.scroll_bar_width + 5, 75)
            
            if self.scroll_offset < self.max_scroll:
                # Down arrow
                down_arrow_rect = pygame.Rect(20 + sidebar_width - self.scroll_bar_width, 505, 
                                            self.scroll_bar_width, 10)
                pygame.draw.rect(self.screen, WHITE, down_arrow_rect)
                self.draw_text("▼", 12, BLACK, 20 + sidebar_width - self.scroll_bar_width + 5, 505)
                
        except Exception as e:
            print(f"[ERROR] Failed to draw scroll bar: {e}")
    
    def handle_scroll_bar_click(self, mouse_pos):
        """Handle clicks on the scroll bar."""
        try:
            if not self.scroll_bar_rect or self.max_scroll <= 0:
                return False
            
            # Check if click is on scroll bar background
            sidebar_width = 210
            scroll_bg_rect = pygame.Rect(20 + sidebar_width - self.scroll_bar_width, 80, 
                                       self.scroll_bar_width, 500 - 80)
            
            if scroll_bg_rect.collidepoint(mouse_pos):
                # Calculate new scroll position based on click
                click_y = mouse_pos[1] - 80
                sidebar_height = 500 - 80
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
                "REPLAY_BUFFER_SIZE": 500,
                "PRIORITY_REPLAY_ALPHA": 0.01,
                "PRIORITY_REPLAY_BETA": 0.01,
                "FACTION_COORDINATION_BONUS": 0.01,
                "SAVE_CHECKPOINT_EVERY": 1,
                "EVALUATION_FREQUENCY": 1,
                "EARLY_STOPPING_PATIENCE": 1
            }
            
            # Update step sizes for existing settings
            for category in self.settings_by_category.values():
                for setting in category:
                    try:
                        if setting["key"] in custom_steps:
                            setting["step"] = custom_steps[setting["key"]]
                    except Exception as e:
                        print(f"[WARNING] Failed to set step size for {setting.get('key', 'unknown')}: {e}")
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
                "ENABLE_EXPERIENCE_REPLAY": "Use experience replay for better training",
                "REPLAY_BUFFER_SIZE": "Size of experience replay buffer (10000+ recommended)",
                "PRIORITY_REPLAY_ALPHA": "Priority replay importance (0.6 recommended)",
                "PRIORITY_REPLAY_BETA": "Priority replay correction (0.4 recommended)",
                "ENABLE_MULTI_AGENT_TRAINING": "Enable coordinated multi-agent learning",
                "INTER_AGENT_COMMUNICATION": "Allow agents to share experiences",
                "FACTION_COORDINATION_BONUS": "Reward for coordinated actions (0.1 recommended)",
                "ENABLE_TRAINING_MONITORING": "Track training progress and metrics",
                "SAVE_CHECKPOINT_EVERY": "Save models every N episodes (5 recommended)",
                "EVALUATION_FREQUENCY": "Evaluate performance every N episodes (3 recommended)",
                "EARLY_STOPPING_PATIENCE": "Episodes without improvement before stopping (10 recommended)",
                "USE_HUBER_LOSS": "Use Huber loss for value function (more robust)",
                "USE_FOCAL_LOSS": "Use focal loss for policy (experimental)",
                "LOSS_NORMALIZATION": "Normalize losses for training stability"
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
                    y = 80 + i * 60
                    # Check if mouse is hovering over the parameter label
                    label_rect = pygame.Rect(250, y, 350, 30)
                    if label_rect.collidepoint(mouse_pos):
                        description = self.get_parameter_description(setting["key"])
                        if description:
                            # Show tooltip
                            self.draw_tooltip(description, mouse_pos[0], mouse_pos[1] - 40)
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
        """Draw a comprehensive help overlay for AI training parameters."""
        try:
            # Semi-transparent background
            overlay_surface = pygame.Surface((800, 600))
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
                "• Decay: 0.95-0.99 recommended for gradual reduction",
                "",
                "Training Parameters:",
                "• Batch Size: 32-128 for optimal training",
                "• Memory Size: At least 128 experiences before training",
                "• GAE Lambda: 0.95-0.99 for advantage estimation",
                "",
                "Loss Coefficients:",
                "• Value Loss: 0.5 for balanced policy/value learning",
                "• Entropy: 0.01-0.1 to encourage exploration",
                "• Gradient Clip: 0.5 for training stability",
                "",
                "Curriculum Learning:",
                "• Start with 30% resources, gradually increase to 100%",
                "• Helps agents learn progressively",
                "",
                "Experience Replay:",
                "• Buffer size: 10,000+ for better sample diversity",
                "• Priority replay for important experiences",
                "",
                "Press ESC or click Help again to close"
            ]
            
            # Draw help content
            y_offset = 50
            for line in help_content:
                try:
                    if line.startswith("•"):
                        color = GREEN
                        size = 16
                    elif line in ["AI Training Parameters Help", "Learning Rates:", "Training Parameters:", "Loss Coefficients:", "Curriculum Learning:", "Experience Replay:"]:
                        color = WHITE
                        size = 20
                        if line == "AI Training Parameters Help":
                            size = 24
                    else:
                        color = GREY
                        size = 16
                    
                    self.draw_text(line, size, color, 50, y_offset)
                    y_offset += 25
                except Exception as e:
                    print(f"[WARNING] Failed to draw help line: {e}")
                    y_offset += 25
                    continue
            
            # Close button
            try:
                close_btn = pygame.Rect(700, 20, 80, 30)
                pygame.draw.rect(self.screen, RED, close_btn)
                self.draw_text("Close", 16, WHITE, 720, 25)
                
                # Handle close button click
                mouse_pos = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed()[0] and close_btn.collidepoint(mouse_pos):
                    self.show_help = False
            except Exception as e:
                print(f"[ERROR] Failed to draw close button: {e}")
                
        except Exception as e:
            print(f"[ERROR] Failed to draw help overlay: {e}")
            traceback.print_exc()

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
                print(f"[CRITICAL] Failed to draw text even with fallback font: {fallback_e}")
    
    

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
        try:
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
            self.draw_text("SETTINGS", 36, WHITE, 20, 20, bold=True)  # Larger font size for the title
            
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
                        y = 80 + idx * 60 - self.scroll_offset
                        
                        # Skip if category is not visible
                        if not self.is_category_visible(idx):
                            continue
                        
                        selected = (label == self.selected_category)
                        color = GREY if selected else DARK_GREY
                        btn = self.create_button(
                            label.upper(), SETTINGS_FONT, 20, color, (120, 120, 120), (180, 180, 180), 20, y, 210, 40)
                        if pygame.mouse.get_pressed()[0] and btn.collidepoint(
                                pygame.mouse.get_pos()):
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
                            try:
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
                            except Exception as e:
                                print(f"[WARNING] Failed to draw toggle button for {label}: {e}")

                        elif "step" in setting:
                            try:
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
                            except Exception as e:
                                print(f"[WARNING] Failed to draw step buttons for {label}: {e}")
                    except Exception as e:
                        print(f"[WARNING] Failed to draw setting {i}: {e}")
                        continue
            except Exception as e:
                print(f"[ERROR] Failed to draw settings: {e}")
                settings = []
                step_buttons = []

            # Draw main buttons
            try:
                back_btn = self.create_button(
                    "Back", SETTINGS_FONT, 20, GREY, (180, 180, 180), (120, 120, 120), 250, 500, 150, 50)
                save_return_btn = self.create_button(
                    "Save and Return", SETTINGS_FONT, 20, BLUE, (80, 80, 255), (50, 50, 200),
                    450, 500, 250, 50)
                reset_all_btn = self.create_button(
                    "Reset All", SETTINGS_FONT, 20, GREY, (180, 180, 180), (120, 120, 120),
                    20, 500, 200, 50)
                
                # Add help button for AI training categories
                help_btn = None
                if self.selected_category in ["ai training", "curriculum learning", "experience replay", "multi-agent", "training monitoring", "advanced loss"]:
                    try:
                        help_btn = self.create_button(
                            "Help", SETTINGS_FONT, 18, GREEN, (100, 200, 100), (80, 160, 80),
                            720, 500, 80, 50)
                    except Exception as e:
                        print(f"[WARNING] Failed to create help button: {e}")
            except Exception as e:
                print(f"[ERROR] Failed to draw main buttons: {e}")
                # Create minimal fallback buttons
                back_btn = pygame.Rect(250, 500, 150, 50)
                save_return_btn = pygame.Rect(450, 500, 250, 50)
                reset_all_btn = pygame.Rect(20, 500, 200, 50)
                help_btn = None
            
            # Show tooltips for AI training parameters
            try:
                if self.selected_category in ["ai training", "curriculum learning", "experience replay", "multi-agent", "training monitoring", "advanced loss"]:
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
            
            # Draw tips and help text
            try:
                self.draw_text("Tip: You can click on a value to input in a custom number.", 
                               24, 
                               GREY, 
                               50, 
                               560)
                self.draw_text("Type and press Enter to confirm", 
                               24, 
                               GREY, 
                               50, 
                               590)
                
                # Add scrolling instructions
                if self.max_scroll > 0:
                    self.draw_text("Scroll: Mouse wheel, Page Up/Down, or click scroll bar", 
                                   18, 
                                   BLUE, 
                                   50, 
                                   620)
                    self.draw_text("Navigation: Arrow keys to move between categories", 
                                   18, 
                                   BLUE, 
                                   50, 
                                   640)
                    self.draw_text("Quick jump: Home (top) / End (bottom)", 
                                   18, 
                                   BLUE, 
                                   50, 
                                   660)
                else:
                    # Add helpful tooltips for AI training parameters
                    if self.selected_category in ["ai training", "curriculum learning", "experience replay", "multi-agent", "training monitoring", "advanced loss"]:
                        self.draw_text("AI Training Tip: Lower learning rates (0.0001) for stability, higher (0.001) for faster learning", 
                                       18, 
                                       BLUE, 
                                       50, 
                                       620)
                        self.draw_text("Curriculum Tip: Start with fewer resources and gradually increase difficulty", 
                                       18, 
                                       BLUE, 
                                       50, 
                                       640)
                        self.draw_text("Monitoring Tip: Save checkpoints frequently to preserve good models", 
                                       18, 
                                       BLUE, 
                                       50, 
                                       660)
            except Exception as e:
                print(f"[WARNING] Failed to draw tips and help text: {e}")

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
                                        y = 80 + i * 60
                                        # matches where you draw the values
                                        value_rect = pygame.Rect(600, y, 80, 30)
                                        if value_rect.collidepoint(
                                                mouse_x, mouse_y) and "step" in setting:
                                            self.input_mode = True
                                            self.input_field = setting
                                            self.input_text = str(setting["value"])
                                            clicked_input = True
                                    except Exception as e:
                                        print(f"[WARNING] Failed to handle setting click {i}: {e}")
                                        continue

                                if back_btn.collidepoint(event.pos):
                                    return False
                                if save_return_btn.collidepoint(event.pos):
                                    # Apply any pending input before saving
                                    try:
                                        if self.input_mode and self.input_field:
                                            try:
                                                value = float(self.input_text) if '.' in self.input_text else int(self.input_text)
                                                self.input_field["value"] = value
                                            except ValueError:
                                                print(f"Invalid input for {self.input_field['key']}. Invalid input: {self.input_text}")
                                                pass  # Invalid input, keep the old value
                                    except Exception as e:
                                        print(f"[ERROR] Failed to process input before saving: {e}")
                                    
                                    self.saved = True
                                    return False
                                if reset_all_btn.collidepoint(event.pos):
                                    try:
                                        for settings in self.settings_by_category.values():
                                            for setting in settings:
                                                try:
                                                    setting["value"] = self.defaults.get(
                                                        setting["key"], setting["value"])
                                                except Exception as e:
                                                    print(f"[WARNING] Failed to reset setting {setting.get('key', 'unknown')}: {e}")
                                                    continue
                                    except Exception as e:
                                        print(f"[ERROR] Failed to reset all settings: {e}")
                                
                                # Handle help button click
                                if self.selected_category in ["ai training", "curriculum learning", "experience replay", "multi-agent", "training monitoring", "advanced loss"]:
                                    try:
                                        help_btn_rect = pygame.Rect(720, 500, 80, 50)
                                        if help_btn_rect.collidepoint(event.pos):
                                            self.show_help = not self.show_help
                                            self.help_timer = 0
                                    except Exception as e:
                                        print(f"[WARNING] Failed to handle help button click: {e}")
                                
                                for action, rect, setting in step_buttons:
                                    try:
                                        if rect.collidepoint(event.pos):
                                            if action == "toggle":
                                                try:
                                                    options = setting["options"]
                                                    current_index = options.index(setting["value"])
                                                    setting["value"] = options[(
                                                        current_index + 1) % len(options)]
                                                except Exception as e:
                                                    print(f"[WARNING] Failed to toggle setting: {e}")
                                            elif action == "minus":
                                                try:
                                                    setting["value"] = round(
                                                        setting["value"] - setting["step"], 3)
                                                except Exception as e:
                                                    print(f"[WARNING] Failed to decrease setting value: {e}")
                                            elif action == "plus":
                                                try:
                                                    setting["value"] = round(
                                                        setting["value"] + setting["step"], 3)
                                                except Exception as e:
                                                    print(f"[WARNING] Failed to increase setting value: {e}")
                                            elif action == "reset":
                                                try:
                                                    default_val = self.defaults.get(
                                                        setting["key"], setting["value"])
                                                    setting["value"] = default_val
                                                except Exception as e:
                                                    print(f"[WARNING] Failed to reset setting value: {e}")
                                    except Exception as e:
                                        print(f"[WARNING] Failed to handle step button action {action}: {e}")
                                        continue
                                        
                                clicked_input = False
                                for i, setting in enumerate(settings):
                                    try:
                                        y = 80 + i * 60
                                        value_rect = pygame.Rect(600, y, 80, 30)
                                        if value_rect.collidepoint(
                                                mouse_x, mouse_y) and "step" in setting:
                                            self.input_mode = True
                                            self.input_field = setting
                                            self.input_text = str(setting["value"])
                                            clicked_input = True
                                    except Exception as e:
                                        print(f"[WARNING] Failed to handle setting input {i}: {e}")
                                        continue

                                if not clicked_input:
                                    self.input_mode = False
                                    self.input_field = None
                            except Exception as e:
                                print(f"[ERROR] Failed to handle mouse button down: {e}")
                        
                        # Handle mouse wheel scrolling
                        if event.type == pygame.MOUSEWHEEL:
                            try:
                                self.handle_scroll_input(event.y)
                            except Exception as e:
                                print(f"[ERROR] Failed to handle mouse wheel: {e}")

                        if event.type == pygame.KEYDOWN and self.input_mode:
                            try:
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
                            except Exception as e:
                                print(f"[ERROR] Failed to handle keyboard input: {e}")
                                # Reset input mode on error
                                self.input_mode = False
                                self.input_field = None
                                self.input_text = ""
                        
                        # Handle ESC key to close help overlay
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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
                                if event.key == pygame.K_PAGEUP:
                                    self.handle_scroll_input(1)  # Scroll up
                                elif event.key == pygame.K_PAGEDOWN:
                                    self.handle_scroll_input(-1)  # Scroll down
                                elif event.key == pygame.K_HOME:
                                    self.scroll_offset = 0  # Jump to top
                                elif event.key == pygame.K_END:
                                    self.scroll_offset = self.max_scroll  # Jump to bottom
                                elif event.key == pygame.K_UP:
                                    # Find current category and move to previous
                                    try:
                                        current_index = self.sidebar_items.index(self.selected_category)
                                        if current_index > 0:
                                            self.selected_category = self.sidebar_items[current_index - 1]
                                            # Ensure the new category is visible
                                            if not self.is_category_visible(current_index - 1):
                                                self.scroll_offset = max(0, (current_index - 1) * 60 - 200)
                                    except Exception as e:
                                        print(f"[WARNING] Failed to navigate to previous category: {e}")
                                elif event.key == pygame.K_DOWN:
                                    # Find current category and move to next
                                    try:
                                        current_index = self.sidebar_items.index(self.selected_category)
                                        if current_index < len(self.sidebar_items) - 1:
                                            self.selected_category = self.sidebar_items[current_index + 1]
                                            # Ensure the new category is visible
                                            if not self.is_category_visible(current_index + 1):
                                                self.scroll_offset = min(self.max_scroll, (current_index + 1) * 60 - 200)
                                    except Exception as e:
                                        print(f"[WARNING] Failed to navigate to next category: {e}")
                            except Exception as e:
                                print(f"[ERROR] Failed to handle keyboard scrolling: {e}")
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
                error_text = error_font.render("Settings Error - Check Console", True, RED)
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
                        print(f"[WARNING] Failed to get setting {setting.get('key', 'unknown')}: {e}")
                        continue
            return settings
        except Exception as e:
            print(f"[ERROR] Failed to get settings: {e}")
            return {}

    