"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from RENDER.Common import MENU_FONT, WHITE, BLACK, BLUE, GREEN, RED, GREY, DARK_GREY, DARK_GREEN
from RENDER.Settings_Renderer import SettingsMenuRenderer
from RENDER.Credits_Renderer import CreditsRenderer


#    __  __    _    ___ _   _   __  __ _____ _   _ _   _
#   |  \/  |  / \  |_ _| \ | | |  \/  | ____| \ | | | | |
#   | |\/| | / _ \  | ||  \| | | |\/| |  _| |  \| | | | |
#   | |  | |/ ___ \ | || |\  | | |  | | |___| |\  | |_| |
#   |_|  |_/_/   \_\___|_| \_| |_|  |_|_____|_| \_|\___/
#




class MenuRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(MENU_FONT, 24)
        self.selected_mode = None
        print("MenuRenderer initialised")

    def draw_text(self, surface, text, font, size, colour, x, y):
        font_obj = pygame.font.SysFont(font, size)
        text_surface = font_obj.render(text, True, colour)
        text_rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, text_rect)

    def create_button(
            self,#Default
            surface,#Required\/
            text,
            font,
            size,
            colour,          # ← default (idle) fill
            hover_colour,    # ← fill while mouse‑over
            click_colour,    # ← fill while “clicked”
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
            MENU_FONT,
            28,
            WHITE,
            SCREEN_WIDTH //
            2,
            base_y -
            100)
        self.draw_text(
            self.screen,
            "Created as part of my BSC Computer Science final year project",
            MENU_FONT,
            20,
            WHITE,
            SCREEN_WIDTH // 2,
            base_y - 70)
        self.draw_text(self.screen, "Main Menu", MENU_FONT,
                    28, WHITE, SCREEN_WIDTH // 2, base_y - 40)

        # Start Simulation button
        start_button_rect = self.create_button(
            self.screen,
            "Start Simulation",
            MENU_FONT,
            button_font_size,
            DARK_GREEN,
            (0, 200, 0),
            (0, 100, 0),
            center_x,
            base_y,
            button_width,
            button_height
        )

        settings_button_rect = self.create_button(
            self.screen, "Settings", MENU_FONT, button_font_size, GREY, (
                180, 180, 180), (100, 100, 100),
            center_x, base_y + (button_height + button_spacing) *
            1, button_width, button_height
        )

        credits_button_rect = self.create_button(
            self.screen, "Credits", MENU_FONT, button_font_size, DARK_GREY, (
                160, 160, 160), (90, 90, 90),
            center_x, base_y + (button_height + button_spacing) *
            2, button_width, button_height
        )

        exit_button_rect = self.create_button(
            self.screen, "Exit", MENU_FONT, button_font_size, RED, (
                150, 0, 0), (200, 0, 0),
            center_x, base_y + (button_height + button_spacing) *
            3, button_width, button_height
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if start_button_rect.collidepoint(event.pos):
                    print("[INFO] Starting simulation setup...")
                    simulation_config = self.game_setup_menu()
                    if simulation_config:
                        print("[INFO] Starting game with config:", simulation_config)
                        start_game_callback(
                            simulation_config,
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

    def game_setup_menu(self):
        """
        Handles the game setup process including mode selection (training/evaluation)
        and model loading options.
        
        Returns a configuration dictionary or None if cancelled.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT
        
        # Step 1: Choose mode (Training or Evaluation)
        mode = self.select_simulation_mode()
        if not mode:
            return None  # User cancelled
        
        # Step 2: Choose whether to load existing models or start fresh
        if mode == "train":
            load_choice = self.render_load_choice()
            if load_choice is None:
                return None  # User cancelled
                
            if load_choice is False:
                # Start fresh training
                return {"mode": "train", "load_existing": False}
            else:
                # Load existing models for training
                return {"mode": "train", "load_existing": True, "models": load_choice["models"]}
                
        elif mode == "evaluate":
            # For evaluation, we always need to load models
            models = self.select_models_by_role()
            if not models:
                self.show_message("Evaluation requires models to load")
                return None
                
            return {"mode": "evaluate", "load_existing": True, "models": models}
        
        return None

    def select_simulation_mode(self):
        """
        Presents a screen for selecting between Training and Evaluation modes.
        Returns "train", "evaluate", or None if cancelled.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT
        self.screen.fill(BLACK)

        self.draw_text(self.screen, "Select Simulation Mode", MENU_FONT, 28, WHITE,
                    SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80)

        button_width = 200
        button_height = 50
        spacing = 30

        train_button = self.create_button(
            self.screen, "Training", MENU_FONT, 22, DARK_GREEN, (0, 200, 0), (0, 100, 0),
            SCREEN_WIDTH // 2 - button_width - spacing, SCREEN_HEIGHT // 2, button_width, button_height)

        evaluate_button = self.create_button(
            self.screen, "Evaluation", MENU_FONT, 22, BLUE, (0, 0, 255), (0, 0, 200),
            SCREEN_WIDTH // 2 + spacing, SCREEN_HEIGHT // 2, button_width, button_height)

        cancel_button = self.create_button(
            self.screen, "Back", MENU_FONT, 20, DARK_GREY, (120, 120, 120), (80, 80, 80),
            SCREEN_WIDTH // 2 - button_width // 2, SCREEN_HEIGHT // 2 + 100, button_width, 40)

        pygame.display.flip()

        # Wait for a selection
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if train_button.collidepoint(event.pos):
                        print("[INFO] Training mode selected.")
                        return "train"
                    elif evaluate_button.collidepoint(event.pos):
                        print("[INFO] Evaluation mode selected.")
                        return "evaluate"
                    elif cancel_button.collidepoint(event.pos):
                        return None

    def render_load_choice(self):
        """
        Presents a screen for selecting between training from scratch or loading saved models.
        Returns False to start fresh, a dictionary with models if loading, or None if cancelled.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT
        self.screen.fill(BLACK)

        self.draw_text(self.screen, "Start From Scratch or Load Existing Models?", MENU_FONT, 28, WHITE,
                    SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 80)

        button_width = 200
        button_height = 50
        spacing = 30

        load_button = self.create_button(
            self.screen, "Load Models", MENU_FONT, 22, DARK_GREEN, GREEN, GREY,
            SCREEN_WIDTH // 2 - button_width - spacing, SCREEN_HEIGHT // 2, button_width, button_height)

        new_button = self.create_button(
            self.screen, "Train Fresh", MENU_FONT, 22, BLUE, (0, 180, 0), GREY,
            SCREEN_WIDTH // 2 + spacing, SCREEN_HEIGHT // 2, button_width, button_height)

        cancel_button = self.create_button(
            self.screen, "Back", MENU_FONT, 20, DARK_GREY, (120, 120, 120), (80, 80, 80),
            SCREEN_WIDTH // 2 - button_width // 2, SCREEN_HEIGHT // 2 + 100, button_width, 40)

        pygame.display.flip()

        # Wait for a selection
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if load_button.collidepoint(event.pos):
                        selected_models = self.select_models_by_role()
                        if selected_models is not None:
                            return {"models": selected_models}
                        # If user cancels model selection, return to this screen
                    elif new_button.collidepoint(event.pos):
                        return False
                    elif cancel_button.collidepoint(event.pos):
                        return None
    
    
    
    def select_models_by_role(self):
        """
        Guides the user through selecting models for each agent role.
        Returns a dictionary mapping roles to selected model paths.
        """
        # Base directory for saved models
        base_dir = os.path.join(os.getcwd(), "NEURAL_NETWORK", "saved_models")
        
        # Check if directory exists
        if not os.path.exists(base_dir):
            self.show_message("No saved models directory found")
            pygame.display.flip()
            return None
        
        # Dictionary to store selected models for each role
        selected_models = {}
        
        # Handle gatherer models
        gatherer_dir = os.path.join(base_dir, "Gatherer")
        if os.path.exists(gatherer_dir):
            try:
                gatherer_models = [f for f in os.listdir(gatherer_dir) if f.endswith('.pth')]
                if gatherer_models:
                    gatherer_models.sort(key=lambda x: (self.extract_episode_number(x), self.extract_reward(x) or 0), reverse=True)
                    selected_gatherer = self.select_model_for_role("Gatherer", gatherer_models, gatherer_dir)
                    if selected_gatherer and selected_gatherer != "SKIP":
                        selected_models["gatherer"] = selected_gatherer
            except Exception as e:
                self.show_message(f"Error accessing gatherer models: {str(e)}")
                pygame.display.flip()

        # Handle peacekeeper models
        peacekeeper_dir = os.path.join(base_dir, "Peacekeeper")
        if os.path.exists(peacekeeper_dir):
            try:
                peacekeeper_models = [f for f in os.listdir(peacekeeper_dir) if f.endswith('.pth')]
                if peacekeeper_models:
                    peacekeeper_models.sort(key=lambda x: (self.extract_episode_number(x), self.extract_reward(x) or 0), reverse=True)
                    selected_peacekeeper = self.select_model_for_role("Peacekeeper", peacekeeper_models, peacekeeper_dir)
                    if selected_peacekeeper and selected_peacekeeper != "SKIP":
                        selected_models["peacekeeper"] = selected_peacekeeper
            except Exception as e:
                self.show_message(f"Error accessing peacekeeper models: {str(e)}")
                pygame.display.flip()
        
        # Handle HQ models
        hq_dir = os.path.join(base_dir, "HQ")
        if os.path.exists(hq_dir):
            try:
                hq_models = [f for f in os.listdir(hq_dir) if f.endswith('.pth')]
                if hq_models:
                    hq_models.sort(key=lambda x: (self.extract_episode_number(x), self.extract_reward(x) or 0), reverse=True)
                    selected_hq = self.select_model_for_role("HQ", hq_models, hq_dir)
                    if selected_hq and selected_hq != "SKIP":
                        selected_models["HQ"] = selected_hq
            except Exception as e:
                self.show_message(f"Error accessing HQ models: {str(e)}")
                pygame.display.flip()
        
        if not selected_models:
            self.show_message("No models were selected")
            pygame.display.flip()
            return None
            
        return selected_models

    def select_model_for_role(self, role, model_files, role_path):
        """
        Displays a list of models for a specific agent role and lets the user select one.
        Returns the selected model path, "SKIP" to skip this role, or None to cancel.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT
        
        # Variables for scrolling
        scroll_offset = 0
        max_visible_models = 8
        button_height = 40
        button_width = 500
        button_spacing = 10
        
        while True:
            self.screen.fill(BLACK)
            
            # Draw title
            self.draw_text(self.screen, f"Select a Model for {role}", MENU_FONT, 28, WHITE,
                        SCREEN_WIDTH // 2, 80)
            
            # Draw scrolling instructions if needed
            if len(model_files) > max_visible_models:
                self.draw_text(self.screen, "Use mouse wheel to scroll", MENU_FONT, 18, GREY,
                            SCREEN_WIDTH // 2, 120)
            
            # Draw model buttons
            visible_models = model_files[scroll_offset:scroll_offset + max_visible_models]
            
            for i, model_name in enumerate(visible_models):
                y_pos = 160 + i * (button_height + button_spacing)
                
                # Extract episode and reward info if available
                episode_num = self.extract_episode_number(model_name)
                reward = self.extract_reward(model_name)
                
                # Create button with model name
                model_button = self.create_button(
                    self.screen, model_name, MENU_FONT, 16, BLUE, (0, 0, 220), (0, 0, 180),
                    SCREEN_WIDTH // 2 - button_width // 2, y_pos, button_width, button_height)
                    
            # Skip button
            skip_button = self.create_button(
                self.screen, f"Skip {role}", MENU_FONT, 18, RED, (0, 180, 0), (0, 120, 0),
                SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT - 120, 200, 40)
                
            # Back button
            back_button = self.create_button(
                self.screen, "Back", MENU_FONT, 18, GREY, (220, 0, 0), (180, 0, 0),
                SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 120, 200, 40)
                
            pygame.display.flip()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if back_button.collidepoint(event.pos):
                            return None
                            
                        if skip_button.collidepoint(event.pos):
                            return "SKIP"
                            
                        # Check if a model was selected
                        for i, model_name in enumerate(visible_models):
                            y_pos = 160 + i * (button_height + button_spacing)
                            model_button = pygame.Rect(
                                SCREEN_WIDTH // 2 - button_width // 2, 
                                y_pos, 
                                button_width, 
                                button_height
                            )
                            
                            if model_button.collidepoint(event.pos):
                                return os.path.join(role_path, model_name)
                                
                    elif event.button == 4:  # Scroll up
                        scroll_offset = max(0, scroll_offset - 1)
                    elif event.button == 5:  # Scroll down
                        max_offset = max(0, len(model_files) - max_visible_models)
                        scroll_offset = min(max_offset, scroll_offset + 1)

    def extract_episode_number(self, filename):
        """Extract episode number from filename if present."""
        try:
            match = re.search(r'episode_(\d+)', filename)
            if match:
                return int(match.group(1))
        except:
            pass
        return 0  # Default if not found

    def extract_reward(self, filename):
        """Extract reward value from filename if present."""
        try:
            match = re.search(r'reward_([\d\.]+)', filename)
            if match:
                # Remove trailing dots before converting to float
                reward_str = match.group(1).rstrip('.')
                return float(reward_str)
        except:
            pass
        return None  # Default if not found
    def show_message(self, message, duration=2000):
        """
        Shows a message on screen for a specified duration.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT
        
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Semi-transparent black
        
        self.screen.blit(overlay, (0, 0))
        
        self.draw_text(self.screen, message, MENU_FONT, 24, WHITE,
                    SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        
        pygame.display.flip()
        
        # Wait for specified duration
        pygame.time.wait(duration)