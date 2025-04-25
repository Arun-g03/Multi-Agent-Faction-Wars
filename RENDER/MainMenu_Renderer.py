"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from UTILITIES.utils_tensorboard import TensorBoardLogger
from RENDER.Common import MENU_FONT, WHITE, BLACK, BLUE, GREEN, RED, GREY, DARK_GREY, DARK_GREEN, get_font
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
        self.font = get_font(24, MENU_FONT)
        self.selected_mode = None
        print("MenuRenderer initialised")
        

    def draw_text(self, surface, text, font, size, colour, x, y, bold=False):
        font_obj = get_font(size, font, bold)
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

    def render_menu(self):
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
            SCREEN_WIDTH // 2,
            base_y - 100
        )
        
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
            center_x, base_y + (button_height + button_spacing) * 1, button_width, button_height
        )

        credits_button_rect = self.create_button(
            self.screen, "Credits", MENU_FONT, button_font_size, DARK_GREY, (
                160, 160, 160), (90, 90, 90),
            center_x, base_y + (button_height + button_spacing) * 2, button_width, button_height
        )

        # Check if log files exist
        log_dir = "RUNTIME_LOGS/Tensorboard_logs"
        if os.path.exists(log_dir):
            log_files_exist = any(os.path.isfile(os.path.join(log_dir, f)) for f in os.listdir(log_dir) if f.startswith("events.out"))
        else:
            log_files_exist = False        
        

        
        tensorboard_button_rect = self.create_button(
            self.screen, " Launch Tensorboard", MENU_FONT, button_font_size, BLUE, (0, 0, 150), (0, 0, 200),
            center_x, base_y + (button_height + button_spacing) * 3, button_width, button_height
        )
        
        exit_button_rect = self.create_button(
            self.screen, "Exit", MENU_FONT, button_font_size, RED, (
                150, 0, 0), (200, 0, 0),
            center_x, base_y + (button_height + button_spacing) * 4, button_width, button_height
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                
                self.cleanup(QUIT=True)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if start_button_rect.collidepoint(event.pos):
                    print("[INFO] Starting simulation setup...")
                    simulation_config = self.game_setup_menu()
                    if simulation_config:
                        print("[INFO] Starting game with config:", simulation_config)
                        self.pending_game_config = simulation_config
                        return False  # exit menu, let main loop handle game launch

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
                    credits = CreditsRenderer(self.screen)
                    credits.run()

                elif tensorboard_button_rect.collidepoint(event.pos):
                    log_dir = "RUNTIME_LOGS/Tensorboard_logs"
                    
                    # Check if the log directory exists
                    if os.path.exists(log_dir):
                        # Get all run directories
                        run_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
                        
                        # Check for log files in run directories
                        log_files_exist = any(
                            any(f.startswith("events.out") for f in os.listdir(os.path.join(log_dir, run_dir)))
                            for run_dir in run_dirs if os.path.exists(os.path.join(log_dir, run_dir))
                        )
                    else:
                        log_files_exist = False
                    
                    # Get the current TensorBoard logger
                    tensorboard_logger = TensorBoardLogger()
                    msg_duration = utils_config.FPS*3*4
                    
                    if tensorboard_logger is None:
                        # TensorBoard is disabled in config
                        self.show_message("TensorBoard is disabled in settings. Please enable it first.", 
                                        duration=msg_duration, bold=True)
                    else:
                        if log_files_exist:
                            # Get all available runs with log files
                            runs_with_logs = [
                                d for d in run_dirs if 
                                any(f.startswith("events.out") for f in os.listdir(os.path.join(log_dir, d)))
                            ]
                            
                            if runs_with_logs:
                                run_list = "\n    - " + "\n    - ".join(runs_with_logs)
                                self.show_message(f"Launching TensorBoard with runs:\n{run_list}", 
                                                duration=msg_duration, bold=True)
                            else:
                                self.show_message("Launching TensorBoard with available logs", 
                                                duration=msg_duration, bold=True)
                            
                            tensorboard_logger.run_tensorboard()
                        else:
                            self.show_message("No TensorBoard logs found yet. Launching TensorBoard anyway.", 
                                            duration=msg_duration, bold=True)
                            tensorboard_logger.run_tensorboard()


                        
                elif exit_button_rect.collidepoint(event.pos):
                    print("[INFO] Exiting game...")
                    
                    self.cleanup(QUIT=True)

        pygame.display.update()
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
        if not mode or mode == "back":
            return None  # Return None to indicate cancellation
        
        # Step 2: Choose whether to load existing models or start fresh
        if mode == "train":
            load_choice = self.render_load_choice()
            if load_choice is None:
                return self.game_setup_menu()  # Return to mode selection
                
            if load_choice is False:
                # Start fresh training
                # Create a new TensorBoard run for this fresh training session
                
                if utils_config.ENABLE_TENSORBOARD:
                    TensorBoardLogger.reset()  # Reset first
                    tensorboard_logger = TensorBoardLogger()
                    tensorboard_logger.run_tensorboard()
                    print(f"[INFO] Created new TensorBoard run for fresh training")
                
                return {"mode": "train", "load_existing": False}
            else:
                try:
                    # Create a new TensorBoard run for this continued training session
                    if utils_config.ENABLE_TENSORBOARD:
                        TensorBoardLogger.reset()  # Reset first
                        tensorboard_logger = TensorBoardLogger()
                        tensorboard_logger.run_tensorboard()
                        print(f"[INFO] Created new TensorBoard run for continued training")
                    
                    # Load existing models for training
                    return self.render_menu()
                except:
                    self.MenuRenderer()
                    
        elif mode == "evaluate":
            # For evaluation, we always need to load models
            models = self.select_models_by_role()
            if not models:
                self.show_message("Evaluation requires models to load")
                return self.game_setup_menu()  # Return to mode selection
                
            # Create a new TensorBoard run for this evaluation session
            tensorboard_logger = TensorBoardLogger()
            if tensorboard_logger:
                TensorBoardLogger.reset()
                print(f"[INFO] Created new TensorBoard run for evaluation")
            
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

        pygame.display.update()

        # Wait for a selection
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    self.cleanup(QUIT=True)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if train_button.collidepoint(event.pos):
                        print("[INFO] Training mode selected.")
                        return "train"
                    elif evaluate_button.collidepoint(event.pos):
                        print("[INFO] Evaluation mode selected.")
                        return "evaluate"
                    elif cancel_button.collidepoint(event.pos):
                        return self.render_menu()  # Return to main menu

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

        pygame.display.update()

        # Wait for a selection
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    self.cleanup(QUIT=True)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if load_button.collidepoint(event.pos):
                        selected_models = self.select_models_by_role()
                        if selected_models is not None:
                            return {"models": selected_models}
                        # If user cancels model selection, return to this screen
                    elif new_button.collidepoint(event.pos):
                        return False
                    elif cancel_button.collidepoint(event.pos):
                        return self.select_simulation_mode()  # Return to mode selection
    
    def select_models_by_role(self):
        """
        Guides the user through selecting models for each agent role.
        Returns a dictionary mapping roles to selected model paths.
        """
        base_dir = os.path.join(os.getcwd(), "NEURAL_NETWORK", "saved_models")
        agent_dir = os.path.join(base_dir, "Agents")
        hq_dir = os.path.join(base_dir, "HQ")

        selected_models = {}
        roles = {
            "gatherer": "gatherer",
            "peacekeeper": "peacekeeper"
        }

        # Handle gatherer and peacekeeper (Agents/)
        for role, keyword in roles.items():
            if os.path.exists(agent_dir):
                files = self.find_model_files_for_role(agent_dir, keyword)
                files.sort(key=lambda x: (self.extract_episode_number(x), self.extract_reward(x) or 0), reverse=True)
                selected = self.select_model_for_role(role.capitalize(), files, agent_dir)
                if selected is None:  # User clicked back
                    return self.render_load_choice()
                if selected != "SKIP":
                    selected_models[role] = selected
            else:
                self.show_message("Missing directory: saved_models/Agents")
                return self.render_load_choice()

        # Handle HQ
        if os.path.exists(hq_dir):
            hq_models = [f for f in os.listdir(hq_dir) if f.endswith('.pth')]
            hq_models.sort(key=lambda x: (self.extract_episode_number(x), self.extract_reward(x) or 0), reverse=True)
            selected = self.select_model_for_role("HQ", hq_models, hq_dir)
            if selected is None:  # User clicked back
                return self.render_load_choice()
            if selected != "SKIP":
                selected_models["HQ"] = selected
        else:
            self.show_message("Missing directory: saved_models/HQ")
            return self.render_load_choice()

        if not selected_models:
            self.show_message("No models were selected")
            return self.render_load_choice()

        return selected_models

    
    
    
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
    
    def find_model_files_for_role(self, directory, keyword):
        """
        Finds and returns a sorted list of model filenames from a directory based on a keyword.
        
        :param directory: Path to the model directory
        :param keyword: Keyword to filter models by (e.g., 'gatherer')
        :return: Sorted list of model filenames (most recent or highest reward first)
        """
        if not os.path.exists(directory):
            self.show_message(f"Missing directory: {directory}")
            return []

        # Filter by keyword and extension
        files = [f for f in os.listdir(directory) if keyword in f.lower() and f.endswith(".pth")]

        # Sort by (episode number, reward) descending
        files.sort(
            key=lambda x: (
                self.extract_episode_number(x),
                self.extract_reward(x) or 0
            ),
            reverse=True
        )
        return files


        
    def show_message(self, message, duration=1000, bold=False):
        """
        Shows a message on screen for a specified duration.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT
        
        
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))  # Semi-transparent black
        
        self.screen.blit(overlay, (0, 0))
        
        self.draw_text(self.screen, message, MENU_FONT, 24, WHITE,
                    SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, bold=bold)
        
        pygame.display.update()
        
        # Wait for specified duration
        pygame.time.wait(duration)


    def select_model_for_role(self, role, model_files, role_path):
        """
        Displays a list of models for a specific agent role and lets the user select one.
        Returns the selected model path, "SKIP" to skip this role, or None to cancel.
        """
        SCREEN_WIDTH = utils_config.SCREEN_WIDTH
        SCREEN_HEIGHT = utils_config.SCREEN_HEIGHT

        scroll_offset = 0
        max_visible_models = 8
        button_height = 40
        button_width = 500
        button_spacing = 10
        selected_model = None

        while True:
            self.screen.fill(BLACK)

            self.draw_text(self.screen, f"Select a Model for {role}", MENU_FONT, 28, WHITE,
                        SCREEN_WIDTH // 2, 80)

            if len(model_files) > max_visible_models:
                self.draw_text(self.screen, "Use mouse wheel to scroll", MENU_FONT, 18, GREY,
                            SCREEN_WIDTH // 2, 120)
            elif len(model_files) == 0:
                self.draw_text(self.screen, f"No saved models found for {role}.", MENU_FONT, 20, RED,
                            SCREEN_WIDTH // 2, 160)

            visible_models = model_files[scroll_offset:scroll_offset + max_visible_models]

            for i, model_name in enumerate(visible_models):
                y_pos = 160 + i * (button_height + button_spacing)
                is_selected = selected_model == os.path.join(role_path, model_name)

                display_name = self.prettify_model_name(model_name)

                model_button = self.create_button(
                    self.screen, display_name, MENU_FONT, 16,
                    GREEN if is_selected else BLUE,
                    (0, 220, 0) if is_selected else (0, 0, 220),
                    (0, 180, 0),
                    SCREEN_WIDTH // 2 - button_width // 2,
                    y_pos,
                    button_width,
                    button_height
                )

            # Skip and Back
            back_button = self.create_button(
                self.screen, "Back", MENU_FONT, 18, GREY, (120, 120, 120), (90, 90, 90),
                SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT - 120, 200, 40)

            skip_button = self.create_button(
                self.screen, f"Skip {role}", MENU_FONT, 18, RED, (220, 0, 0), (180, 0, 0),
                SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT - 120, 200, 40)
            # Continue
            continue_enabled = selected_model is not None and len(model_files) > 0
            if continue_enabled:
                continue_button = self.create_button(
                    self.screen, "Continue", MENU_FONT, 18,
                    GREEN if continue_enabled else DARK_GREY,
                    (0, 180, 0) if continue_enabled else GREY,
                    (0, 120, 0) if continue_enabled else GREY,
                    SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 60, 200, 40)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    
                    self.cleanup(QUIT=True)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if back_button.collidepoint(event.pos):
                            return None
                        if skip_button.collidepoint(event.pos):
                            return "SKIP"

                        for i, model_name in enumerate(visible_models):
                            y_pos = 160 + i * (button_height + button_spacing)
                            model_button = pygame.Rect(
                                SCREEN_WIDTH // 2 - button_width // 2,
                                y_pos,
                                button_width,
                                button_height
                            )
                            if model_button.collidepoint(event.pos):
                                selected_model = os.path.join(role_path, model_name)
                                print(f"[SELECTED] {role} model: {selected_model}")

                        if continue_enabled and continue_button.collidepoint(event.pos):
                            return selected_model if selected_model else "SKIP"

                    elif event.button == 4:  # scroll up
                        scroll_offset = max(0, scroll_offset - 1)
                    elif event.button == 5:  # scroll down
                        max_offset = max(0, len(model_files) - max_visible_models)
                        scroll_offset = min(max_offset, scroll_offset + 1)

    def prettify_model_name(self, filename):
        """
        Converts filenames like:
        Best_gatherer_episode_3_reward_29.00.pth
        Into:
        Gatherer  |  Episode 3  |  Reward 29.00
        """
        name = filename.replace(".pth", "")

        # Remove "Best" if present
        name = name.replace("Best_", "").replace("Best", "")

        # Extract parts
        parts = name.split("_")
        role = ""
        episode = ""
        reward = ""

        for i, part in enumerate(parts):
            if part.lower() in ["gatherer", "peacekeeper", "hq"]:
                role = part.capitalize()
            elif part == "episode" and i + 1 < len(parts):
                episode = f"Episode {parts[i + 1]}"
            elif part == "reward" and i + 1 < len(parts):
                try:
                    reward_val = float(parts[i + 1])
                    reward = f"Reward {reward_val:.2f}"
                except ValueError:
                    reward = f"Reward {parts[i + 1]}"

        # Fallback
        if not role:
            role = "Unknown"

        return f"{role:<10} | {episode:<10} | {reward:<10}"


    def cleanup(self, QUIT):
        if utils_config.ENABLE_TENSORBOARD:
            tensorboard_logger = TensorBoardLogger()
            tensorboard_logger.stop_tensorboard()  # Kill TensorBoard if running


        if QUIT:
            pygame.quit()
            sys.exit()  # Ensure the system fully exits when quitting the game
            print("[INFO] - MainMenu_Renderer ---- Game closed successfully.")