"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from RENDER.Common import FONT_NAME, WHITE, BLACK, BLUE, GREEN, RED, GREY, DARK_GREY
from RENDER.Settings_Renderer import SettingsMenuRenderer


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
            (0, 200, 0),
            (0, 100, 0),
            SCREEN_WIDTH // 2 - half_width - 150,
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

