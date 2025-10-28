"""Common Imports"""

from SHARED.core_imports import *

"""File Specific Imports"""
import UTILITIES.utils_config as utils_config
from RENDER.Common import get_text_surface


class PauseMenuRenderer:
    """
    Renderer for the pause menu overlay.
    Displays menu options when the game is paused.
    """

    def __init__(self, screen):
        """
        Initialize the pause menu renderer.

        Args:
            screen: Pygame screen surface to render on
        """
        self.screen = screen

    def render(self):
        """
        Render the pause menu overlay with options.
        """
        if utils_config.HEADLESS_MODE:
            return  # Skip pause menu rendering in headless mode

        # Get screen dimensions
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Create and draw semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(180)  # Semi-transparent
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Pause menu text
        menu_title = get_text_surface("PAUSED", "Arial", 64, (255, 255, 255))
        menu_instructions = [
            "Press ESC to Resume",
            "Press M to Restart Episode",
            "Press Q to Quit",
        ]

        # Title
        title_rect = menu_title.get_rect(
            center=(screen_width // 2, screen_height // 2 - 100)
        )
        self.screen.blit(menu_title, title_rect)

        # Instructions
        font_size = 32
        y_start = screen_height // 2 - 50
        for i, instruction in enumerate(menu_instructions):
            text_surface = get_text_surface(
                instruction, "Arial", font_size, (200, 200, 200)
            )
            text_rect = text_surface.get_rect(
                center=(screen_width // 2, y_start + i * 60)
            )
            self.screen.blit(text_surface, text_rect)
