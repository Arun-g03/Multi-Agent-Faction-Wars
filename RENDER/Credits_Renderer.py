"""Common Imports"""
from SHARED.core_imports import *
"""File Specific Imports"""
import UTILITIES.utils_config as utils_config



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
