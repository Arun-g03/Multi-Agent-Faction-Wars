import pygame

# Constants for the menu screen
MENU_FONT = "Segoe UI"  # Clean, modern UI font for menus
GAME_FONT = "Roboto"  # Pixel-style retro font, great for in-game HUDs
CREDITS_FONT = "Georgia"  # Serif font for a formal, readable credit roll
SETTINGS_FONT = "Verdana"  # Clean sans-serif, easy to read in settings screens


# Define some reusable colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
GREEN = (50, 255, 50)
RED = (255, 50, 50)
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
DARK_GREEN = (0, 100, 0)

FONT_CACHE = {}


def get_font(size, font_name=GAME_FONT, bold=False):
    """
    Retrieves a font from cache or creates a new one if not cached.
    args:
        size: int
        font_name: str
    """
    cache_key = (font_name, size, bold)
    if cache_key not in FONT_CACHE:
        FONT_CACHE[cache_key] = pygame.font.SysFont(font_name, size, bold=bold)
    return FONT_CACHE[cache_key]


TEXT_SURFACE_CACHE = {}


def get_text_surface(text, font_name, size, color):
    cache_key = (text, font_name, size, color)
    if cache_key not in TEXT_SURFACE_CACHE:
        font_obj = get_font(size, font_name)
        TEXT_SURFACE_CACHE[cache_key] = font_obj.render(text, True, color)
    return TEXT_SURFACE_CACHE[cache_key]
