"""Common Imports"""
from SHARED.core_imports import *
"""File Specific Imports"""
import UTILITIES.utils_config as utils_config

class Camera:
    """ Game camera class for controlling the viewport. """

    def __init__(
            self,
            WORLD_WIDTH,
            WORLD_HEIGHT,
            screen_width,
            SCREEN_HEIGHT,
            restrict_bounds=False):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.WORLD_WIDTH = WORLD_WIDTH
        self.WORLD_HEIGHT = WORLD_HEIGHT
        self.screen_width = screen_width
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.speed = 30
        self.restrict_bounds = restrict_bounds  # Flag to toggle bounds restriction

    def move(self, dx, dy):
        dx /= self.zoom
        dy /= self.zoom
        self.x += dx
        self.y += dy

        if self.restrict_bounds:  # Apply bounds if restriction is enabled
            self.x = max(0, min(self.WORLD_WIDTH -
                         self.screen_width / self.zoom, self.x))
            self.y = max(0, min(self.WORLD_HEIGHT -
                         self.SCREEN_HEIGHT / self.zoom, self.y))

    def apply(self, position):
        """
        Transform world coordinates to screen coordinates relative to the camera.
        """
        x, y = position
        screen_x = (x - self.x) * self.zoom
        screen_y = (y - self.y) * self.zoom

        return screen_x, screen_y

    

    def zoom_around_mouse(self, zoom_in, mouse_x, mouse_y):
        """
        Zoom in or out, keeping the world coordinates under the mouse pointer stationary.
        :param zoom_in: Boolean indicating whether to zoom in or out.
        :param mouse_x: Mouse x position in screen coordinates.
        :param mouse_y: Mouse y position in screen coordinates.
        """
        # Calculate world coordinates of the mouse before zooming
        world_mouse_x = self.x + mouse_x / self.zoom
        world_mouse_y = self.y + mouse_y / self.zoom

        # Adjust the zoom level
        if zoom_in:
            self.zoom = min(self.zoom * 1.1, 4.0)  # Zoom in (max 400%)
        else:
            self.zoom = max(self.zoom / 1.1, 0.5)  # Zoom out (min 50%)

        # Calculate new camera position so the mouse points to the same world
        # coordinate
        self.x = world_mouse_x - mouse_x / self.zoom
        self.y = world_mouse_y - mouse_y / self.zoom

        # Ensure the camera stays within the bounds of the world
        self.x = max(0, min(self.utils_config.WORLD_WIDTH -
                     self.screen_width / self.zoom, self.x))
        self.y = max(0, min(self.utils_config.WORLD_HEIGHT -
                     self.utils_config.SCREEN_HEIGHT / self.zoom, self.y))
