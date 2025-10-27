class Camera:
    """Game camera class for controlling the viewport."""

    def __init__(
        self,
        WORLD_WIDTH,
        WORLD_HEIGHT,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        restrict_bounds=False,
    ):
        self.x = 0
        self.y = 0
        self.zoom = 1.0
        self.WORLD_WIDTH = WORLD_WIDTH
        self.WORLD_HEIGHT = WORLD_HEIGHT
        self.screen_width = SCREEN_WIDTH  # Fix: lowercase
        self.screen_height = SCREEN_HEIGHT  # Fix: lowercase
        self.speed = 30
        self.restrict_bounds = restrict_bounds  # Flag to toggle bounds restriction

    def move(self, dx, dy):
        """Move the camera by (dx, dy), adjusted for current zoom."""
        dx /= self.zoom
        dy /= self.zoom
        self.x += dx
        self.y += dy

        if self.restrict_bounds:
            self.x = max(
                0, min(self.WORLD_WIDTH - self.screen_width / self.zoom, self.x)
            )
            self.y = max(
                0, min(self.WORLD_HEIGHT - self.screen_height / self.zoom, self.y)
            )

    def apply(self, position):
        """Transform world coordinates to screen coordinates relative to the camera."""
        x, y = position
        screen_x = (x - self.x) * self.zoom
        screen_y = (y - self.y) * self.zoom
        return screen_x, screen_y

    def zoom_around_mouse(self, zoom_in, mouse_x, mouse_y, enabled=False):
        """
        Zoom in or out, keeping the world coordinates under the mouse pointer stationary.
        :param zoom_in: Boolean indicating whether to zoom in or out.
        :param mouse_x: Mouse x position in screen coordinates.
        :param mouse_y: Mouse y position in screen coordinates.
        :param enabled: Boolean to enable/disable zooming.
        """
        if not enabled:
            return

        # 1. Save world coordinates under mouse
        world_mouse_x = self.x + mouse_x / self.zoom
        world_mouse_y = self.y + mouse_y / self.zoom

        # 2. Adjust zoom level
        zoom_factor = 1.1
        if zoom_in:
            self.zoom = min(self.zoom * zoom_factor, 4.0)
        else:
            self.zoom = max(self.zoom / zoom_factor, 0.5)

        # 3. Adjust camera x, y to keep mouse pointing same world spot
        self.x = world_mouse_x - mouse_x / self.zoom
        self.y = world_mouse_y - mouse_y / self.zoom

        # 4. Bounds check if enabled
        if self.restrict_bounds:
            self.x = max(
                0, min(self.WORLD_WIDTH - self.screen_width / self.zoom, self.x)
            )
            self.y = max(
                0, min(self.WORLD_HEIGHT - self.screen_height / self.zoom, self.y)
            )
