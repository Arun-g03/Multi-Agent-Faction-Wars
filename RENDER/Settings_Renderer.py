"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from RENDER.Common import FONT_NAME, WHITE, BLACK, BLUE, GREEN, RED, GREY, DARK_GREY
#    ____       _   _   _                   __  __
#   / ___|  ___| |_| |_(_)_ __   __ _ ___  |  \/  | ___ _ __  _   _
#   \___ \ / _ \ __| __| | '_ \ / _` / __| | |\/| |/ _ \ '_ \| | | |
#    ___) |  __/ |_| |_| | | | | (_| \__ \ | |  | |  __/ | | | |_| |
#   |____/ \___|\__|\__|_|_| |_|\__, |___/ |_|  |_|\___|_| |_|\__,_|
#                               |___/


class SettingsMenuRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(FONT_NAME, 24)
        self.selected_category = "debugging"
        self.saved = False
        self.input_mode = False
        self.input_field = None
        self.input_text = ""
        self.cursor_visible = True
        self.cursor_timer = 0

        # Define category labels
        self.sidebar_items = [
            "debugging", "episode settings", "screen",
            "world", "resources", "agent", "faction"
        ]

        # Store config state and original defaults
        self.settings_by_category = {}
        self.defaults = {}

        raw_settings = {
            "debugging": [
                ("TensorBoard", "ENABLE_TENSORBOARD"),
                ("Logging", "ENABLE_LOGGING"),
                ("Profiling", "ENABLE_PROFILE_BOOL"),

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
                ("Agent Spawn Radius", "HQ_Agent_Spawn_Radius", 2),
                ("Faction Count", "FACTON_COUNT", 1),
                ("Initial Gatherers", "INITAL_GATHERER_COUNT", 1),
                ("Initial Peacekeepers", "INITAL_PEACEKEEPER_COUNT", 1),
            ]
        }

        for category, fields in raw_settings.items():
            self.settings_by_category[category] = []
            for item in fields:
                label, key = item[:2]
                value = getattr(utils_config, key)
                self.defaults[key] = value
                setting = {"label": label, "key": key, "value": value}
                if len(item) == 3:
                    setting["step"] = item[2]
                else:
                    setting["options"] = [True, False]
                self.settings_by_category[category].append(setting)

        self.check_icon = pygame.font.SysFont(
            FONT_NAME, 40).render('✔', True, GREEN)
        self.cross_icon = pygame.font.SysFont(
            FONT_NAME, 40).render('❌', True, RED)

    def draw_text(self, text, size, colour, x, y):
        font_obj = pygame.font.SysFont(FONT_NAME, size)
        text_surface = font_obj.render(text, True, colour)
        text_rect = text_surface.get_rect(topleft=(x, y))
        self.screen.blit(text_surface, text_rect)

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

        self.draw_text(text, size, WHITE, x + 10, y + 10)

        if icon:
            icon_size = 40
            icon_x = x + width - icon_size - 10
            icon_y = y + (height // 2) - (icon_size // 2)
            self.screen.blit(icon, (icon_x, icon_y))

        return button_rect

    def render(self):
        self.screen.fill(BLACK)
        self.cursor_timer += 1
        if self.cursor_timer >= 30:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0

        for idx, label in enumerate(self.sidebar_items):
            y = 80 + idx * 60
            selected = (label == self.selected_category)
            color = GREY if selected else DARK_GREY
            btn = self.create_button(
                label.upper(), FONT_NAME, 20, color, (120, 120, 120), (180, 180, 180), 20, y, 200, 40)
            if pygame.mouse.get_pressed()[0] and btn.collidepoint(
                    pygame.mouse.get_pos()):
                self.selected_category = label

        settings = self.settings_by_category.get(self.selected_category, [])
        step_buttons = []
        for i, setting in enumerate(settings):
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

            elif "step" in setting:
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

        back_btn = self.create_button(
            "Back", FONT_NAME, 20, GREY, (180, 180, 180), (120, 120, 120), 250, 500, 150, 50)
        save_return_btn = self.create_button(
            "Save and Return", FONT_NAME, 20, BLUE, (80, 80, 255), (50, 50, 200),
            450, 500, 250, 50)
        reset_all_btn = self.create_button(
            "Reset All", FONT_NAME, 20, GREY, (180, 180, 180), (120, 120, 120),
            20, 500, 200, 50)
        note_text = "Tip: You can click on a value to input in a custom number. Type and press Enter to confirm"
        screen_width = self.screen.get_width()
        self.draw_text(note_text, 24, GREY, 50, 560)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, setting in enumerate(settings):
                    y = 80 + i * 60
                    # matches where you draw the values
                    value_rect = pygame.Rect(600, y, 80, 30)
                    if value_rect.collidepoint(
                            mouse_x, mouse_y) and "step" in setting:
                        self.input_mode = True
                        self.input_field = setting
                        self.input_text = str(setting["value"])
                        clicked_input = True

                if back_btn.collidepoint(event.pos):
                    return False
                if save_return_btn.collidepoint(event.pos):
                    self.saved = True
                    return False
                if reset_all_btn.collidepoint(event.pos):
                    for settings in self.settings_by_category.values():
                        for setting in settings:
                            setting["value"] = self.defaults.get(
                                setting["key"], setting["value"])
                for action, rect, setting in step_buttons:
                    if rect.collidepoint(event.pos):
                        if action == "toggle":
                            options = setting["options"]
                            current_index = options.index(setting["value"])
                            setting["value"] = options[(
                                current_index + 1) % len(options)]
                        elif action == "minus":
                            setting["value"] = round(
                                setting["value"] - setting["step"], 3)
                        elif action == "plus":
                            setting["value"] = round(
                                setting["value"] + setting["step"], 3)
                        elif action == "reset":
                            default_val = self.defaults.get(
                                setting["key"], setting["value"])
                            setting["value"] = default_val
                clicked_input = False
                for i, setting in enumerate(settings):
                    y = 80 + i * 60
                    value_rect = pygame.Rect(600, y, 80, 30)
                    if value_rect.collidepoint(
                            mouse_x, mouse_y) and "step" in setting:
                        self.input_mode = True
                        self.input_field = setting
                        self.input_text = str(setting["value"])
                        clicked_input = True

                if not clicked_input:
                    self.input_mode = False
                    self.input_field = None

            if event.type == pygame.KEYDOWN and self.input_mode:
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

        return True

    def get_settings(self):
        settings = {}
        for cat in self.settings_by_category.values():
            for setting in cat:
                settings[setting["key"]] = setting["value"]
        return settings


