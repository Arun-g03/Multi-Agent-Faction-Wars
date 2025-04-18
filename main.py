

from SHARED.core_imports import *
from GAME.game_manager import GameManager


from RENDER.render_display import GameRenderer, MenuRenderer


# Run the Start_up script to check/install dependencies before anything else
startup_script = "UTILITIES\Startup_installer.py"

try:
    result = subprocess.run([sys.executable, startup_script], check=True)
except subprocess.CalledProcessError:
    sys.exit("[ERROR] Failed to verify dependencies. Exiting.")

# Continue with main program after startup script dependency check


# Constants for the menu screen
FONT_NAME = "Arial"
TITLE = "Multi-agent competitive and cooperative strategy (MACCS) - Main Menu"

# Initialise tensorboard logger
tensorboard_logger = TensorBoardLogger(log_dir="RUNTIME_LOGS\Tensorboard_logs")


def main():
    """Main function to run the game."""
    try:
        # Initialise pygame
        pygame.init()
        screen = pygame.display.set_mode(
            (utils_config.SCREEN_WIDTH, utils_config.SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)

        # Create instances of MenuRenderer
        menu_renderer = MenuRenderer(screen)

        # Track game state
        is_menu = True
        is_game = False
        game_manager = None
        game_renderer = None

        # Track menu options
        mode = None
        utils_config.ENABLE_TENSORBOARD = False
        auto_ENABLE_TENSORBOARD = False
        clock = pygame.time.Clock()

        # Main loop
        running = True  # Track if the game is running
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\033[91m[INFO] Window closed. Exiting game..\033[0m")
                    if utils_config.ENABLE_TENSORBOARD:
                        try:
                            TensorBoardLogger().close()
                        except BaseException:
                            raise

                    running = False  # Stop the loop
                    break  # Exit event processing

            if is_menu:
                is_menu = menu_renderer.render_menu(
                    utils_config.ENABLE_TENSORBOARD,
                    auto_ENABLE_TENSORBOARD,
                    mode,
                    start_game_callback=lambda m,
                    d,
                    t: start_game(
                        screen,
                        m,
                        d,
                        t))

                if not is_menu and menu_renderer.selected_mode:
                    game_manager = start_game(
                        screen=screen,
                        mode=menu_renderer.selected_mode,
                        ENABLE_TENSORBOARD=utils_config.ENABLE_TENSORBOARD,
                        auto_ENABLE_TENSORBOARD=auto_ENABLE_TENSORBOARD
                    )

                    if game_manager is None:
                        print("GameManager failed to Initialise. Exiting.")
                        running = False  # Stop the loop
                        break

                    game_renderer = GameRenderer(
                        screen=screen,
                        terrain=game_manager.terrain,
                        resources=game_manager.resource_manager.resources,
                        factions=game_manager.faction_manager.factions,
                        agents=game_manager.agents,
                        camera=game_manager.camera
                    )

                    print("[INFO] Switching to GameRenderer...")
                    is_game = True

            elif is_game:
                if game_manager and game_renderer:
                    try:
                        running = game_manager.run()  # Run the game loop and check if it should continue

                        if not running:  # If `False`, stop execution
                            print(
                                "\033[91m[INFO] Exiting game after run() stopped.\033[0m")
                            pygame.quit()
                            sys.exit()

                        game_renderer.render(
                            game_manager.camera,
                            game_manager.terrain,
                            game_manager.resource_manager.resources,
                            game_manager.faction_manager.factions,
                            game_manager.agents
                        )

                        clock.tick(utils_config.FPS)
                    except SystemExit:
                        print("[INFO] Game closed successfully.")
                        pygame.quit()
                        sys.exit()
                    except Exception as e:
                        print(f"Unexpected error during the game loop: {e}")
                        traceback.print_exc()
                        break
                else:
                    is_menu = True
                    print(
                        "[ERROR] Game manager or Game Renderer is None. Returning to menu.")

        print("\033[91m[INFO] Exiting game...\033[0m")
        pygame.quit()  # Ensure Pygame fully shuts down
        sys.exit()  # Ensure Python exits completely

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        pygame.quit()
        sys.exit()


def start_game(
        screen,
        mode,
        ENABLE_TENSORBOARD=True,
        auto_ENABLE_TENSORBOARD=True):
    """
    Initialises and starts the game.
    """
    try:
        print(f"[INFO] Starting game in {mode} mode...")

        # Initialise GameManager
        game_manager = GameManager(screen=screen, mode=mode)

        # Initialise the game with the selected mode
        game_manager.Initialise(mode)

        print("[INFO] GameManager Initialised successfully.")

        return game_manager  # Return GameManager so `main()` can run it

    except Exception as e:
        print(f"Error starting the game: {e}")
        traceback.print_exc()
        return None  # Return None to indicate failure


# Run the main function with profiling if enabled
if __name__ == "__main__":
    if utils_config.ENABLE_PROFILE_BOOL:
        profile_function(main)
    else:
        main()
