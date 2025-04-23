

from SHARED.core_imports import *
from GAME.game_manager import GameManager


from RENDER.Game_Renderer import GameRenderer
from RENDER.MainMenu_Renderer import MenuRenderer
import UTILITIES.utils_config as utils_config



# Run the Start_up script to check/install dependencies before anything else
startup_script = "UTILITIES\Startup_installer.py"

try:
    result = subprocess.run([sys.executable, startup_script], check=True)
except subprocess.CalledProcessError:
    sys.exit("[ERROR] Failed to verify dependencies. Exiting.")

# Continue with main program after startup script dependency check


# Constants for the menu screen

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
                is_menu = menu_renderer.render_menu()

                # 🚨 Only act once menu has completed
                if not is_menu and hasattr(menu_renderer, "pending_game_config"):
                    config = menu_renderer.pending_game_config

                    game_manager = start_game(
                        screen=screen,
                        mode=config["mode"],
                        load_existing=config.get("load_existing", False),
                        models_to_load=config.get("models", {})
                    )

                    if game_manager is None:
                        print("[ERROR] GameManager is None. Exiting.")
                        running = False
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
                            tensorboard_logger = TensorBoardLogger()
                            tensorboard_logger.stop_tensorboard() # Kill TensorBoard if running
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
                        tensorboard_logger = TensorBoardLogger()
                        tensorboard_logger.stop_tensorboard() # Kill TensorBoard if running
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
        tensorboard_logger = TensorBoardLogger()
        tensorboard_logger.stop_tensorboard()

        print ("[INFO] Game closed successfully.")
        print ("Bye!")
        sys.exit()  # Ensure Python exits completely

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        pygame.quit()
        tensorboard_logger = TensorBoardLogger()
        tensorboard_logger.stop_tensorboard() # Kill TensorBoard if running
        sys.exit()


def start_game(
    screen,
    # plain string: "train" | "evaluate"
    mode="train",
    # True  → load .pth files you picked in the menu
    load_existing=False,
    # dict of paths → {"Agents": "...", "HQ": "...", …}
    models_to_load=None
                        ):
    """
    Initialise and start the game.

    Returns:
        GameManager instance on success, or None on failure.
    """
    try:
        print(f"[INFO] Starting game in {mode} mode…")

        # ── create GameManager ────────────────────────────────────────────────
        game_manager = GameManager(
            screen=screen,
            mode=mode,
            load_existing=load_existing,
            models=models_to_load or {}      # keep an empty dict if None
            
        )

        # ── call its own initialise routine ──────────────────────────────────
        game_manager.Initialise(mode)
        print("[INFO] GameManager initialised successfully.")

        return game_manager

    except Exception as e:
        print(f"[ERROR] Could not start the game: {e}")
        traceback.print_exc()
        return None


# Run the main function with profiling if enabled
if __name__ == "__main__":
    if utils_config.ENABLE_PROFILE_BOOL:
        profile_function(main)
    else:
        main()
