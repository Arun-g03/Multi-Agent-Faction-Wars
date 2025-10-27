import subprocess
import sys
import os

# Run the Start_up script to check/install dependencies FIRST
startup_script = os.path.join("UTILITIES", "Startup_installer.py")

print("[INFO] - Main.py ---- Starting the game...\n")
print("[INFO] Checking and installing dependencies...")

# Run the startup installer BEFORE any other imports
try:
    result = subprocess.run([sys.executable, startup_script], check=True)
except subprocess.CalledProcessError:
    sys.exit("[ERROR] Failed to verify dependencies. Exiting.")

# Prompt for headless mode after dependencies are installed
print("\nWould you like to run in HEADLESS_MODE?")
print(
    "\033[93mHEADLESS MODE WILL DISABLE GAME RENDERING BUT WILL ALLOW FOR BETTER PERFORMANCE\033[0m"
)

print("\nEnter y for yes n for no: ", end="")
user_input = input().strip().lower()
headless_response = user_input == "y"

# Inject HEADLESS_MODE before utils_config is imported
with open("UTILITIES/utils_config.py", "r") as f:
    config_lines = f.readlines()

with open("UTILITIES/utils_config.py", "w") as f:
    for line in config_lines:
        if line.startswith("HEADLESS_MODE"):
            f.write(f"HEADLESS_MODE = {str(headless_response)}\n")
        else:
            f.write(line)

# Continue with main program after startup script dependency check
from SHARED.core_imports import *
from GAME.game_manager import GameManager
from RENDER.Game_Renderer import GameRenderer
from RENDER.MainMenu_Renderer import MenuRenderer
import UTILITIES.utils_config as utils_config

if utils_config.ENABLE_TENSORBOARD:
    from SHARED.core_imports import tensorboard_logger


class MainGame:
    def __init__(self):
        self.MainTITLE = (
            "Multi-agent competitive and cooperative strategy (MACCS) - Main Menu"
        )

    def main(self):
        """Main function to run the game."""
        try:
            # Initialise pygame
            pygame.init()
            screen = pygame.display.set_mode(
                (utils_config.SCREEN_WIDTH, utils_config.SCREEN_HEIGHT)
            )
            pygame.display.set_caption(self.MainTITLE)
            clock = pygame.time.Clock()

            # Setup menu
            menu_renderer = MenuRenderer(screen)
            is_menu = True
            is_game = False
            program_running = True
            game_running = False
            game_manager = None
            game_renderer = None

            while program_running:
                # Handle events (only in GUI mode)
                if not utils_config.HEADLESS_MODE:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\033[91m[INFO] Window closed. Exiting game..\033[0m")
                            if utils_config.ENABLE_TENSORBOARD:
                                try:
                                    tensorboard_logger.stop_tensorboard()
                                except Exception as e:
                                    print(f"[ERROR] Failed to stop TensorBoard: {e}")
                            program_running = False
                            break
                else:
                    # Sleep briefly to avoid CPU overload in headless idle state
                    if is_menu or not game_running:
                        time.sleep(0.001)

                # Menu logic
                if is_menu:
                    is_menu = menu_renderer.render_menu()

                    # If the menu exits with config, start the game
                    config = getattr(menu_renderer, "pending_game_config", None)
                    if not is_menu and isinstance(config, dict):
                        try:
                            game_manager = self.start_game(
                                screen=screen,
                                mode=config["mode"],
                                load_existing=config.get("load_existing", False),
                                models_to_load=config.get("models", {}),
                            )
                        except Exception as e:
                            print(f"[ERROR] Failed to start game: {e}")
                            traceback.print_exc()
                            is_menu = True
                            continue

                        if game_manager is None:
                            print("[ERROR] GameManager is None. Exiting.")
                            program_running = False
                            break

                        game_renderer = GameRenderer(
                            screen=screen,
                            terrain=game_manager.terrain,
                            resources=game_manager.resource_manager.resources,
                            factions=game_manager.faction_manager.factions,
                            agents=game_manager.agents,
                            camera=game_manager.camera,
                        )

                        print("[INFO] Switching to GameRenderer...")
                        is_game = True
                        game_running = True

                # Game loop
                elif is_game:
                    try:
                        if game_manager and game_renderer:
                            if utils_config.HEADLESS_MODE and pygame.display.get_init():
                                print(
                                    "[INFO] HEADLESS_MODE is active — closing display."
                                )
                                pygame.display.quit()  # Disable rendering from this point on

                            game_running = game_manager.run()

                            if not game_running:
                                print(
                                    "\033[91m[INFO] Exiting game after run() stopped.\033[0m"
                                )

                                # Reinit display before returning to menu
                                if (
                                    utils_config.HEADLESS_MODE
                                    and not pygame.display.get_init()
                                ):
                                    print(
                                        "[INFO] Reinitializing display to return to main menu..."
                                    )
                                    pygame.display.init()
                                    screen = pygame.display.set_mode(
                                        (
                                            utils_config.SCREEN_WIDTH,
                                            utils_config.SCREEN_HEIGHT,
                                        )
                                    )
                                    pygame.display.set_caption(self.MainTITLE)
                                    clock = pygame.time.Clock()
                                    menu_renderer = MenuRenderer(
                                        screen
                                    )  # Re-create with new screen

                                print("[INFO] Returning to main menu.")
                                is_menu = True
                                is_game = False
                                game_manager = None
                                game_renderer = None
                                continue  # Jump back to top of while loop

                            game_renderer.render(
                                game_manager.camera,
                                game_manager.terrain,
                                game_manager.resource_manager.resources,
                                game_manager.faction_manager.factions,
                                game_manager.agents,
                                step=game_manager.current_step,
                                episode=game_manager.episode,
                                resource_counts=game_manager.resource_counts,
                            )

                            if not utils_config.HEADLESS_MODE:
                                clock.tick(utils_config.FPS)
                            else:
                                clock.tick(0)

                        else:
                            print(
                                "[ERROR] GameManager or GameRenderer missing. Returning to menu."
                            )
                            is_menu = True
                            is_game = False

                    except SystemExit:
                        print("Whoops - SystemExit - Main.py")
                        print("[INFO] Game closed successfully.")
                        if utils_config.ENABLE_TENSORBOARD:
                            tensorboard_logger.stop_tensorboard()
                        pygame.quit()
                        sys.exit()

                    except Exception as e:
                        print(f"[ERROR] Unexpected error in game loop: {e}")
                        traceback.print_exc()
                        is_menu = True
                        is_game = False

            print("[INFO] Game closed successfully. Bye!")
            pygame.quit()
            sys.exit()

        except Exception as e:
            print(f"[FATAL] An error occurred in main(): {e}")
            traceback.print_exc()
            cleanup(QUIT=True)

    def start_game(
        self,
        screen,
        # plain string: "train" | "evaluate"
        mode="train",
        # True  → load .pth files you picked in the menu
        load_existing=False,
        # dict of paths → {"Agents": "...", "HQ": "...", …}
        models_to_load=None,
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
                models=models_to_load or {},  # keep an empty dict if None
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
    game = MainGame()
    if utils_config.ENABLE_PROFILE_BOOL:
        profile_function(game.main)
    else:
        game.main()
