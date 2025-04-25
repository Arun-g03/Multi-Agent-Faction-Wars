
import subprocess
import sys
# Run the Start_up script to check/install dependencies before anything else
startup_script = "UTILITIES\Startup_installer.py"

try:
    result = subprocess.run([sys.executable, startup_script], check=True)
except subprocess.CalledProcessError:
    sys.exit("[ERROR] Failed to verify dependencies. Exiting.")

# Continue with main program after startup script dependency check

from SHARED.core_imports import *
from GAME.game_manager import GameManager
from RENDER.Game_Renderer import GameRenderer
from RENDER.MainMenu_Renderer import MenuRenderer
import UTILITIES.utils_config as utils_config


class MainGame:
    def __init__(self):
        self.MainTITLE = "Multi-agent competitive and cooperative strategy (MACCS) - Main Menu"
        
       
        

    def main(self):
        """Main function to run the game."""
        try:
            # Initialise pygame
            pygame.init()
            screen = pygame.display.set_mode(
                (utils_config.SCREEN_WIDTH, utils_config.SCREEN_HEIGHT))
            pygame.display.set_caption(self.MainTITLE)

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
            program_running = True  # Track if the program is running
            game_running = False  # Track if the game is running

            while program_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\033[91m[INFO] Window closed. Exiting game..\033[0m")
                        if utils_config.ENABLE_TENSORBOARD:
                            try:
                                tensorboard_logger = TensorBoardLogger()
                                tensorboard_logger.stop_tensorboard()  # Stop TensorBoard if running
                            except Exception as e:
                                print(f"[ERROR] Failed to stop TensorBoard: {e}")

                        program_running = False  # Stop the loop
                        break  # Exit event processing

                if is_menu:
                    is_menu = menu_renderer.render_menu()

                    # Only act once the menu has completed
                    if not is_menu and hasattr(menu_renderer, "pending_game_config"):
                        config = menu_renderer.pending_game_config
                        try:
                            game_manager = self.start_game(
                                screen=screen,
                                mode=config["mode"],
                                load_existing=config.get("load_existing", False),
                                models_to_load=config.get("models", {})
                            )
                        except (TypeError, AttributeError):
                            print("[ERROR] Invalid menu configuration")
                            is_menu = True
                            menu_renderer.render_menu()
                            continue

                        if game_manager is None:
                            print("[ERROR] GameManager is None. Exiting.")
                            program_running = False
                            game_running = False
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
                        game_running = True  # Mark game as running

                elif is_game:
                    try:
                        if game_manager and game_renderer:
                            try:
                                # Check the game state (running or not)
                                game_running = game_manager.run()  # Run the game loop and check if it should continue

                                if not game_running:
                                    print("\033[91m[INFO] Exiting game after run() stopped.\033[0m")
                                    # Reset game state and return to the menu
                                    is_menu = True
                                    is_game = False
                                    print("[INFO] Returning to main menu.")
                                    game_manager = None
                                    game_renderer = None
                                    # Transition to menu
                                    menu_renderer.render_menu()  # Switch back to the menu
                                    game_running = False

                                # If game is still running, render the game
                                if game_running:
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

                                    clock.tick(utils_config.FPS)
                                
                                # If the game is done, transition back to the menu
                                

                            except SystemExit:
                                print("Whoops - SystemExit - Main.py")
                                print("[INFO] Game closed successfully.")
                                if utils_config.ENABLE_TENSORBOARD:
                                    tensorboard_logger = TensorBoardLogger()
                                    tensorboard_logger.stop_tensorboard()  # Stop TensorBoard if running
                                pygame.quit()
                                sys.exit()

                            except Exception as e:
                                print(f"Unexpected error during the game loop: {e}")
                                traceback.print_exc()
                                break
                        else:
                            is_menu = True
                            print("[ERROR] Game manager or Game Renderer is None. Returning to menu.")

                    except Exception as e:
                        print(f"An error occurred: {e}")
                        traceback.print_exc()
                        break
            print ("[INFO] Game closed successfully.")
            print ("Bye!")
        
            pygame.quit()  # Ensure Pygame fully shuts down
            sys.exit()  # Ensure Python exits completely

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        
            self.cleanup(QUIT=True)
        

        


    def start_game(
        self,
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

    def cleanup(self, QUIT):
        if utils_config.ENABLE_TENSORBOARD:
            tensorboard_logger = TensorBoardLogger()
            tensorboard_logger.stop_tensorboard()  # Kill TensorBoard if running

        

        if QUIT:
            pygame.quit()
            sys.exit()  # Ensure the system fully exits when quitting the game
            print("[INFO] - Game_manager.py ---- Game closed successfully.")

# Run the main function with profiling if enabled
if __name__ == "__main__":
    game = MainGame()
    if utils_config.ENABLE_PROFILE_BOOL:
        profile_function(game.main)
    else:
        game.main()
