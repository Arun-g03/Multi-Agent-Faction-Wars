"""
This file contains the GameManager class, which is responsible for managing the game state, including the game loop, game rules, and rendering.

"""
"""Common Imports"""
from SHARED.core_imports import *

"""File Specific Imports"""
from AGENT.Agent_Types import Peacekeeper, Gatherer
from AGENT.agent_factions import Faction
from AGENT.agent_faction_manager import FactionManager
from AGENT.agent_communication import CommunicationSystem
from NEURAL_NETWORK.Common import Training_device, save_checkpoint, load_checkpoint, clone_best_agents
from GAME.game_rules import check_victory, calculate_resource_victory_targets
from ENVIRONMENT.env_terrain import Terrain
from ENVIRONMENT.env_resources import AppleTree, GoldLump, ResourceManager
from GAME.camera import Camera
from RENDER.Game_Renderer import GameRenderer
from RENDER.MainMenu_Renderer import MenuRenderer
import UTILITIES.utils_config as utils_config






"""Importing codebase things to bring the componenets together and form the game"""

"""Importing necessary constants"""

"""Logging  and profiling for performance and execution analysis"""
"""Neural Network libraries and GPU enabling"""
device = Training_device


class GameManager:
    def __init__(self,
                 screen,
                 mode,
                 save_dir="NEURAL_NETWORK/saved_models/Agents",
                 load_existing=False,
                 models=None):
        
        # Initialise a logger specific to GameManager
        self.logger = Logger(
            log_file="game_manager_log.txt", log_level=logging.DEBUG)
        self.metric_history = defaultdict(lambda: defaultdict(list))
        # Structure: {faction_id: {metric_name: [values over episodes]}}

        
        # Log initialisation
        if utils_config.ENABLE_LOGGING:
            self.logger.log_msg(
                "GameManager initialised. Logs cleared for a new game.",
                level=logging.INFO)
        

        self.load_existing = load_existing
        self.models = models or {}
        self.mode = mode or {} # Set the mode

        self.set_mode(
            mode,                          # 'train' | 'evaluate'
            load_existing=self.load_existing,
            models=self.models
        )

        # Initialise the terrain
        self.terrain = Terrain()
        self.faction_manager = FactionManager()  # Initialise faction manager
        self.faction_manager.factions = []  # Ensure factions list is initialised
        self.agents = []  # List to store all agents
        self.screen = screen  # Initialise the screen
        
        self.episode = 1

        # Initialise the resource manager with the terrain
        self.resource_manager = ResourceManager(self.terrain)

        self.resource_spawn_history = {
            "episode": [],
            "gold_lumps": [],
            "gold_quantity": [],
            "apple_trees": [],
            "apple_quantity": []
        }
        self.camera = Camera(
            utils_config.WORLD_WIDTH,
            utils_config.WORLD_HEIGHT,
            utils_config.SCREEN_WIDTH,
            utils_config.SCREEN_HEIGHT)
        try:
            # Initialise GameRenderer
            self.renderer = GameRenderer(
                self.screen,  # Pass the screen for rendering
                self.terrain,  # Pass terrain
                # Pass resources (ensure this is correctly initialised)
                self.resource_manager.resources,
                self.faction_manager.factions,  # Pass factions
                self.agents,  # Pass agents
                self.camera  # Pass camera for any rendering requirements
            )
        except Exception as e:
            raise (f"Error Initialising GameRenderer: {e}")

        
        self.current_step = 0  # Initialise the current step

        """Current step in episode (tick counter)"""
        self.save_dir = save_dir  # Ensure save_dir is a string (directory path)
        
        self.last_activity_step = 0
        self.communication_system = None  # Communication system

        # Initialise factions and agents
        self.Initialise_factions()

        # Pass resource manager, agents, factions, and renderer to EventManager
        self.event_manager = EventManager(
            resource_manager=self.resource_manager,
            faction_manager=self.faction_manager,
            agents=self.agents,
            renderer=self.renderer,
            camera=self.camera
        )

        # Create directory for saved models if it doesn't exist
        # Here `save_dir` should be a valid path string
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.models_loaded_once = False # Ensure models are loaded only once at the start
        self.exit_after_episode = False # Set to True if you want to exit the sim loop and return to menu after an episode


        
    
        

    
    def set_mode(
        self,
        mode: Union[str, Dict[str, Any]],  # can be 'train', 'evaluate', or a config dict
        load_existing: bool = False,
        models: Optional[Dict[str, str]] = None
    ):
        """
        Sets the game mode and configures loading preferences.

        Supports:
            set_mode('train')
            set_mode({
                'mode': 'train',
                'load_existing': True,
                'models': {
                    'gatherer': 'path/to/gatherer.pth',
                    'peacekeeper': 'path/to/peacekeeper.pth',
                    'HQ': 'path/to/hq.pth'
                }
            })
        """

        # ── 1) Unpack dict if needed ───────────────────────────────────────
        if isinstance(mode, dict):
            load_existing = mode.get('load_existing', load_existing)
            models        = mode.get('models', models)
            mode          = mode.get('mode', 'train')

        # ── 2) Validate mode ───────────────────────────────────────────────
        if mode not in ('train', 'evaluate'):
            raise ValueError("mode must be 'train' or 'evaluate'")

        # ── 3) Store config ────────────────────────────────────────────────
        self.mode = mode
        self.load_existing = load_existing
        self.models = models or {}

        print(f"[MODE] Game mode set to:       {self.mode}")
        print(f"[MODE] Load existing models:   {self.load_existing}")
        print(f"[MODE] Models to load:         {self.models}")

            




    def Initialise_factions(self):
        try:
            # Example: Initialise factions with the specified network type
            # (e.g., PPOModel)
            network_type = "PPOModel"  # This can be dynamic based on the needs
            for faction in self.faction_manager.factions:
                faction.initialise_network(network_type)
                print(
                    f"initialised network for faction {faction.name} with type {network_type}")
        except Exception as e:
            print(f"An error occurred in Initialise_factions: {e}")
            traceback.print_exc()


#        _
#    ___| |_ ___ _ __
#   / __| __/ _ \ '_ \
#   \__ \ ||  __/ |_) |
#   |___/\__\___| .__/
#               |_|

    

    def step(self):

        """
        Executes a single game step with episode.
        """
        try:
            # Handle camera movement input
            if not utils_config.HEADLESS_MODE:
                keys = pygame.key.get_pressed()
                self.handle_camera_movement(keys)
            plotter = MatplotlibPlotter()

            #Clear prior-step action decisions (important for batching)
            for agent in self.agents:
                agent.current_action = None
                agent.log_prob = None
                agent.value = None

            # Update each agent (movement, task execution, learning)
            for agent in self.agents[:]:
                try:
                    if agent.Health <= 0:
                        print(f"Agent {agent.role} from faction {agent.faction.id} has died.")
                        if agent in agent.faction.agents:
                            agent.faction.remove_agent(agent)
                        self.agents.remove(agent)
                        continue

                    hq_state = agent.faction.aggregate_faction_state()
                    self.logger.log_msg(f"agent postion: {agent.x} {agent.y}")
                    agent.update(self.resource_manager, self.agents, hq_state, step=self.current_step, episode=self.episode)

                except Exception as e:
                    print(f"An error occurred for agent {agent.role}: {e}")
                    traceback.print_exc()

            # Let agents share what they observed
            for agent in self.agents:
                agent.observe(
                    all_agents=self.agents,
                    enemy_hq={"position": agent.faction.home_base["position"]},
                    resource_manager=self.resource_manager
                )

            # Let HQs assign strategies/tasks
            for faction in self.faction_manager.factions:
                faction.update(self.resource_manager, self.agents, self.current_step)

                # === Track the HQ strategy for each faction and log it as a heatmap ===
                # Get the current strategy for the faction (e.g., "DEFEND_HQ", "COLLECT_GOLD", etc.)
                if faction.current_strategy is not None:
                    strategy_index = utils_config.HQ_STRATEGY_OPTIONS.index(faction.current_strategy)
                    
                    # Use one-hot encoding to visualize which strategy is being used
                    one_hot_strategy = np.zeros((1, len(utils_config.HQ_STRATEGY_OPTIONS)), dtype=int)
                    one_hot_strategy[0, strategy_index] = 1
                    
                    # Define the strategy keys for X-axis labeling
                    strategy_keys = utils_config.HQ_STRATEGY_OPTIONS  # List of strategies
                    
                    plotter.add_episode_matrix(
                        name=f"Faction_{faction.id}_HQ_Strategy",
                        matrix=one_hot_strategy,
                        step=self.current_step,
                        episode=self.episode,
                        plot_type="heatmap",
                        keys=strategy_keys  # Pass strategy keys for X-axis labels
                    )
                            
                            
            

            # ========== TensorBoard Logging + Plotting ==========

            if utils_config.ENABLE_TENSORBOARD:
                role_action_buffers = defaultdict(list)
                role_task_buffers = defaultdict(list)

                plotter = MatplotlibPlotter()
                


                for agent in self.agents:
                    if agent.current_action is not None:
                        role = agent.role
                        action_index = agent.current_action

                        role_action_buffers[role].append(action_index)

                        task_type = agent.current_task.get("type", "none") if agent.current_task else "none"
                        task_index = utils_config.TASK_TYPE_MAPPING.get(task_type, 0)

                        role_task_buffers[role].append(task_index)

                        
                        # ==== Add episode matrix for Matplotlib ====
                        one_hot_action = np.zeros((1, len(utils_config.ROLE_ACTIONS_MAP[role])), dtype=int)
                        one_hot_action[0, action_index] = 1

                        # Define the keys (e.g., action names or strategy names) for X-axis labeling
                        action_keys = utils_config.ROLE_ACTIONS_MAP[role]  # List of action names for current role


                        plotter.add_episode_matrix(
                            name=f"{role}_actions (Grouped by Role)",
                            matrix=one_hot_action,
                            step=self.current_step,
                            episode=self.episode,
                            plot_type="heatmap",
                            keys=action_keys  # Pass action keys for X-axis labels
                        )

                        num_task_types = len(utils_config.TASK_TYPE_MAPPING)
                        one_hot_task = np.zeros((1, num_task_types), dtype=int)
                        one_hot_task[0, task_index] = 1

                        # Define the keys (e.g., task names) for X-axis labeling
                        task_keys = list(utils_config.TASK_TYPE_MAPPING.keys())

                        plotter.add_episode_matrix(
                            name=f"{role}_task_distribution",
                            matrix=one_hot_task,
                            step=self.current_step,
                            episode=self.episode,
                            plot_type="heatmap",
                            keys=task_keys  # Pass task keys for X-axis labels
                        )

                    # ==== Log agent models once ====
                    if not hasattr(agent, "has_logged_model"):
                        agent.has_logged_model = False

                    if not agent.has_logged_model and self.current_step == 0:
                        try:
                            real_state = agent.get_state(agent.resource_manager, self.agents, agent.faction, agent.faction.global_state)
                            state_tensor = torch.tensor(real_state, dtype=torch.float32).unsqueeze(0).to(Training_device)

                            tensorboard_logger.log_model(agent.ai, state_tensor, name=f"Agent_{agent.agent_id}_Model")

                            agent.has_logged_model = True
                        except Exception as e:
                            print(f"[TensorBoard] Failed to log model graph for agent {agent.agent_id}: {e}")

                # Log combined distributions AFTER processing all agents
                for role, actions in role_action_buffers.items():
                    tensorboard_logger.log_distribution(
                        name=f"{role}_action_distribution",
                        values=np.array(actions),
                        step=self.current_step
                    )

                for role, tasks in role_task_buffers.items():
                    tensorboard_logger.log_distribution(
                        name=f"{role}_task_distribution",
                        values=np.array(tasks),
                        step=self.current_step
                    )


            




            

            # Step forward
            self.current_step += 1

        except Exception as e:
            print(f"An error occurred in step: {e}")
            traceback.print_exc()


#                       _
#    _ __ ___  ___  ___| |_
#   | '__/ _ \/ __|/ _ \ __|
#   | | |  __/\__ \  __/ |_
#   |_|  \___||___/\___|\__|
#


    def reset(self):

        try:
            """
            Reset the environment for a new training episode.
            """
            print("Resetting the environment...")
            self.terrain = Terrain()
            self.resource_manager = ResourceManager(self.terrain)
            calculate_resource_victory_targets(
                resources=self.resource_manager.resources,
                faction_count=len(self.faction_manager.factions),
                target_ratio=utils_config.RESOURCE_VICTORY_TARGET_RATIO
            )

            

            # Reset factions and agents
            self.agents_initialised = False  # Reset flag for agents initialisation
            self.faction_manager.reset_factions(
                utils_config.FACTON_COUNT, self.resource_manager, self.agents, self)
            self.agents.clear()
            self.Initialise_agents(mode=self.mode)

            # Reset communication system
            self.communication_system = CommunicationSystem(
                self.agents, self.faction_manager.factions)
            for faction in self.faction_manager.factions:
                faction.clean_global_state()

            self.current_step = 0
        except Exception as e:
            print(f"An error occurred during reset: {e}")
            raise


#    ___       _ _   _       _ _
#   |_ _|_ __ (_) |_(_) __ _| (_)___  ___
#    | || '_ \| | __| |/ _` | | / __|/ _ \
#    | || | | | | |_| | (_| | | \__ \  __/
#   |___|_| |_|_|\__|_|\__,_|_|_|___/\___|
#

    def Initialise(self, mode):
        """
        Initialise the game environment, factions, agents, resources, and systems.
        :param mode: The mode in which to Initialise the game ('train' or 'evaluate').
        """
        try:
            
            print(f"Initialising game in {mode} mode...")

            # Store the mode
            self.mode = mode

            # Pass the Tensorboard instance to the Agent

            # Step 1: Generate Terrain
            self.terrain = Terrain()
            print("Terrain generated.")

            # Step 2: Initialise Resources
            print("Initialising resources...")
            self.resource_manager = ResourceManager(self.terrain)
            
            

            # Now set the `resources` attribute after generating them
            # Assign resources after generation
            self.resources = self.resource_manager.resources
            print("Resources generated.")

            # Step 3: Initialise Factions
            print("Initialising factions...")
            self.faction_manager.reset_factions(
                utils_config.FACTON_COUNT, self.resource_manager, self.agents, self)
            print(f"{utils_config.FACTON_COUNT} factions initialised.")

            

            # Step 4: Initialise Agents and Place HQs
            self.agents.clear()  # Clear any existing agents
            self.Initialise_agents(mode)  # Pass the mode to agents
            print(f"{len(self.agents)} agents initialised across factions.")

            # Step 5: Initialise Communication System
            self.communication_system = CommunicationSystem(
                self.agents, self.faction_manager.factions)
            print("Communication system initialised.")

            # Step 6: Finalise initialisation
            self.current_step = 0  # Reset step counter for the new game
            self.episode = 1  # Reset episode counter for the new game
            global CurrEpisode
            CurrEpisode = self.episode
            print(f"Game initialised in {mode} mode.")


            print(
                  
                  f"Starting episode {self.mode} for {utils_config.EPISODES_LIMIT} episodes.\n",
                  f"total steps: {utils_config.STEPS_PER_EPISODE}\n"
                  
                  )
        except BaseException:
            print("An error occurred during initialisation.")
            raise


#    ___       _ _   _       _ _                                   _
#   |_ _|_ __ (_) |_(_) __ _| (_)___  ___    __ _  __ _  ___ _ __ | |_ ___
#    | || '_ \| | __| |/ _` | | / __|/ _ \  / _` |/ _` |/ _ \ '_ \| __/ __|
#    | || | | | | |_| | (_| | | \__ \  __/ | (_| | (_| |  __/ | | | |_\__ \
#   |___|_| |_|_|\__|_|\__,_|_|_|___/\___|  \__,_|\__, |\___|_| |_|\__|___/
#                                                 |___/

    # In the GameManager class

    def Initialise_agents(self, mode):
        """
        Initialise agents for the game, passing the mode to each agent for appropriate setup.
        :param mode: The mode in which to Initialise the agents ('train' or 'evaluate').
        """
        if hasattr(self, "agents_initialised") and self.agents_initialised:
            print("Agents already initialised. Skipping duplicate initialisation.")
            return
        self.agents_initialised = True
        print("Initialising agents...")

        """Creates agents and assigns them to factions using stratified grid coverage for HQ placement."""
        minimum_distance = utils_config.HQ_SPAWN_RADIUS  # Minimum distance between HQs
        print("Calculating valid positions...")
        num_factions = len(self.faction_manager.factions)

        # Step 1: Precompute valid HQ positions
        valid_positions = [
            (i, j)
            for i in range(self.terrain.grid.shape[0])
            for j in range(self.terrain.grid.shape[1])
            if self.terrain.grid[i][j]['type'] == 'land' and not self.terrain.grid[i][j]['occupied']
        ]

        # Shuffle positions for randomness
        random.shuffle(valid_positions)

        # Step 2: Select HQ positions with minimum distance enforcement
        selected_positions = []
        for pos in valid_positions:
            if all(math.dist(pos, existing_hq) >=
                   minimum_distance for existing_hq in selected_positions):
                selected_positions.append(pos)
            if len(selected_positions) == num_factions:
                break

        if len(selected_positions) < num_factions:
            print("Failure: Not enough valid positions for all factions' HQs.")
            raise ValueError(
                "Not enough valid positions for all factions' HQs.")

        # Step 3: Assign positions to factions
        for faction, position in zip(
                self.faction_manager.factions, selected_positions):
            # Convert grid position to pixel coordinates
            base_pixel_x, base_pixel_y = position[0] * \
                utils_config.CELL_SIZE, position[1] * utils_config.CELL_SIZE
            faction.home_base["position"] = (base_pixel_x, base_pixel_y)

            # Mark the HQ position in the terrain grid
            self.terrain.grid[position[0]][position[1]]['occupied'] = True
            self.terrain.grid[position[0]][position[1]]['faction'] = faction.id
            self.terrain.grid[position[0]][position[1]]['resource_type'] = None

            print(
                f"Faction {faction.id} HQ placed at grid {position}, (pixel ({base_pixel_x}, {base_pixel_y}))")

            # Step 4: Spawn agents for the faction using spawn_agent
            faction_agents = []
            print(f"Spawning agents for faction {faction.id}...")

            # Define network_type for each agent (you can modify this logic to dynamically choose the network type)
            # Example network type (can also be dynamic based on the agent role
            # or other factors)
            network_type = "PPOModel"

            print("Peacekeeper")
            # Pass the required arguments to spawn_agent
            for _ in range(utils_config.INITAL_PEACEKEEPER_COUNT):
                agent = self.spawn_agent(
                    base_x=base_pixel_x,
                    base_y=base_pixel_y,
                    faction=faction,
                    agent_class=Peacekeeper,  # Pass the agent class
                    state_size=utils_config.DEF_AGENT_STATE_SIZE,  # Pass the state size
                    role_actions=utils_config.ROLE_ACTIONS_MAP,  # Pass role-specific actions
                    communication_system=self.communication_system,  # Pass the communication system
                    event_manager=self.event_manager,  # Pass EventManager
                    # Pass network type for the agent (e.g., PPOModel,
                    # DQNModel)
                    network_type=network_type,
                )
                if agent:
                    faction_agents.append(agent)

            print("Gatherer")
            # Pass the required arguments to spawn_agent
            for _ in range(utils_config.INITAL_GATHERER_COUNT):
                agent = self.spawn_agent(
                    base_x=base_pixel_x,
                    base_y=base_pixel_y,
                    faction=faction,
                    agent_class=Gatherer,
                    state_size=utils_config.DEF_AGENT_STATE_SIZE,  # Pass the state size
                    role_actions=utils_config.ROLE_ACTIONS_MAP,  # Pass role-specific actions
                    communication_system=self.communication_system,  # Pass the communication system
                    event_manager=self.event_manager,  # Pass EventManager
                    # Pass network type for the agent (e.g., PPOModel,
                    # DQNModel)
                    network_type=network_type
                )
                if agent:
                    faction_agents.append(agent)

            # Add the newly spawned agents to the faction and the global agent
            # list
            print(f"Faction {faction.id} has {len(faction_agents)} agents.")

            # Clear and reassign to avoid shared references
            faction.agents = []  # Make sure each faction starts fresh
            faction.agents.extend(faction_agents)  # Assign its agents properly
            self.agents.extend(faction_agents)  # Keep global tracking

            if self.load_existing and self.models and not self.models_loaded_once:
                self.load_models()
                self.models_loaded_once = True
            else:
                pass #Skip model loading if load_existing is False

            # Print faction agent counts
            peacekeeper_count = sum(
                1 for agent in faction.agents if isinstance(
                    agent, Peacekeeper))
            gatherer_count = sum(
                1 for agent in faction.agents if isinstance(agent, Gatherer))
            print(
                f"Faction {faction.id} has {peacekeeper_count} Peacekeepers and {gatherer_count} Gatherers.")
            
    def load_models(self):
        """
        If load_existing is True, 
            load the models from the paths specified in mode{} from menu_Render and gameManager.Set_Mode .

        
        """
        print("[INFO] Loading model checkpoints...")
        self.logger.log_msg(f"INFO", "Loading model checkpoints...{self.models}")
        # HQ model (shared or per-faction if needed)
        hq_path = self.models.get("HQ")
        if hq_path:
            for fac in self.faction_manager.factions:
                if hasattr(fac, "network"):
                    load_checkpoint(fac.network, hq_path)
                    print(f"[LOAD] HQ checkpoint loaded for Faction {fac.id}")

        # Agent role-specific models
        for agent in self.agents:
            ckpt = self.models.get(agent.role)
            if ckpt and hasattr(agent, "ai"):
                load_checkpoint(agent.ai, ckpt)
                print(f"[LOAD] {agent.role.capitalize()} {agent.agent_id} restored from {ckpt}")



#    ____                                  _                    _
#   / ___| _ __   __ ___      ___ __      / \   __ _  ___ _ __ | |_ ___
#   \___ \| '_ \ / _` \ \ /\ / / '_ \    / _ \ / _` |/ _ \ '_ \| __/ __|
#    ___) | |_) | (_| |\ V  V /| | | |  / ___ \ (_| |  __/ | | | |_\__ \
#   |____/| .__/ \__,_| \_/\_/ |_| |_| /_/   \_\__, |\___|_| |_|\__|___/
#         |_|                                  |___/

    def spawn_agent(
            self,
            base_x,
            base_y,
            faction,
            agent_class,
            state_size,
            role_actions,
            communication_system,
            event_manager,
            network_type="PPOModel",
            action_size=None):
        """
        Spawn an agent near the given base position, ensuring it spawns on land.
        :param network_type: Network model type (e.g., "PPOModel", "DQNModel")
        :param action_size: The action size for the specific role (peacekeeper, gatherer, etc.)
        :return: The spawned agent.
        """
        spawn_radius = utils_config.HQ_Agent_Spawn_Radius
        attempts = 0
        max_attempts = 1000  # Prevent infinite loops
        max_radius = 100  # Maximum radius to prevent infinite growth

        # Maintain a persistent counter for agent IDs in the faction
        if not hasattr(faction, "next_agent_id"):
            faction.next_agent_id = 1  # Initialise if not already set

        while attempts < max_attempts:
            # Dynamically expand the spawn radius
            current_radius = spawn_radius + \
                (attempts // 100) * 10  # Expand every 100 attempts
            current_radius = min(current_radius, max_radius)  # Cap the radius

            offset_x = random.randint(-current_radius, current_radius)
            offset_y = random.randint(-current_radius, current_radius)
            spawn_x = base_x + offset_x
            spawn_y = base_y + offset_y

            # Convert pixel coordinates to grid coordinates
            grid_x = int(spawn_x // utils_config.CELL_SIZE)
            grid_y = int(spawn_y // utils_config.CELL_SIZE)

            # Check if the spawn position is valid
            if (0 <= grid_x < self.terrain.grid.shape[0] and
                0 <= grid_y < self.terrain.grid.shape[1] and
                self.terrain.grid[grid_x][grid_y]['type'] == 'land' and
                    not self.terrain.grid[grid_x][grid_y]['occupied']):

                # Mark the cell as occupied
                self.terrain.grid[grid_x][grid_y]['occupied'] = True

                # Create the agent with a unique AgentID
                agent = agent_class(
                    x=spawn_x,
                    y=spawn_y,
                    faction=faction,
                    base_sprite_path=utils_config.Peacekeeper_PNG if agent_class == Peacekeeper else utils_config.Gatherer_PNG,
                    terrain=self.terrain,
                    agents=self.agents,
                    agent_id=faction.next_agent_id,
                    resource_manager=self.resource_manager,
                    role_actions=role_actions,
                    state_size=state_size,
                    communication_system=communication_system,
                    event_manager=event_manager,
                    network_type=network_type,
                    
                )

                faction.next_agent_id += 1  # Increment the counter
                return agent

            attempts += 1

        # If no valid position found, fallback to random valid land cell
        print(f"Failed to spawn agent after {max_attempts} attempts.")
        return None

    


#    ____                 __                     _   __
#   |  _ \ _   _ _ __    / /___ _ __   ___   ___| |__\ \
#   | |_) | | | | '_ \  | |/ _ \ '_ \ / _ \ / __| '_ \| |
#   |  _ <| |_| | | | | | |  __/ |_) | (_) | (__| | | | |
#   |_| \_\\__,_|_| |_| | |\___| .__/ \___/ \___|_| |_| |
#                        \_\   |_|                   /_/

    

    def run(self):
        self.best_scores_per_role = {}
        self.global_hq_top5 = []
        running = True

        try:
            print(f"Running game in {self.mode} mode...")

            

            while running and (self.episode <= utils_config.EPISODES_LIMIT):
                self.reset()
                self.episode_reward = 0
                if not utils_config.HEADLESS_MODE:
                    print(f"\033[92mStarting {self.mode} Episode {self.episode}\033[0m")
                if utils_config.ENABLE_LOGGING:
                    self.logger.log_msg(f"Starting {self.mode} Episode", level=logging.INFO)

                self.current_step = 0
                role_rewards = {}

                # Inner bar: steps
                step_bar = None
                if utils_config.HEADLESS_MODE:
                    step_bar = tqdm(
                        total=utils_config.STEPS_PER_EPISODE,
                        desc=f"Episode {self.episode} / {utils_config.EPISODES_LIMIT} | Steps",
                        position=1,
                        leave=False
                    )

                while self.current_step < utils_config.STEPS_PER_EPISODE:
                    self.process_pygame_events()
                    self.step()
                    self.update_resources()
                    # Track resource spawn at the start of the episode
                    

                    

                    


                    if not utils_config.HEADLESS_MODE:
                        self.renderer.render(
                            self.camera, self.terrain, self.resource_manager.resources,
                            self.faction_manager.factions, self.agents,
                            self.episode, self.current_step, self.resource_counts
                        )
                        pygame.display.update()
                    else:
                        step_bar.update(1)

                    self.collect_episode_rewards()
                    
                    # Mini-batch training during episode (every 1000 steps if we have enough samples)
                    if self.mode == "train" and self.current_step % 1000 == 0 and self.current_step > 0:
                        self.train_agents_mini()

                    winner = check_victory(self.faction_manager.factions)
                    winner_id = winner.id if winner else -1
                    victory_type = getattr(winner, "victory_reason", "none") if winner else "none"

                    if not hasattr(self, "victory_history"):
                        self.victory_history = {
                            "episode": [],
                            "winner_id": [],
                            "victory_type": []
                        }

                    self.victory_history["episode"].append(self.episode)
                    self.victory_history["winner_id"].append(winner_id)
                    self.victory_history["victory_type"].append(victory_type)


                    if winner:
                        self.handle_victory(winner)
                        break

                    self.current_step += 1

                
                self.resource_spawn_history["episode"].append(self.episode)
                self.resource_spawn_history["gold_lumps"].append(self.resource_counts["gold_lumps"])
                self.resource_spawn_history["gold_quantity"].append(self.resource_counts["gold_quantity"])
                self.resource_spawn_history["apple_trees"].append(self.resource_counts["apple_trees"])
                self.resource_spawn_history["apple_quantity"].append(self.resource_counts["apple_quantity"])
                

                if utils_config.HEADLESS_MODE:
                    step_bar.close()

                if utils_config.ENABLE_TENSORBOARD:
                    tensorboard_logger.log_scalar("Episode/Steps_Taken", self.current_step, self.episode)

                plotter = MatplotlibPlotter()
                self.collect_role_rewards(role_rewards)

                hq_rewards = {
                    faction.id: faction.compute_hq_reward(victory=True)
                    for faction in self.faction_manager.factions
                }
                print(f"HQ Rewards: {hq_rewards}")
                processed_role_rewards = {}
                for (faction_id, role), agents in role_rewards.items():
                    avg = np.mean([r for _, r in agents])
                    if faction_id not in processed_role_rewards:
                        processed_role_rewards[faction_id] = {}
                    processed_role_rewards[faction_id][role] = avg

                self.log_faction_metrics(tensorboard_logger, plotter, processed_role_rewards, hq_rewards)


                if self.mode == "train":
                    self.train_agents()
                    self.save_top_models(role_rewards)
                    self.train_hq_networks(winner_id)

                plotter.flush_episode_plots(tensorboard_logger=tensorboard_logger)
                #self.print_episode_summary()

                for faction in self.faction_manager.factions:
                    faction.hq_step_rewards = []
                    faction.assigned_tasks = {}
                    faction.threats = []
                    faction.resources = []
                
                # Clear agent memories after training to prepare for next episode
                if self.mode == "train":
                    for agent in self.agents:
                        agent.ai.clear_memory()

                self.episode += 1
                
                if self.mode == "train" and self.episode > utils_config.EPISODES_LIMIT:
                    print(f"Training completed after {utils_config.EPISODES_LIMIT} episodes")
                    running = False


            
            return running

        except SystemExit:
            print("[INFO] Game closed successfully.")
        except Exception as e:
            print(f"An error occurred in {self.mode}: {e}")
            traceback.print_exc()
            cleanup(QUIT=True)



                


    def log_faction_metrics(self, tensorboard_logger, plotter, role_rewards, hq_rewards):
        if utils_config.ENABLE_TENSORBOARD:
            tensorboard_logger.log_scalar(
                "Episode/Steps_Taken", self.current_step, self.episode
            )

        # Local dictionary to track tasks for the current episode
        task_tracking = {}

        # HQ Strategy Names
        HQ_STRATEGY_OPTIONS = utils_config.HQ_STRATEGY_OPTIONS

        # At end of run() or after final episode
    
        
        plotter.plot_scalar_over_time(
            names=["Gold Lumps", "Apple Trees", "Total Gold", "Total Food"],
            values_list=[
                self.resource_spawn_history["gold_lumps"],
                self.resource_spawn_history["apple_trees"],
                self.resource_spawn_history["gold_quantity"],
                self.resource_spawn_history["apple_quantity"]
            ],
            episodes=self.resource_spawn_history["episode"],
            tensorboard_logger=tensorboard_logger
        )

        
        plotter.plot_victory_timeline(
            episodes=self.victory_history["episode"],
            winner_ids=self.victory_history["winner_id"],
            victory_types=self.victory_history["victory_type"],
            tensorboard_logger=tensorboard_logger  # or None
        )







        for faction in self.faction_manager.factions:
            faction_total_reward = 0
            agent_count = len(faction.agents)

            for agent in faction.agents:
                rewards = agent.ai.memory.get("rewards", [])
                if rewards:
                    agent_total = sum(rewards)
                    faction_total_reward += agent_total

            average_reward = faction_total_reward / agent_count if agent_count > 0 else 0

            # Store the metrics in self.metric_history for tracking over time
            self.metric_history[faction.id]["gold_balance"].append(faction.gold_balance)
            self.metric_history[faction.id]["food_balance"].append(faction.food_balance)
            self.metric_history[faction.id]["agents_alive"].append(agent_count)

            # Add role-specific rewards to the history for each role
            for role, reward in role_rewards.get(faction.id, {}).items():
                if role not in self.metric_history[faction.id]:
                    self.metric_history[faction.id][role] = []
                self.metric_history[faction.id][role].append(reward)

            # Add HQ reward to the history
            if hq_rewards is not None:
                if "hq_reward" not in self.metric_history[faction.id]:
                    self.metric_history[faction.id]["hq_reward"] = []
                self.metric_history[faction.id]["hq_reward"].append(hq_rewards[faction.id])

            if utils_config.ENABLE_TENSORBOARD:
                tensorboard_logger.log_scalar(
                    f"Faction_{faction.id}/Average_Reward", average_reward, self.episode
                )

            # --- Join Gold, Food, and Agents Alive into one grouped plot per faction ---
            plotter.plot_scalar_over_time(
                names=[f"Faction {faction.id} Gold Balance", f"Faction {faction.id} Food Balance", f"Faction {faction.id} Agents Alive"],
                values_list=[
                    self.metric_history[faction.id]["gold_balance"],
                    self.metric_history[faction.id]["food_balance"],
                    self.metric_history[faction.id]["agents_alive"]
                ],
                episodes=list(range(1, self.episode + 1)),
                tensorboard_logger=tensorboard_logger
            )

            # --- Combine Peacekeeper, Gatherer, and HQ rewards into one plot ---
            faction_rewards = []
            role_names = ["gatherer", "peacekeeper"]  # Define the roles you want to plot (adjust as necessary)

            # Add each role reward to the plot
            for role in role_names:
                if role in self.metric_history[faction.id]:
                    faction_rewards.append(self.metric_history[faction.id][role])

            # Add HQ reward if it exists
            if "hq_reward" in self.metric_history[faction.id]:
                faction_rewards.append(self.metric_history[faction.id]["hq_reward"])

            # Now plot them together
            plotter.plot_scalar_over_time(
                names=[f"Faction {faction.id} Gatherer Reward", f"Faction {faction.id} Peacekeeper Reward", f"Faction {faction.id} HQ Reward"],
                values_list=faction_rewards,
                episodes=list(range(1, self.episode + 1)),
                tensorboard_logger=tensorboard_logger
            )

            # --- Task Timeline: Clustered Stacked Bar Plot (by task type for the current episode) ---
            if hasattr(faction, "assigned_tasks") and faction.assigned_tasks:
                # For the current episode, gather the success, failure, and ongoing counts for each task type
                task_records = faction.assigned_tasks

                # Initialise counts for success, failure, ongoing tasks per task type per episode
                task_types = ['mine', 'forage', 'explore', 'defend', 'gather', 'eliminate', 'move_to']  # Define task types you want to categorize
                success_counts = {task_type: 0 for task_type in task_types}
                failure_counts = {task_type: 0 for task_type in task_types}
                ongoing_counts = {task_type: 0 for task_type in task_types}

                for task_type in task_types:
                    # Aggregate the task results for each task type
                    for task_id, task_info in task_records.items():
                        if task_info.get('type', '') == task_type:  # Directly check task type instead of task_id
                            for agent_id, result in task_info.get('agents', {}).items():
                                # Increment task counts based on the result
                                if result == utils_config.TaskState.ONGOING:
                                    ongoing_counts[task_type] += 1
                                elif result == utils_config.TaskState.SUCCESS:
                                    success_counts[task_type] += 1
                                elif result == utils_config.TaskState.FAILURE:
                                    failure_counts[task_type] += 1

                # Plot the clustered stacked bar chart for each faction in the current episode
                plotter.plot_clustered_stacked_bar_chart(
                    task_types=task_types,  # Task types (mine, forage, etc.)
                    success_counts=[success_counts[task] for task in task_types],
                    failure_counts=[failure_counts[task] for task in task_types],
                    ongoing_counts=[ongoing_counts[task] for task in task_types],
                    episodes=[self.episode],  # Only the current episode
                    tensorboard_logger=tensorboard_logger,
                    step=self.current_step,
                    name=f"Faction {faction.id} Task Timeline {self.episode}"  # Use faction ID to name the plot
                )
            

           





    def collect_episode_rewards(self):
        """
        Collect the latest step reward from all agents for this episode.
        """
        for agent in self.agents:
            if agent.ai.memory["rewards"]:
                self.episode_reward += agent.ai.memory["rewards"][-1]
















    def collect_role_rewards(self, role_rewards):
        """
        Collect rewards per role **per faction**.
        """
        for faction in self.faction_manager.factions:
            for agent in faction.agents:
                rewards = agent.ai.memory.get("rewards", [])
                if rewards:
                    role = agent.role
                    total_reward = sum(rewards)
                    # Key by (faction_id, role)
                    key = (faction.id, role)
                    role_rewards.setdefault(key, []).append((agent.ai, total_reward))


















    def save_top_models(self, role_rewards):
        agent_model_dir = "NEURAL_NETWORK/saved_models/Agents/"
        os.makedirs(agent_model_dir, exist_ok=True)

        for role, reward_list in role_rewards.items():
            best_model, best_reward = max(reward_list, key=lambda x: x[1])

            if role not in self.best_scores_per_role:
                self.best_scores_per_role[role] = []

            # Load current on-disk models for this role
            model_files = [
                (os.path.join(agent_model_dir, f), float(f.split("_reward_")[-1].replace(".pth", "")))
                for f in os.listdir(agent_model_dir)
                if f.startswith(f"Best_{role}_") and "_reward_" in f
            ]

            all_scores = list({(p, r) for p, r in model_files + self.best_scores_per_role[role] + [(None, best_reward)]})
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top5 = all_scores[:5]

            if (None, best_reward) in top5:
                save_name = f"Best_{role}_episode_{self.episode}_reward_{best_reward:.2f}.pth"
                save_path = os.path.join(agent_model_dir, save_name)
                try:
                    best_model.save_model(save_path)
                    print(f"[SAVE] Agent model for {role} saved at {save_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save model for role {role}: {e}")
                    traceback.print_exc()

                # Update paths in top5
                top5 = [(save_path if p is None else p, r) for p, r in top5]

                if utils_config.ENABLE_LOGGING:
                    self.logger.log_msg(
                        f"[SAVE] Top-5 agent model saved for {role} at {save_path} (reward={best_reward:.2f})",
                        level=logging.INFO
                    )

            # Prune older files if >5 exist
            if len(model_files) > 5:
                keep_paths = set(p for p, _ in top5)
                for f in os.listdir(agent_model_dir):
                    f_path = os.path.join(agent_model_dir, f)
                    if f_path not in keep_paths and f.startswith(f"Best_{role}_") and "_reward_" in f:
                        os.remove(f_path)
                        if utils_config.ENABLE_LOGGING:
                            self.logger.log_msg(f"[PRUNE] Removed outdated model: {f_path}", level=logging.INFO)

            self.best_scores_per_role[role] = top5

        # Clone best agents per role into bottom 50% (softly)
        for role in role_rewards:
            agents_of_role = [a for a in self.agents if a.role == role]
            clone_best_agents(agents_of_role)






    def train_agents(self):
        """
        Train agents based on their accumulated rewards and experiences.
        Uses mini-batch training during episodes and full batch at episode end.
        """
        if utils_config.ENABLE_LOGGING:
            self.logger.log_msg(
                "[TRAINING] Training agents at end of episode.",
                level=logging.INFO
            )

        trained_count = 0
        skipped_count = 0
        
        for agent in self.agents:
            if agent.mode == "train" and len(agent.ai.memory["rewards"]) > 0:
                memory_size = len(agent.ai.memory["rewards"])
                
                # Only train if we have enough samples
                if memory_size >= utils_config.MIN_MEMORY_SIZE:
                    if utils_config.ENABLE_LOGGING:
                        self.logger.log_msg(
                            f"[TRAIN CALL] Agent {agent.agent_id} training with {memory_size} samples...",
                            level=logging.INFO
                        )

                    try:
                        agent.ai.train(mode="train", batching=True)
                        trained_count += 1
                    except Exception as e:
                        print(f"Training failed for agent {agent.agent_id}: {e}")
                        traceback.print_exc()
                else:
                    skipped_count += 1
                    if utils_config.ENABLE_LOGGING:
                        self.logger.log_msg(
                            f"[TRAIN SKIP] Agent {agent.agent_id} skipped (memory: {memory_size} < {utils_config.MIN_MEMORY_SIZE})",
                            level=logging.DEBUG
                        )
        
        if utils_config.ENABLE_LOGGING and trained_count > 0:
            self.logger.log_msg(
                f"[TRAINING COMPLETE] Trained {trained_count} agents, skipped {skipped_count}",
                level=logging.INFO
            )

    def train_agents_mini(self):
        """
        Perform mini-batch training during an episode.
        Trains on recent samples without clearing memory.
        """
        for agent in self.agents:
            if agent.mode == "train" and len(agent.ai.memory["rewards"]) >= utils_config.MIN_MEMORY_SIZE:
                try:
                    # Train with batching but don't clear memory
                    agent.ai.train(mode="train", batching=True)
                except Exception as e:
                    # Silently continue if training fails during mini-batch
                    pass

    def train_hq_networks(self, winner_id):
        """
        Train the HQ strategy networks and save the top models.
        """
        for faction in self.faction_manager.factions:
            is_winner = (faction.id == winner_id)
            hq_reward = faction.compute_hq_reward(victory=is_winner)
            
            hq_reward += sum(faction.hq_step_rewards)
            print(f"HQ Reward: {hq_reward}")

            
            faction.network.update_memory_rewards(hq_reward)

            if hasattr(faction.network, "train") and faction.network.hq_memory:
                try:
                    if utils_config.ENABLE_LOGGING:
                        self.logger.log_msg(
                            f"[HQ TRAIN] Training strategy network for Faction {faction.id} with {len(faction.network.hq_memory)} samples.",
                            level=logging.INFO
                        )

                    faction.network.train(faction.network.hq_memory, faction.optimizer)

                    # -- Top-5 Tracking --
                    if not hasattr(self, "global_hq_top5"):
                        self.global_hq_top5 = []

                    hq_model_dir = "NEURAL_NETWORK/saved_models/HQ/"
                    os.makedirs(hq_model_dir, exist_ok=True)

                    # Load all HQ models regardless of faction
                    model_files = [
                        (os.path.join(hq_model_dir, f), float(f.split("_reward_")[-1].replace(".pth", "")))
                        for f in os.listdir(hq_model_dir)
                        if f.startswith(f"HQ_Faction_") and "_reward_" in f
                    ]

                    # Combine disk models + in-memory top list + current result
                    combined_scores = model_files + self.global_hq_top5 + [(None, hq_reward)]
                    combined_scores = list({(p, r) for p, r in combined_scores})  # deduplicate
                    combined_scores.sort(key=lambda x: x[1], reverse=True)

                    top5 = combined_scores[:5]

                    # Save if this reward is in the top-5 and not yet saved
                    if (None, hq_reward) in top5:
                        save_name = f"HQ_Faction_{faction.id}_episode_{self.episode}_reward_{hq_reward:.2f}.pth"
                        save_path = os.path.join(hq_model_dir, save_name)
                        if hasattr(faction.network, "save_model"):
                            faction.network.save_model(save_path)
                            print(f"[SAVE] HQ model for Faction {faction.id} saved at {save_path}")

                        # Update saved path in top5
                        top5 = [(save_path if p is None else p, r) for p, r in top5]

                        if utils_config.ENABLE_LOGGING:
                            self.logger.log_msg(
                                f"[SAVE] Top-5 HQ model saved globally at {save_path} (reward={hq_reward:.2f})",
                                level=logging.INFO
                            )

                    # Prune any excess beyond global top 5
                    if len(model_files) > 5:
                        keep_paths = set(p for p, _ in top5)
                        for f in os.listdir(hq_model_dir):
                            f_path = os.path.join(hq_model_dir, f)
                            if f_path not in keep_paths and f.startswith(f"HQ_Faction_") and "_reward_" in f:
                                os.remove(f_path)
                                if utils_config.ENABLE_LOGGING:
                                    self.logger.log_msg(f"[PRUNE] Removed outdated HQ model: {f_path}", level=logging.INFO)

                    # Save the updated global top-5 HQ networks list
                    self.global_hq_top5 = top5

                    faction.network.clear_memory()

                except Exception as e:
                    print(f"[HQ TRAIN ERROR] Failed to train HQ network for Faction {faction.id}: {e}")
                    traceback.print_exc()










    def print_episode_summary(self):
        """
        Print a summary at the end of an episode.
        """
        print(f"End of {self.mode} Episode {self.episode}")

        for faction in self.faction_manager.factions:
            print(f"Faction {faction.id} assigned tasks: {faction.assigned_tasks}\n\n")

        if utils_config.ENABLE_LOGGING:
            self.logger.log_msg(
                f"End of {self.mode} Episode {self.episode}",
                level=logging.INFO
            )

    def update_resources(self):
        """
        Update resource objects (e.g., AppleTree growth).
        """
        for res in self.resource_manager.resources:
            if isinstance(res, AppleTree):
                res.update()

        self.resource_counts = {
            "gold_lumps": sum(
                1 for res in self.resource_manager.resources if isinstance(res, GoldLump)
            ),
            "gold_quantity": sum(
                res.quantity for res in self.resource_manager.resources if isinstance(res, GoldLump)
            ),
            "apple_trees": sum(
                1 for res in self.resource_manager.resources if isinstance(res, AppleTree)
            ),
            "apple_quantity": sum(
                res.quantity for res in self.resource_manager.resources if isinstance(res, AppleTree)
            ),
        }









        

    def handle_event(self, event):
        """
        Handle events from the EventManager.
        """
        if event["type"] == "attack_animation":
            position = event["data"]["position"]
            duration = event["data"]["duration"]
            print(
                f"Game Manager/Handle_event - Playing attack animation at {position} for {duration} seconds.")
            self.renderer.play_attack_animation(position, duration)

        elif event["type"] == "dynamic_event":
            print("Dynamic event triggered.")
            self.event_manager.trigger_dynamic_event(
                max_trees=event["data"].get("max_trees", 10),
                max_gold_lumps=event["data"].get("max_gold_lumps", 5),
                health_penalty=event["data"].get("health_penalty", 10))









    def prepare_victory_scalar_history(self):
        """
        Transforms victory history into per-faction scalar data for plotting.
        """
        faction_count = len(self.faction_manager.factions)

        # Initialize series for each faction and a 'No Winner' line
        victory_series = {f"Faction {i} Wins": [] for i in range(faction_count)}
        victory_series["No Winner"] = []

        for wid in self.victory_history["winner_id"]:
            for i in range(faction_count):
                victory_series[f"Faction {i} Wins"].append(1 if wid == i else 0)
            victory_series["No Winner"].append(1 if wid == -1 else 0)

        return victory_series






    def process_pygame_events(self):
        """
        Handle pygame events like quitting and camera zooming.
        """
        if utils_config.HEADLESS_MODE or not pygame.display.get_init():
            return  # Skip processing events in headless mode or if display is closed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("[INFO] Window closed. Exiting game...")
                if utils_config.ENABLE_LOGGING:
                    self.logger.log_msg("Window closed - Exiting game.", level=logging.INFO)
                cleanup(QUIT=True)

            elif event.type == pygame.KEYDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.camera.zoom_around_mouse(True, mouse_x, mouse_y)
                elif event.key == pygame.K_MINUS:
                    self.camera.zoom_around_mouse(False, mouse_x, mouse_y)

    


















    
    def handle_victory(self, winner):
        print(f"Faction {winner.id} wins! Moving to next episode...")
        if utils_config.ENABLE_LOGGING:
            self.logger.log_msg(
                f"Faction {winner.id} wins! Ending episode early.",
                level=logging.INFO)


            











#    _   _                 _ _         ____
#   | | | | __ _ _ __   __| | | ___   / ___|__ _ _ __ ___   ___ _ __ __ _
#   | |_| |/ _` | '_ \ / _` | |/ _ \ | |   / _` | '_ ` _ \ / _ \ '__/ _` |
#   |  _  | (_| | | | | (_| | |  __/ | |__| (_| | | | | | |  __/ | | (_| |
#   |_| |_|\__,_|_| |_|\__,_|_|\___|  \____\__,_|_| |_| |_|\___|_|  \__,_|
#

    def handle_camera_movement(self, keys):
        """
        Handle camera movement and zoom.
        """
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera.move(-self.camera.speed, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera.move(self.camera.speed, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera.move(0, -self.camera.speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera.move(0, self.camera.speed)
       
















    

#    ____                              _        _____                 _
#   |  _ \ _   _ _ __   __ _ _ __ ___ (_) ___  | ____|_   _____ _ __ | |_
#   | | | | | | | '_ \ / _` | '_ ` _ \| |/ __| |  _| \ \ / / _ \ '_ \| __|
#   | |_| | |_| | | | | (_| | | | | | | | (__  | |___ \ V /  __/ | | | |_
#   |____/ \__, |_| |_|\__,_|_| |_| |_|_|\___| |_____| \_/ \___|_| |_|\__|
#          |___/


class EventManager:
    def __init__(
            self,
            resource_manager,
            faction_manager,
            agents,
            renderer,
            camera):
        """
        Initialise the EventManager.

        :param resource_manager: Resource manager to handle resource operations.
        :param faction_manager: Faction manager to handle faction operations.
        :param agents: List of agents in the game.
        :param renderer: Renderer for handling visual elements.
        """
        self.events = []
        self.resource_manager = resource_manager
        self.faction_manager = faction_manager
        self.agents = agents
        self.renderer = renderer  # Renderer to draw animations
        self.camera = camera  # Camera to position the animation

    def add_event(self, event_type, data=None):
        """
        Add an event to the queue.

        :param event_type: The type of event (e.g., 'attack_animation', 'dynamic_event').
        :param data: Additional data for the event (e.g., position or event-specific information).
        """
        self.events.append({"type": event_type, "data": data})

    def get_events(self):
        """
        Retrieve and clear the event queue.

        :return: A list of all pending events.
        """
        events = self.events[:]
        self.events.clear()
        return events

    def trigger_attack_animation(self, position, duration=500):
        """
        Trigger an attack animation at a given grid position.

        :param grid_position: Tuple (grid_x, grid_y) for the animation's grid coordinates.
        :param duration: Duration of the animation in milliseconds.
        """
        # Convert grid position to world position
        world_x = position[0]
        world_y = position[1]
        world_position = (world_x, world_y)

        # Convert world position to screen position using the camera
        screen_x, screen_y = self.camera.apply(world_position)

        # Debug with terrain grid context

        #print(f"[Event Manager] Attack Triggered @ Grid Pos: {position}, World Pos: {world_position}, Screen Pos: ({screen_x}, {screen_y})")

        # Trigger animation at calculated screen position
        self.add_event("attack_animation", {
                       "position": world_position, "duration": duration})

    def trigger_dynamic_event(
            self,
            max_trees=10,
            max_gold_lumps=5,
            health_penalty=10):
        """
        Trigger a dynamic event to redistribute resources and penalise health.
        """
        print("Triggering Dynamic Event: Redistributing resources and applying health penalty!")

        # Clean faction global states
        for faction in self.faction_manager.factions:
            faction.clean_global_state()

        # Generate new resources
        self.resource_manager.generate_resources(
            add_trees=max_trees, add_gold_lumps=max_gold_lumps)

        # Apply health penalty to all agents
        for agent in self.agents:
            agent.Health -= health_penalty
            if agent.Health <= 0:
                print(f"Agent {agent.agent_id} died due to dynamic event.")
