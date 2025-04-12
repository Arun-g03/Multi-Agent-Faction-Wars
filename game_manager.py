"""
This file contains the GameManager class, which is responsible for managing the game state, including the game loop, game rules, and rendering.

"""
"""basic imports"""
import os
import sys
import pygame
import random
import math

from utils_helpers import profile_function
"""Importing codebase things to bring the componenets together and form the game"""
from env_terrain import Terrain
from env_resources import ResourceManager, AppleTree, GoldLump
from render_display import GameRenderer
from camera import Camera
from game_rules import check_victory
from agent_factions import Faction
from agent_faction_manager import FactionManager
from agent_base import Peacekeeper, Gatherer
from agent_communication import CommunicationSystem

"""Importing necessary constants"""
import utils_config

"""Logging  and profiling for performance and execution analysis"""
import logging
from utils_logger import Logger
import cProfile
import pstats
import io
import traceback
from utils_logger import TensorBoardLogger
"""Neural Network libraries and GPU enabling""" 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GameManager:
    def __init__(self, mode, save_dir="saved_models", max_steps_per_episode=utils_config.STEPS_PER_EPISODE, screen=None):
        # Initialise a logger specific to GameManager
        self.logger = Logger(log_file="game_manager_log.txt", log_level=logging.DEBUG)
        self.mode = None  # Default mode is None
        # Log initialisation
        if utils_config.ENABLE_LOGGING: self.logger.log_msg("GameManager initialised. Logs cleared for a new game.", level=logging.INFO)

        # Initialise the terrain
        self.terrain = Terrain()
        self.faction_manager = FactionManager()  # Initialise faction manager
        self.faction_manager.factions = []  # Ensure factions list is initialised
        self.agents = []  # List to store all agents
        self.screen = screen  # Initialise the screen
        self.mode = mode  # Set the mode
        self.episode = 1
        

        # Initialise the resource manager with the terrain
        self.resource_manager = ResourceManager(self.terrain)
        self.camera = Camera(utils_config.WORLD_WIDTH, utils_config.WORLD_HEIGHT, utils_config.SCREEN_WIDTH, utils_config.SCREEN_HEIGHT)
        try:
            # Initialise GameRenderer
            self.renderer = GameRenderer(
                self.screen,  # Pass the screen for rendering
                self.terrain,  # Pass terrain
                self.resource_manager.resources,  # Pass resources (ensure this is correctly initialised)
                self.faction_manager.factions,  # Pass factions
                self.agents,  # Pass agents
                self.camera  # Pass camera for any rendering requirements
            )
        except Exception as e:
            raise(f"Error Initialising GameRenderer: {e}")
            
        
        """Global variable to track the current step in the episode"""
        self.current_step = 0  # Initialise the current step
        
        """Current step in episode (tick counter)"""        
        self.save_dir = save_dir  # Ensure save_dir is a string (directory path)
        self.max_steps_per_episode = max_steps_per_episode
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
        if not os.path.exists(self.save_dir):  # Here `save_dir` should be a valid path string
            os.makedirs(self.save_dir)
        self.set_mode(mode)




        
    def set_mode(self, mode):
        """
        Set the mode of the game (training, evaluation).
        :param mode: The selected mode ('train' or 'evaluate').
        """
        if mode not in ['train', 'evaluate']:
            raise ValueError(f"Invalid mode: {mode}. Choose either 'train' or 'evaluate'.")
        
        self.mode = mode
        print(f"Game mode set to: {self.mode}")
    
    def Initialise_factions(self):
        try:
            # Example: Initialise factions with the specified network type (e.g., PPOModel)
            network_type = "PPOModel"  # This can be dynamic based on the needs
            for faction in self.faction_manager.factions:
                faction.initialise_network(network_type)
                print(f"initialised network for faction {faction.name} with type {network_type}")
        except Exception as e:
                print(f"An error occurred in Initialise_factions: {e}")
                traceback.print_exc()
    
    
#        _             
#    ___| |_ ___ _ __  
#   / __| __/ _ \ '_ \ 
#   \__ \ ||  __/ |_) |
#   |___/\__\___| .__/ 
#               |_|    

    """
    Executes a single game step with .
    """

    def step(self):
        try:
            # 🧭 Handle camera movement input
            keys = pygame.key.get_pressed()
            self.handle_camera_movement(keys)

            # 👥 Update each agent (movement, task execution, learning)
            for agent in self.agents[:]:
                try:
                    if agent.Health <= 0:
                        print(f"Agent {agent.role} from faction {agent.faction.id} has died.")
                        if agent in agent.faction.agents:
                            agent.faction.remove_agent(agent)
                        self.agents.remove(agent)
                        continue

                    hq_state = agent.faction.aggregate_faction_state()
                    agent.update(self.resource_manager, self.agents, hq_state)

                except Exception as e:
                    print(f"An error occurred for agent {agent.role}: {e}")
                    traceback.print_exc()

            # 🧠 Let agents share what they observed (threats, resources)
            for agent in self.agents:
                agent.observe(
                    all_agents=self.agents,
                    enemy_hq={"position": agent.faction.home_base["position"]},
                    resource_manager=self.resource_manager
                )

            # Update HQ logic, assign new tasks, and handle strategy switching
            for faction in self.faction_manager.factions:
                faction.update(self.resource_manager, self.agents, self.current_step)


            # 🔁 Increment the simulation step
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

            # Reset factions and agents
            self.agents_initialised = False  # Reset flag for agents initialisation
            self.faction_manager.reset_factions(utils_config.FACTON_COUNT, self.resource_manager, self.agents, self)
            self.agents.clear()
            self.Initialise_agents(mode=self.mode)
            

            # Reset communication system
            self.communication_system = CommunicationSystem(self.agents, self.faction_manager.factions)
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
            if mode not in ['train', 'evaluate']:
                mode = 'train'  # Default to 'train' if an invalid mode is provided
            print(f"Initialising game in {mode} mode...")

            # Store the mode
            self.mode = mode
            

            #Pass the Tensorboard instance to the Agent

            # Step 1: Generate Terrain
            self.terrain = Terrain()
            print("Terrain generated.")

            # Step 2: Initialise Resources
            print("Initialising resources...")
            self.resource_manager = ResourceManager(self.terrain)
            self.resource_manager.generate_resources(episode=self.episode)
            
            # Now set the `resources` attribute after generating them
            self.resources = self.resource_manager.resources  # Assign resources after generation
            print("Resources generated.")
            

            # Step 3: Initialise Factions
            print("Initialising factions...")
            self.faction_manager.reset_factions(utils_config.FACTON_COUNT, self.resource_manager, self.agents, self)
            print(f"{utils_config.FACTON_COUNT} factions initialised.")

            # Step 4: Initialise Agents and Place HQs
            self.agents.clear()  # Clear any existing agents
            self.Initialise_agents(mode)  # Pass the mode to agents
            print(f"{len(self.agents)} agents initialised across factions.")

            # Step 5: Initialise Communication System
            self.communication_system = CommunicationSystem(self.agents, self.faction_manager.factions)
            print("Communication system initialised.")

            # Step 6: Finalise initialisation
            self.current_step = 0  # Reset step counter for the new game
            self.episode = 1  # Reset episode counter for the new game
            global CurrEpisode
            CurrEpisode = self.episode          
            print(f"Game initialised in {mode} mode.")
        except:
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
            if all(math.dist(pos, existing_hq) >= minimum_distance for existing_hq in selected_positions):
                selected_positions.append(pos)
            if len(selected_positions) == num_factions:
                break

        if len(selected_positions) < num_factions:
            print("Failure: Not enough valid positions for all factions' HQs.")
            raise ValueError("Not enough valid positions for all factions' HQs.")

        # Step 3: Assign positions to factions
        for faction, position in zip(self.faction_manager.factions, selected_positions):
            # Convert grid position to pixel coordinates
            base_pixel_x, base_pixel_y = position[0] * utils_config.CELL_SIZE, position[1] * utils_config.CELL_SIZE
            faction.home_base["position"] = (base_pixel_x, base_pixel_y)

            # Mark the HQ position in the terrain grid
            self.terrain.grid[position[0]][position[1]]['occupied'] = True
            self.terrain.grid[position[0]][position[1]]['faction'] = faction.id
            self.terrain.grid[position[0]][position[1]]['resource_type'] = None

            print(f"Faction {faction.id} HQ placed at grid {position}, (pixel ({base_pixel_x}, {base_pixel_y}))")

            # Step 4: Spawn agents for the faction using spawn_agent
            faction_agents = []
            print(f"Spawning agents for faction {faction.id}...")

            # Define network_type for each agent (you can modify this logic to dynamically choose the network type)
            network_type = "PPOModel"  # Example network type (can also be dynamic based on the agent role or other factors)

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
                    network_type=network_type,  # Pass network type for the agent (e.g., PPOModel, DQNModel)
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
                    network_type=network_type  # Pass network type for the agent (e.g., PPOModel, DQNModel)
                )
                if agent:
                    faction_agents.append(agent)

            # Add the newly spawned agents to the faction and the global agent list
            print(f"Faction {faction.id} has {len(faction_agents)} agents.")

            #Clear and reassign to avoid shared references
            faction.agents = []  # Make sure each faction starts fresh
            faction.agents.extend(faction_agents)  # Assign its agents properly
            self.agents.extend(faction_agents)  # Keep global tracking

            # Print faction agent counts
            peacekeeper_count = sum(1 for agent in faction.agents if isinstance(agent, Peacekeeper))
            gatherer_count = sum(1 for agent in faction.agents if isinstance(agent, Gatherer))
            print(f"Faction {faction.id} has {peacekeeper_count} Peacekeepers and {gatherer_count} Gatherers.")


#    ____                                  _                    _       
#   / ___| _ __   __ ___      ___ __      / \   __ _  ___ _ __ | |_ ___ 
#   \___ \| '_ \ / _` \ \ /\ / / '_ \    / _ \ / _` |/ _ \ '_ \| __/ __|
#    ___) | |_) | (_| |\ V  V /| | | |  / ___ \ (_| |  __/ | | | |_\__ \
#   |____/| .__/ \__,_| \_/\_/ |_| |_| /_/   \_\__, |\___|_| |_|\__|___/
#         |_|                                  |___/                    

            
    def spawn_agent(self, base_x, base_y, faction, agent_class, state_size, role_actions, communication_system, event_manager, network_type="PPOModel", action_size=None):
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
            current_radius = spawn_radius + (attempts // 100) * 10  # Expand every 100 attempts
            current_radius = min(current_radius, max_radius)  # Cap the radius

            offset_x = random.randint(-current_radius, current_radius)
            offset_y = random.randint(-current_radius, current_radius)
            spawn_x = base_x + offset_x
            spawn_y = base_y + offset_y

            # Convert pixel coordinates to grid coordinates
            grid_x = spawn_x // utils_config.CELL_SIZE
            grid_y = spawn_y // utils_config.CELL_SIZE

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


    def get_agent_state(self, agent, hq_state):  #Returns the state of the agent
        return agent.get_state(self.resource_manager, self.agents, agent.faction, hq_state)







#    ____                 __                     _   __  
#   |  _ \ _   _ _ __    / /___ _ __   ___   ___| |__\ \ 
#   | |_) | | | | '_ \  | |/ _ \ '_ \ / _ \ / __| '_ \| |
#   |  _ <| |_| | | | | | |  __/ |_) | (_) | (__| | | | |
#   |_| \_\\__,_|_| |_| | |\___| .__/ \___/ \___|_| |_| |
#                        \_\   |_|                   /_/ 


    def run(self):
        """
        Main game loop handling both training and evaluation modes.
        """
        self.best_scores_per_role = {}  # role -> (best_reward, agent)

        try:
            print(f"Running game in {self.mode} mode...")
            running = True

            while running and (self.episode <= utils_config.EPISODES_LIMIT):
                self.reset()
                print("\033[92m" + f"Starting {self.mode} Episode {self.episode}" + "\033[0m")
                if utils_config.ENABLE_LOGGING: self.logger.log_msg(f"Starting {self.mode} Episode", level=logging.INFO)

                self.current_step = 0
                episode_reward = 0

                while self.current_step < self.max_steps_per_episode:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("[INFO] Window closed. Exiting game...")
                            if utils_config.ENABLE_LOGGING:self. logger.log_msg("Window closed - Exiting game.", level=logging.INFO)
                            pygame.quit()
                            sys.exit()

                    self.step()
                    self.current_step += 1

                    for res in self.resource_manager.resources:
                        if isinstance(res, AppleTree):
                            res.update()

                    resource_counts = {
                        "gold_lumps": sum(1 for res in self.resource_manager.resources if isinstance(res, GoldLump)),
                        "gold_quantity": sum(res.quantity for res in self.resource_manager.resources if isinstance(res, GoldLump)),
                        "apple_trees": sum(1 for res in self.resource_manager.resources if isinstance(res, AppleTree)),
                        "apple_quantity": sum(res.quantity for res in self.resource_manager.resources if isinstance(res, AppleTree)),
                    }

                    self.renderer.render(
                        self.camera,
                        self.terrain,
                        self.resource_manager.resources,
                        self.faction_manager.factions,
                        self.agents,
                        self.episode,
                        self.current_step,
                        resource_counts
                    )

                    for agent in self.agents:
                        if agent.ai.memory["rewards"]:
                            episode_reward += agent.ai.memory["rewards"][-1]

                    pygame.display.flip()

                    events = self.event_manager.get_events()
                    for event in events:
                        self.handle_event(event)

                    winner = check_victory(self.faction_manager.factions)
                    winner_id = winner.id if winner else None
                    if winner:
                        print(f"Faction {winner.id} wins! Moving to next episode...")
                        if utils_config.ENABLE_LOGGING: self.logger.log_msg(f"Faction {winner.id} wins! Ending episode early.", level=logging.INFO)
                        break

                # TensorBoard: Episode summary
                if utils_config.ENABLE_TENSORBOARD:
                    TensorBoardLogger().log_scalar("Episode/Steps_Taken", self.current_step, self.episode)

                for faction in self.faction_manager.factions:
                    # Aggregate rewards per role
                    role_rewards = {}
                    total_reward = 0

                    for agent in faction.agents:
                        rewards = agent.ai.memory.get("rewards", [])
                        if rewards:
                            last_reward = rewards[-1]
                            total_reward += last_reward

                            role = agent.role
                            role_rewards[role] = role_rewards.get(role, 0) + last_reward

                    # Log overall faction reward
                    if utils_config.ENABLE_TENSORBOARD:
                        TensorBoardLogger().log_scalar(f"Faction_{faction.id}/Total_Reward", total_reward, self.episode)
                        TensorBoardLogger().log_scalar(f"Faction_{faction.id}/Gold_Balance", faction.gold_balance, self.episode)
                        TensorBoardLogger().log_scalar(f"Faction_{faction.id}/Food_Balance", faction.food_balance, self.episode)
                        TensorBoardLogger().log_scalar(f"Faction_{faction.id}/Agents_Alive", len(faction.agents), self.episode)

                        # Log reward by role
                        for role, reward in role_rewards.items():
                            TensorBoardLogger().log_scalar(f"Faction_{faction.id}/Reward_{role}", reward, self.episode)
                            
                    print(f"Faction {faction.id} total reward: {total_reward}")
                    print(f"Faction {faction.id} gold balance: {faction.gold_balance}")
                    print(f"Faction {faction.id} food balance: {faction.food_balance}")
                    print(f"Faction {faction.id} agents alive: {len(faction.agents)}")
                    print(f"Faction {faction.id} reward by role: {role_rewards}")

                    
                        
                # 🔁 Train agents at the end of the episode
                if self.mode == "train":
                    if utils_config.ENABLE_LOGGING: self.logger.log_msg("[TRAINING] Starting PPO training at end of episode.", level=logging.INFO)
                    for agent in self.agents:
                        if agent.mode == "train" and len(agent.ai.memory["rewards"]) > 0:
                            if utils_config.ENABLE_LOGGING: self.logger.log_msg(f"[TRAIN CALL] Agent {agent.agent_id} training...", level=logging.INFO)
                            try:
                                agent.ai.train(mode="train")
                            except Exception as e:
                                print(f"Training failed for agent {agent.agent_id}: {e}")
                                traceback.print_exc()
                    # Save best-performing model per role
                    role_rewards = {}

                    # Collect rewards per role
                    for agent in self.agents:
                        if agent.ai.memory["rewards"]:
                            role = agent.role
                            total_reward = sum(agent.ai.memory["rewards"])
                            role_rewards.setdefault(role, []).append((agent, total_reward))

                    # Evaluate best agent for each role
                    for role, reward_list in role_rewards.items():
                        best_agent, best_reward = max(reward_list, key=lambda x: x[1])
                        prev_best, _ = self.best_scores_per_role.get(role, (None, float("-inf")))

                        if best_reward > _:
                            self.best_scores_per_role[role] = (best_agent, best_reward)
                            model_path = f"saved_models/Best_{role}_episode_{self.episode}.pth"
                            torch.save(best_agent.ai.state_dict(), model_path)
                            if utils_config.ENABLE_LOGGING: self.logger.log_msg(
                                f"[SAVE] New best {role} model saved at {model_path} with reward {best_reward:.2f}",
                                level=logging.INFO
                            )

                    for faction in self.faction_manager.factions:
                        is_winner = (faction.id == winner_id)
                        hq_reward = faction.compute_hq_reward(victory=is_winner)

                        if hasattr(faction.network, "update_memory_rewards"):
                            faction.network.update_memory_rewards(hq_reward)

                        if hasattr(faction.network, "train") and faction.network.hq_memory:
                            try:
                                if utils_config.ENABLE_LOGGING:
                                    self.logger.log_msg(f"[HQ TRAIN] Training strategy network for Faction {faction.id} with {len(faction.network.hq_memory)} samples.", level=logging.INFO)
                                faction.network.train(faction.network.hq_memory, faction.optimizer)
                                faction.network.hq_memory.clear_memory()
                            except Exception as e:
                                print(f"[HQ TRAIN ERROR] Failed to train HQ network for Faction {faction.id}: {e}")
                                traceback.print_exc()



                # Wrap up the episode
                print(f"End of {self.mode} Episode {self.episode}")
                if utils_config.ENABLE_LOGGING: self.logger.log_msg(f"End of {self.mode} Episode {self.episode}", level=logging.INFO)
                self.episode += 1

            if self.mode == "train" and self.episode > utils.EPISODES_LIMIT:
                print(f"Training completed after {utils.EPISODES_LIMIT} episodes")
                pygame.quit()
                sys.exit()

            return True

        except SystemExit:
            print("[INFO] Game closed successfully.")
        except Exception as e:
            print(f"An error occurred in {self.mode}: {e}")
            traceback.print_exc()
            pygame.quit()
            sys.exit()





    def handle_event(self, event):
        """
        Handle events from the EventManager.
        """
        if event["type"] == "attack_animation":
            position = event["data"]["position"]
            duration = event["data"]["duration"]
            print (f"Game Manager/Handle_event - Playing attack animation at {position} for {duration} seconds.")
            self.renderer.play_attack_animation(position, duration)

        elif event["type"] == "dynamic_event":
            print("Dynamic event triggered.")
            self.event_manager.trigger_dynamic_event(
                max_trees=event["data"].get("max_trees", 10),
                max_gold_lumps=event["data"].get("max_gold_lumps", 5),
                health_penalty=event["data"].get("health_penalty", 10)
            )

    



    
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
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:  # Zoom in
            self.camera.zoom_around_mouse(True, mouse_x, mouse_y)
        if keys[pygame.K_MINUS]:  # Zoom out
            self.camera.zoom_around_mouse(False, mouse_x, mouse_y)







#    ____                              _        _____                 _   
#   |  _ \ _   _ _ __   __ _ _ __ ___ (_) ___  | ____|_   _____ _ __ | |_ 
#   | | | | | | | '_ \ / _` | '_ ` _ \| |/ __| |  _| \ \ / / _ \ '_ \| __|
#   | |_| | |_| | | | | (_| | | | | | | | (__  | |___ \ V /  __/ | | | |_ 
#   |____/ \__, |_| |_|\__,_|_| |_| |_|_|\___| |_____| \_/ \___|_| |_|\__|
#          |___/                                                          


class EventManager:
    def __init__(self, resource_manager, faction_manager, agents, renderer, camera):
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
        
        print(f"[Event Manager] Grid Pos: {position}, World Pos: {world_position}, Screen Pos: ({screen_x}, {screen_y})")

        # Trigger animation at calculated screen position
        self.add_event("attack_animation", {"position": world_position, "duration": duration})


    def trigger_dynamic_event(self, max_trees=10, max_gold_lumps=5, health_penalty=10):
        """
        Trigger a dynamic event to redistribute resources and penalise health.
        """
        print("Triggering Dynamic Event: Redistributing resources and applying health penalty!")

        # Clean faction global states
        for faction in self.faction_manager.factions:
            faction.clean_global_state()

        # Generate new resources
        self.resource_manager.generate_resources(add_trees=max_trees, add_gold_lumps=max_gold_lumps)

        # Apply health penalty to all agents
        for agent in self.agents:
            agent.Health -= health_penalty
            if agent.Health <= 0:
                print(f"Agent {agent.agent_id} died due to dynamic event.")












