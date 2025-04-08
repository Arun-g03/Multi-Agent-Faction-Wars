import traceback
import random
from typing import Optional

# Manages factions and their agents.
from utils_config import (
     FACTON_COUNT, 
    INITAL_GATHERER_COUNT, 
    INITAL_PEACEKEEPER_COUNT, 
    CELL_SIZE, 
    TaskState, 
    AgentIDStruc, 
    create_task, 
    DEF_AGENT_STATE_SIZE,
    Agent_Interact_Range,
    STATE_FEATURES_MAP,
    NETWORK_TYPE_MAPPING,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    LOGGING_ENABLED)


from utils_helpers import (
    generate_random_colour,
    find_closest_actor
)
import torch

import torch.nn as nn
import torch.optim as optim

from Agent_NeuralNetwork import PPOModel, DQNModel, HQ_Network 
from env_terrain import Terrain
from env_resources import AppleTree, GoldLump
from agent_communication import CommunicationSystem
 # Adjust this to match the correct file name

import logging
from utils_logger import Logger






logger = Logger(log_file="agent_factions.txt", log_level=logging.DEBUG)


class Faction():
    def __init__(self,game_manager, name, colour, id, resource_manager, agents, state_size, action_size, role_size, local_state_size, global_state_size, network_type="HQNetwork"):
        try:
            # Initialise Faction-specific attributes
            self.name = name
            self.colour = colour
            self.id = id
            self.agents = agents  # List of agents
            self.resource_manager = resource_manager  # Reference to the resource manager
            self.gold_balance = 0
            self.food_balance = 0
            self.experience_buffer = []
            self.resources = []  # Initialise known resources
            self.threats = []  # Initialise known threats
            self.assigned_tasks = {}   # Track assigned tasks
            self.unvisited_cells = set()
            self.reports = []
            self.create_task = create_task
            # Initialise home_base with default values
            self.home_base = {
                "position": (0,0),  # To be set during initialisation
                "size": 50,  # Default size of the base
                "colour": colour  # Match faction colour
            }

            self.game_manager = game_manager
            
            self.global_state = {key: None for key in STATE_FEATURES_MAP["global_state"]}
            # Populate the initial global state
            self.global_state.update({
                "HQ_health": 100,  # Default HQ health
                "gold_balance": 0,  # Starting gold
                "food_balance": 0,  # Starting food
                "resource_count": 0,  # Total resources count
                "threat_count": 0,  # Total threats count
            })
            try:
                #  Fix: Only pass required arguments
                self.network = self.initialise_network(network_type, state_size, action_size, role_size, local_state_size, global_state_size)


                if self.network is None:
                    if LOGGING_ENABLED: logger.debug_log(f"[ERROR] Network failed to Initialise for faction {self.id} (Type: {network_type})", level=logging.ERROR)
                    raise AttributeError(f"[ERROR] Network failed to Initialise for faction {self.id}")
                else:
                    if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Faction {self.id}: Successfully initialised {type(self.network).__name__}", level=logging.INFO)

            except Exception as e:
                if LOGGING_ENABLED: logger.debug_log(f"[ERROR] Failed to Initialise network for faction {self.id}: {e}", level=logging.ERROR)
                if LOGGING_ENABLED: logger.debug_log(traceback.format_exc(), level=logging.ERROR)

            
            self.communication_system = CommunicationSystem(self.agents, [self])

            

            # Populate unvisited cells with land tiles from the terrain grid
            for x in range(len(self.resource_manager.terrain.grid)):
                for y in range(len(self.resource_manager.terrain.grid[0])):
                    cell = self.resource_manager.terrain.grid[x][y]
                    if cell['type'] == 'land' and not cell['occupied']:
                        self.unvisited_cells.add((x * CELL_SIZE, y * CELL_SIZE))  # Convert to pixel coordinates

            self.health = 100  # Initial health

            
            # Dynamically calculate the input size for the critic
            if len(self.agents) > 0:
                example_state = self.agents[0].get_state(self.resource_manager, self.agents, self)
                input_size = len(example_state)
            else:
                print("No agents available for dynamic input size calculation. Using fallback input size.")
                input_size = 14  # Default fallback size
            
            
            
            
            
            # Now self.hq_network and self.critic are properly initialised via network

            # Initialise the optimiser for the critic (if needed)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

            # Final confirmation print
            print(f"{self.name} created with ID {self.id} and colour {self.colour}")
        
        except Exception as e:
            print(f"An error occurred in Faction class __init__: {e}")
            import traceback
            traceback.print_exc()  # This will print the full traceback to the console

    def initialise_network(self, network_type, state_size, action_size, role_size, local_state_size, global_state_size):
        """
        Initialise the network model based on the selected network type.
        """
        try:
            if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Faction {self.id}: Attempting to Initialise {network_type}...", level=logging.DEBUG)

            if network_type == "PPOModel":
                if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Initialising PPOModel for faction {self.id} (state_size={state_size}, action_size={action_size})", level=logging.DEBUG)
                return PPOModel(state_size, action_size)
            elif network_type == "DQNModel":
                if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Initialising DQNModel for faction {self.id} (state_size={state_size}, action_size={action_size})", level=logging.DEBUG)
                return DQNModel(state_size, action_size)
            elif network_type == "HQNetwork":
                if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Initialising HQNetwork for faction {self.id} (state_size={state_size}, role_size={role_size}, local_state_size={local_state_size}, global_state_size={global_state_size}, action_size={action_size})", level=logging.DEBUG)
                return HQ_Network(state_size, role_size, local_state_size, global_state_size, action_size)
            else:
                raise ValueError(f"[ERROR] Unsupported network type: {network_type}")

        except Exception as e:
            if LOGGING_ENABLED: logger.debug_log(f"[ERROR] Network initialisation failed for faction {self.id} (Type: {network_type}): {e}", level=logging.ERROR)
            
            if LOGGING_ENABLED: logger.debug_log(traceback.format_exc(), level=logging.ERROR)
            return None  #  Prevent crashes by returning None instead of failing silently

             


    





    def add_agent(self, agent):
        self.agents.append(agent)

    def remove_agent(self, agent):
        """
        Safely remove an agent from the faction's agent list.
        """
        if agent in self.agents:
            self.agents.remove(agent)
            print(f"Agent {agent.role} removed from faction {self.id}.")
        else:
            print(f"Warning: Agent {agent.role} not found in faction {self.id}.")

    

    def receive_experience(self, experience):
        """
        Store experience for centralised training.
        """
        self.experience_buffer.append(experience)

    def clear_experience_buffer(self):
        """
        Clear the experience buffer after training.
        """
        self.experience_buffer = []


    
#    _             _         _   _             __            _   _                         _ _   _      
#   | |_ _ __ __ _(_)_ __   | |_| |__   ___   / _| __ _  ___| |_(_) ___  _ __     ___ _ __(_) |_(_) ___ 
#   | __| '__/ _` | | '_ \  | __| '_ \ / _ \ | |_ / _` |/ __| __| |/ _ \| '_ \   / __| '__| | __| |/ __|
#   | |_| | | (_| | | | | | | |_| | | |  __/ |  _| (_| | (__| |_| | (_) | | | | | (__| |  | | |_| | (__ 
#    \__|_|  \__,_|_|_| |_|  \__|_| |_|\___| |_|  \__,_|\___|\__|_|\___/|_| |_|  \___|_|  |_|\__|_|\___|
#                                                                                                       



  

#       _                                    _         _        __                            _   _                __                     
#      / \   __ _  __ _ _ __ ___  __ _  __ _| |_ ___  (_)_ __  / _| ___  _ __ _ __ ___   __ _| |_(_) ___  _ __    / _|_ __ ___  _ __ ___  
#     / _ \ / _` |/ _` | '__/ _ \/ _` |/ _` | __/ _ \ | | '_ \| |_ / _ \| '__| '_ ` _ \ / _` | __| |/ _ \| '_ \  | |_| '__/ _ \| '_ ` _ \ 
#    / ___ \ (_| | (_| | | |  __/ (_| | (_| | ||  __/ | | | | |  _| (_) | |  | | | | | | (_| | |_| | (_) | | | | |  _| | | (_) | | | | | |
#   /_/   \_\__, |\__, |_|  \___|\__, |\__,_|\__\___| |_|_| |_|_|  \___/|_|  |_| |_| |_|\__,_|\__|_|\___/|_| |_| |_| |_|  \___/|_| |_| |_|
#           |___/ |___/          |___/                                                                                                    
#                           _         _       _                        _       _           _       _        _                             
#     __ _  __ _  ___ _ __ | |_ ___  (_)_ __ | |_ ___     __ _    __ _| | ___ | |__   __ _| |  ___| |_ __ _| |_ ___                       
#    / _` |/ _` |/ _ \ '_ \| __/ __| | | '_ \| __/ _ \   / _` |  / _` | |/ _ \| '_ \ / _` | | / __| __/ _` | __/ _ \                      
#   | (_| | (_| |  __/ | | | |_\__ \ | | | | | || (_) | | (_| | | (_| | | (_) | |_) | (_| | | \__ \ || (_| | ||  __/                      
#    \__,_|\__, |\___|_| |_|\__|___/ |_|_| |_|\__\___/   \__,_|  \__, |_|\___/|_.__/ \__,_|_| |___/\__\__,_|\__\___|                     
#          |___/                                                 |___/          


    def aggregate_faction_state(self):
        """
        Aggregate faction-wide state, ensuring all required features exist.
        """

        #  Ensure required fields exist before processing
        required_keys = [
            "HQ_health", "gold_balance", "food_balance", "resource_count", "threat_count",
            "nearest_threat", "nearest_resource", "friendly_agent_count", "enemy_agent_count",
            "agent_density", "total_agents"
        ]

        for key in required_keys:
            if key not in self.global_state:
                self.global_state[key] = 0  # Default missing values to zero

        #  Ensure HQ position exists before using it
        if "position" not in self.home_base or self.home_base["position"] == (0, 0):
            print(f"[WARNING] HQ position for Faction {self.id} is missing! Assigning default location.")
            self.home_base["position"] = (random.randint(0, 100), random.randint(0, 100))  # Assign a random position

        hq_x, hq_y = self.home_base["position"]

        #  Ensure `nearest_threat` and `nearest_resource` are structured correctly
        self.global_state["nearest_threat"] = self.global_state.get("nearest_threat", {"location": (-1, -1)})
        self.global_state["nearest_resource"] = self.global_state.get("nearest_resource", {"location": (-1, -1)})

        #  Fetch agents correctly
        all_agents = self.game_manager.agents  #  Game-wide agents
        enemy_agents = [agent for agent in all_agents if agent.faction != self]

        #  Compute agent-related metrics
        self.global_state["friendly_agent_count"] = len(self.agents)  #  Use faction-level agents
        self.global_state["enemy_agent_count"] = len(enemy_agents)  #  Use game-wide agents
        self.global_state["total_agents"] = len(all_agents)  #  Use game-wide agents

        #  Compute agent density near HQ
        nearby_agents = [agent for agent in self.agents if ((agent.x - hq_x)**2 + (agent.y - hq_y)**2) ** 0.5 < 50]
        self.global_state["agent_density"] = len(nearby_agents)

        #  Ensure `agent_states` are properly formatted
        self.global_state["agent_states"] = [
            agent.get_state(self.resource_manager, self.agents, self, self.global_state)
            for agent in self.agents
        ]

        #  Debug Log to verify correct agent count
        if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Faction {self.id} State: {self.global_state}")

        return self.global_state










    

    def receive_report(self, report):
        """Process reports received from agents."""
        if "type" not in report or "data" not in report:
            logger.warning(f"Invalid report format received by Faction {self.id}: {report}")
            return

        report_type = report["type"]
        data = report["data"]

        if report_type == "threat":
            # Add or update the threat in the global state
            existing_threat = next(
                (threat for threat in self.global_state["threats"]
                 if threat["id"] == data["id"]),
                None
            )
            if existing_threat:
                if existing_threat["location"] != data["location"]:
                    existing_threat["location"] = data["location"]
                    if LOGGING_ENABLED: logger.debug_log(f"Faction {self.id} updated threat ID {data['id']} to location {data['location']}.")
            else:
                self.global_state["threats"].append(data)
                if LOGGING_ENABLED: logger.debug_log(f"Faction {self.id} added new threat: ID {data['id']} at {data['location']}.")

        elif report_type == "resource":
            # Extract relevant data from the resource object
            if hasattr(data, "grid_x") and hasattr(data, "grid_y") and hasattr(data, "__class__"):
                resource_data = {
                    "location": (data.grid_x, data.grid_y),
                    "type": data.__class__.__name__,
                }

                # Check if the resource already exists
                existing_resource = next(
                    (res for res in self.global_state["resources"]
                     if res["location"] == resource_data["location"]),
                    None
                )

                if existing_resource:
                    pass  # Do nothing if the resource already exists
                else:
                    self.global_state["resources"].append(resource_data)
                    if LOGGING_ENABLED: logger.debug_log(f"Faction {self.id} added resource: {resource_data}.")
            else:
                if LOGGING_ENABLED: logger.debug_log(f"Invalid resource object in report for Faction {self.id}: {data}")

        else:
            if LOGGING_ENABLED: logger.debug_log(f"Unknown report type '{report_type}' received by Faction {self.id}: {report}")




#    ____                 _     _            _        _         _                                         _                
#   |  _ \ _ __ _____   _(_) __| | ___   ___| |_ __ _| |_ ___  | |_ ___    _ __ ___  __ _ _   _  ___  ___| |_ ___ _ __ ___ 
#   | |_) | '__/ _ \ \ / / |/ _` |/ _ \ / __| __/ _` | __/ _ \ | __/ _ \  | '__/ _ \/ _` | | | |/ _ \/ __| __/ _ \ '__/ __|
#   |  __/| | | (_) \ V /| | (_| |  __/ \__ \ || (_| | ||  __/ | || (_) | | | |  __/ (_| | |_| |  __/\__ \ ||  __/ |  \__ \
#   |_|   |_|  \___/ \_/ |_|\__,_|\___| |___/\__\__,_|\__\___|  \__\___/  |_|  \___|\__, |\__,_|\___||___/\__\___|_|  |___/
#                                                                                      |_|  



    def provide_state(self):
        """
        Provide the faction's global state.
        """
        return self.global_state


#     ____ _                    _   _                  _       _           _       _        _       
#    / ___| | ___  __ _ _ __   | |_| |__   ___    __ _| | ___ | |__   __ _| |  ___| |_ __ _| |_ ___ 
#   | |   | |/ _ \/ _` | '_ \  | __| '_ \ / _ \  / _` | |/ _ \| '_ \ / _` | | / __| __/ _` | __/ _ \
#   | |___| |  __/ (_| | | | | | |_| | | |  __/ | (_| | | (_) | |_) | (_| | | \__ \ || (_| | ||  __/
#    \____|_|\___|\__,_|_| |_|  \__|_| |_|\___|  \__, |_|\___/|_.__/ \__,_|_| |___/\__\__,_|\__\___|
#                                                |___/                                       




    def clean_global_state(self):
        """
        Clean up outdated entries in the global state and ensure required features exist.
        """

        if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Cleaning global state for Faction {self.id} BEFORE reset: {self.global_state}")

        #  Clean up resources (remove depleted)
        if "resources" in self.global_state and isinstance(self.global_state["resources"], list):
            self.global_state["resources"] = [
                res for res in self.global_state["resources"] if not res.get("is_depleted", False)
            ]
            self.global_state["resource_count"] = len(self.global_state["resources"])  # Correctly update as integer
        else:
            self.global_state["resources"] = []  # Fallback if the key is missing or not a list
            self.global_state["resource_count"] = 0

        #  Clean up threats (remove inactive)
        if "threats" in self.global_state and isinstance(self.global_state["threats"], list):
            self.global_state["threats"] = [
                threat for threat in self.global_state["threats"] if threat.get("is_active", True)
            ]
        else:
            self.global_state["threats"] = []  # Fallback if the key is missing or not a list

        self.global_state["threat_count"] = len(self.global_state["threats"])

        #  Ensure `nearest_threat` and `nearest_resource` exist
        self.global_state["nearest_threat"] = self.global_state.get("nearest_threat", {"location": (-1, -1)})
        self.global_state["nearest_resource"] = self.global_state.get("nearest_resource", {"location": (-1, -1)})

        #  Recompute agent-related metrics after cleanup
        self.global_state["friendly_agent_count"] = len(self.agents)
        
        enemy_agents = [agent for agent in self.game_manager.agents if agent.faction != self]
        self.global_state["enemy_agent_count"] = len(enemy_agents)

        #  Compute agent density near HQ
        hq_x, hq_y = self.home_base["position"]
        nearby_agents = [agent for agent in self.agents if ((agent.x - hq_x)**2 + (agent.y - hq_y)**2) ** 0.5 < 50]
        self.global_state["agent_density"] = len(nearby_agents)

        #  Ensure total agent count is up to date
        self.global_state["total_agents"] = len(self.game_manager.agents)

        #  Ensure HQ health, gold, and food balances are maintained
        self.global_state["HQ_health"] = self.global_state.get("HQ_health", 100)
        self.global_state["gold_balance"] = self.global_state.get("gold_balance", 0)
        self.global_state["food_balance"] = self.global_state.get("food_balance", 0)

        if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Cleaned global state for Faction {self.id}: {self.global_state}")



    
#       _            _               _            _          _                                 _       
#      / \   ___ ___(_) __ _ _ __   | |_ __ _ ___| | _____  | |_ ___     __ _  __ _  ___ _ __ | |_ ___ 
#     / _ \ / __/ __| |/ _` | '_ \  | __/ _` / __| |/ / __| | __/ _ \   / _` |/ _` |/ _ \ '_ \| __/ __|
#    / ___ \\__ \__ \ | (_| | | | | | || (_| \__ \   <\__ \ | || (_) | | (_| | (_| |  __/ | | | |_\__ \
#   /_/   \_\___/___/_|\__, |_| |_|  \__\__,_|___/_|\_\___/  \__\___/   \__,_|\__, |\___|_| |_|\__|___/
#                      |___/                                                  |___/          
    def assign_high_level_tasks(self):
        """
        HQ chooses a strategic action, executes it, and assigns tasks to idle agents accordingly.
        """
        if LOGGING_ENABLED: logger.debug_log(f"[HQ] Faction {self.id} assigning high-level tasks...", level=logging.INFO)

        # 🧠 Step 1: Choose and execute strategy
        action = self.choose_action()
        self.perform_action(action)
        self.current_strategy = action  # Set for reference during agent-level decisions

        # 🧠 Step 2: Assign tasks to idle agents based on HQ strategy
        for agent in self.agents:
            if agent.current_task:
                continue  # Skip agents that already have a task

            task = self.assign_task(agent)
            if task:
                agent.current_task = task

                # ✅ Support multiple agents per task (teamwork-ready)
                task_id = task["id"]
                if task_id not in self.assigned_tasks:
                    self.assigned_tasks[task_id] = []
                self.assigned_tasks[task_id].append(agent.agent_id)


                # Optional debug print
                if LOGGING_ENABLED:
                    if LOGGING_ENABLED: logger.debug_log(
                        f"[TASK ASSIGNED] {agent.agent_id} => {task['type']} at {task['target'].get('position')}",
                        level=logging.INFO
                    )
                    if LOGGING_ENABLED: logger.debug_log(
                        f"[DEBUG] {agent.agent_id} has task: {agent.current_task}",
                        level=logging.DEBUG
                )


    #This is where tasks are created

    def assign_task(self, agent) -> Optional[dict]:
        """
        Decides the most relevant task for the agent based only on current global state.
        Supports multiple agents sharing tasks.
        """
        role = getattr(agent, "role", None)

        MAX_TASK_ASSIGNMENTS = {
            "eliminate": 2,
            "gather": 1,
            "explore": 1,
        }

        if role not in ["gatherer", "peacekeeper"]:
            if LOGGING_ENABLED: logger.debug_log(f"[WARN] Unknown role for agent {agent.agent_id}: {role}", level=logging.WARNING)
            return None

        # ========== GATHERER ==========
        if role == "gatherer":
            unclaimed_resources = [
                res for res in self.global_state.get("resources", [])
                if len(self.assigned_tasks.get(f"Resource-{res['location']}", [])) < MAX_TASK_ASSIGNMENTS["gather"]
            ]

            if not unclaimed_resources:
                if LOGGING_ENABLED: logger.debug_log(f"[NO TASK] No resources left for gatherer {agent.agent_id}", level=logging.INFO)
                return None

            # Sort by resource weight (distance/amount)
            nearest_resource = sorted(
                unclaimed_resources,
                key=lambda r: self.calculate_resource_weight(
                    agent,
                    next((res for res in self.resource_manager.resources if (res.grid_x, res.grid_y) == r["location"]), None)
                )
            )[0]

            task_id = f"Resource-{nearest_resource['location']}"
            target = {"position": nearest_resource["location"], "type": nearest_resource["type"]}
            return create_task(self, "gather", target, task_id)

        # ========== PEACEKEEPER ==========
        if role == "peacekeeper":
            available_threats = [
                threat for threat in self.global_state.get("threats", [])
                if isinstance(threat.get("id"), AgentIDStruc)
                and threat["id"].faction_id != self.id
                and threat["id"] != agent.agent_id
                and len(self.assigned_tasks.get(f"Threat-{threat['id']}", [])) < MAX_TASK_ASSIGNMENTS["eliminate"]
            ]

            if available_threats:
                nearest_threat = sorted(available_threats, key=lambda t: self.calculate_threat_weight(agent, t))[0]
                task_id = f"Threat-{nearest_threat['id']}"
                target = {
                    "position": nearest_threat["location"],
                    "id": nearest_threat["id"],
                    "type": "agent"
                }
                return create_task(self, "eliminate", target, task_id)

            # === Fallback: Explore unexplored global-state land
            terrain = self.resource_manager.terrain
            grid_width = len(terrain.grid)
            grid_height = len(terrain.grid[0])

            unexplored_cells = [
                (x, y)
                for x in range(grid_width)
                for y in range(grid_height)
                if terrain.grid[x][y]["faction"] != self.id
                and len(self.assigned_tasks.get(f"Explore-{x}-{y}", [])) < MAX_TASK_ASSIGNMENTS["explore"]
            ]

            if unexplored_cells:
                cell_x, cell_y = random.choice(unexplored_cells)
                task_id = f"Explore-{cell_x}-{cell_y}"
                target = {"position": (cell_x, cell_y)}
                if LOGGING_ENABLED: logger.debug_log(f"[EXPLORE] Agent {agent.agent_id} exploring ({cell_x}, {cell_y})", level=logging.INFO)
                return create_task(self, "explore", target, task_id)

            if LOGGING_ENABLED: logger.debug_log(f"[NO TASK] No threats or unexplored regions for agent {agent.agent_id}", level=logging.INFO)
            return None







#     ____      _            _       _         _            _                   _       _     _       
#    / ___|__ _| | ___ _   _| | __ _| |_ ___  | |_ __ _ ___| | __ __      _____(_) __ _| |__ | |_ ___ 
#   | |   / _` | |/ __| | | | |/ _` | __/ _ \ | __/ _` / __| |/ / \ \ /\ / / _ \ |/ _` | '_ \| __/ __|
#   | |__| (_| | | (__| |_| | | (_| | ||  __/ | || (_| \__ \   <   \ V  V /  __/ | (_| | | | | |_\__ \
#    \____\__,_|_|\___|\__,_|_|\__,_|\__\___|  \__\__,_|___/_|\_\   \_/\_/ \___|_|\__, |_| |_|\__|___/
#                                                                                 |___/  






    def calculate_resource_weight(self, agent, resource):
        if resource is None:
            return float("inf")
        distance = ((resource.grid_x - agent.x // CELL_SIZE) ** 2 +
                    (resource.grid_y - agent.y // CELL_SIZE) ** 2) ** 0.5
        weight = distance / (resource.quantity + 1)
        return weight


    
    def calculate_threat_weight(self, agent, threat):
        """
        Calculate the weight of a threat task based on proximity and threat type.
        Lower weight is better (higher priority).
        """
        distance = ((threat["location"][0] - agent.x) ** 2 + (threat["location"][1] - agent.y) ** 2) ** 0.5
        type_weight = 1  # Default weight for general threats
        if threat["type"] == "Faction HQ":
            type_weight = 0.5  # Higher priority for enemy HQs
        weight = distance * type_weight
        return weight





#         _               _      _   _            _            _                                                    _      _           _ 
#     ___| |__   ___  ___| | __ | |_| |__   ___  | |_ __ _ ___| | __ __      ____ _ ___    ___ ___  _ __ ___  _ __ | | ___| |_ ___  __| |
#    / __| '_ \ / _ \/ __| |/ / | __| '_ \ / _ \ | __/ _` / __| |/ / \ \ /\ / / _` / __|  / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \/ _` |
#   | (__| | | |  __/ (__|   <  | |_| | | |  __/ | || (_| \__ \   <   \ V  V / (_| \__ \ | (_| (_) | | | | | | |_) | |  __/ ||  __/ (_| |
#    \___|_| |_|\___|\___|_|\_\  \__|_| |_|\___|  \__\__,_|___/_|\_\   \_/\_/ \__,_|___/  \___\___/|_| |_| |_| .__/|_|\___|\__\___|\__,_|
#                                                                                                            |_|                        






    def complete_task(self, task_id, agent, task_state):
        """
        Clean up task assignment after an agent completes or fails a task.
        """
        if task_id in self.assigned_tasks:
            try:
                self.assigned_tasks[task_id].remove(agent.agent_id)
                if LOGGING_ENABLED: logger.debug_log(
                    f"[TASK COMPLETE] Agent {agent.agent_id} removed from task {task_id} ({task_state.name})",
                    level=logging.INFO
                )

                if not self.assigned_tasks[task_id]:  # All agents finished
                    del self.assigned_tasks[task_id]
                    if LOGGING_ENABLED: logger.debug_log(f"[TASK CLEARED] Task {task_id} fully cleared.", level=logging.DEBUG)

            except ValueError:
                if LOGGING_ENABLED: logger.debug_log(f"[WARN] Agent {agent.agent_id} not in task {task_id} list.", level=logging.WARNING)



    def update_tasks(self, agents):
        """
        Update tasks for all agents and handle their TaskState transitions.
        """
        for agent in agents:
            if agent.current_task_state == TaskState.SUCCESS or agent.current_task_state == TaskState.FAILURE:
                task = agent.current_task
                if task:
                    # If the task is complete, mark it and reassigned if needed
                    self.complete_task(task.get("id"), agent, agent.current_task_state)

                    
                    # Clear the task only if the task is successfully completed or failed
                    if task["state"] == TaskState.SUCCESS or task["state"] == TaskState.FAILURE:
                        agent.current_task = None
                        agent.update_task_state(TaskState.NONE)  # Reset the task state if the task is cleared

                else:
                    if LOGGING_ENABLED: logger.debug_log(f"No task assigned to agent {agent.agent_id}. Skipping.", level=logging.DEBUG)



        
        
    
    
    
    
    
    def calculate_territory(self, terrain):
        """Calculate the number of cells owned by this faction."""
        self.territory_count = sum(
            1 for row in terrain.grid for cell in row if cell['faction'] == self.id
        )

    #################################################################################
    #    _____ _____ ____ _____ ___ _   _  ____      _    ____  _____    _    
    #   |_   _| ____/ ___|_   _|_ _| \ | |/ ___|    / \  |  _ \| ____|  / \   
    #     | | |  _| \___ \ | |  | ||  \| | |  _    / _ \ | |_) |  _|   / _ \  
    #     | | | |___ ___) || |  | || |\  | |_| |  / ___ \|  _ <| |___ / ___ \ 
    #     |_| |_____|____/ |_| |___|_| \_|\____| /_/   \_\_| \_\_____/_/   \_\
    #                                                                         
    ################################################################################






    def count_role(self, role_name):
        return sum(1 for agent in self.agents if agent.role == role_name)





    def choose_action(self):
        """
        HQ decides the best action based on faction state.
        """
        if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} evaluating global state...", level=logging.INFO)

        # Immediate defense takes top priority
        if self.is_under_attack():
            if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} is under attack — choosing DEFEND_HQ.", level=logging.INFO)
            return "DEFEND_HQ"

        # Economy expansion if both gold and food are low
        if self.gold_balance < 10 and self.food_balance < 10:
            if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} has low resources — choosing EXPAND_ECONOMY.", level=logging.INFO)
            return "EXPAND_ECONOMY"

        # Recruit peacekeepers if under target count
        if self.gold_balance > 15 and self.count_role("peacekeeper") < INITAL_PEACEKEEPER_COUNT:
            if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} recruiting peacekeepers — choosing RECRUIT_PEACEKEEPERS.", level=logging.INFO)
            return "RECRUIT_PEACEKEEPERS"

        # Recruit gatherers if under target count
        if self.gold_balance > 15 and self.count_role("gatherer") < INITAL_GATHERER_COUNT:
            if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} recruiting gatherers — choosing RECRUIT_GATHERERS.", level=logging.INFO)
            return "RECRUIT_GATHERERS"

        # Offensive action if enemy HQ is exposed
        if self.should_attack_enemy_hq():
            if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} identifies enemy HQ vulnerability — choosing ATTACK_ENEMY_HQ.", level=logging.INFO)
            return "ATTACK_ENEMY_HQ"

        # Default fallback
        if LOGGING_ENABLED: logger.debug_log(f"[HQ STRATEGY] Faction {self.id} is defaulting to SAVE_GOLD.", level=logging.INFO)
        return "SAVE_GOLD"

    

    def perform_action(self, action):
        """
        HQ executes the chosen strategic action and coordinates agents accordingly.
        """
        self.current_strategy = action  # Store it for use in assign_task()

        if LOGGING_ENABLED: logger.debug_log(f"[HQ EXECUTE] Faction {self.id} performing strategy: {action}", level=logging.INFO)

        if action == "RECRUIT_PEACEKEEPERS":
            self.recruit_agent("peacekeeper")

        elif action == "RECRUIT_GATHERERS":
            self.recruit_agent("gatherer")

        elif action == "DEFEND_HQ":
            self.defend_hq()

        elif action == "ATTACK_ENEMY_HQ":
            self.attack_target = self.select_enemy_hq()


        elif action == "SAVE_GOLD":
            if LOGGING_ENABLED: logger.debug_log(f"[HQ EXECUTE] Faction {self.id} conserving resources.", level=logging.INFO)





    def recruit_agent(self, role: str):
        """
        Recruits an agent of the given role if resources allow.
        """
        cost = 10  # Example recruitment cost per agent

        if self.gold_balance < cost:
            if LOGGING_ENABLED: logger.debug_log(f"[HQ RECRUIT] Faction {self.id} lacks gold to recruit {role}.", level=logging.WARNING)
            return

        # Deduct cost and create agent
        self.gold_balance -= cost

        new_agent = self.create_agent(role)
        self.agents.append(new_agent)

        if LOGGING_ENABLED: logger.debug_log(
            f"[HQ RECRUIT] Faction {self.id} recruited new {role} — Gold: {self.gold_balance}, Total agents: {len(self.agents)}",
            level=logging.INFO
        )
    def create_agent(self, role: str):
        """
        Spawns a new agent instance of the given role at the faction's HQ.
        """
        from agent_base import Agent  # or wherever your base class is defined

        spawn_x, spawn_y = self.home_base["position"]

        agent_id = AgentIDStruc(faction_id=self.id, agent_id=len(self.agents) + 1)

        new_agent = Agent(
            agent_id=agent_id,
            x=spawn_x,
            y=spawn_y,
            faction=self,
            role=role,
            network_type=self.network_type  # PPOModel, etc.
        )

        if LOGGING_ENABLED: logger.debug_log(f"[SPAWN] Created {role} at ({spawn_x}, {spawn_y}) for Faction {self.id}.", level=logging.DEBUG)
        return new_agent
    
    def is_under_attack(self) -> bool:
        """
        Returns True if enemy agents are within field of view range of the faction HQ.
        """
        hq_x, hq_y = self.home_base["position"]
        detection_radius = Agent_Interact_Range * CELL_SIZE

        # Scan known threats in the global state
        for threat in self.global_state.get("threats", []):
            threat_id = threat.get("id")
            if not isinstance(threat_id, AgentIDStruc):
                continue

            # Ignore self threats (shouldn’t happen, but just in case)
            if threat_id.faction_id == self.id:
                continue

            tx, ty = threat.get("location", (-999, -999))
            dist_sq = (tx - hq_x) ** 2 + (ty - hq_y) ** 2

            if dist_sq <= detection_radius ** 2:
                if LOGGING_ENABLED: logger.debug_log(
                    f"[HQ THREAT] Faction {self.id} HQ is under threat from enemy agent {threat_id} at {tx, ty}",
                    level=logging.INFO
                )
                return True

        return False


    def select_enemy_hq(self):
        """
        Returns the location of the nearest enemy HQ, or None if not visible.
        """
        enemy_hqs = [
            t for t in self.global_state.get("threats", [])
            if t["type"] == "Faction HQ" and isinstance(t.get("id"), AgentIDStruc) and t["id"].faction_id != self.id
        ]

        if not enemy_hqs:
            return None

        # Pick the closest one to the faction HQ
        my_hq_x, my_hq_y = self.home_base["position"]

        nearest = sorted(enemy_hqs, key=lambda t: (t["location"][0] - my_hq_x) ** 2 + (t["location"][1] - my_hq_y) ** 2)[0]
        return nearest["location"]
    
    def defend_hq(self):
        """
        Marks the HQ for defense by setting a defensive rally point.
        """
        self.defensive_position = self.home_base["position"]
        if LOGGING_ENABLED: logger.debug_log(f"[HQ DEFENSE] Faction {self.id} rallying units to defend HQ at {self.defensive_position}.", level=logging.INFO)






#    _____ _    ____ _____ ___ ___  _   _   __  __    _    _   _    _    ____ _____ ____     ____ _        _    ____ ____  
#   |  ___/ \  / ___|_   _|_ _/ _ \| \ | | |  \/  |  / \  | \ | |  / \  / ___| ____|  _ \   / ___| |      / \  / ___/ ___| 
#   | |_ / _ \| |     | |  | | | | |  \| | | |\/| | / _ \ |  \| | / _ \| |  _|  _| | |_) | | |   | |     / _ \ \___ \___ \ 
#   |  _/ ___ \ |___  | |  | | |_| | |\  | | |  | |/ ___ \| |\  |/ ___ \ |_| | |___|  _ <  | |___| |___ / ___ \ ___) |__) |
#   |_|/_/   \_\____| |_| |___\___/|_| \_| |_|  |_/_/   \_\_| \_/_/   \_\____|_____|_| \_\  \____|_____/_/   \_\____/____/ 
#                                                                                                                          




class FactionManager:
    def __init__(self):
        self.factions = []
        self.faction_counter = 1  # Initialise a counter for unique IDs
        if LOGGING_ENABLED: logger.debug_log("FactionManager initialised.", level=logging.INFO)

    def update(self, resource_manager, agents):
        """
        Update each faction by delegating tasks to its agents and leveraging the network for high-level decisions.
        """
        try:
            #1 Check factions
            for faction in self.factions:
                if not isinstance(faction, Faction):
                    raise TypeError(f"Invalid faction type: {type(faction)}. Expected Faction instance.")

                #2 Validate network
                if faction.network is None:
                    if LOGGING_ENABLED: logger.debug_log(f"[ERROR] Faction {faction.id} has no valid network. Skipping update.", level=logging.ERROR)
                    continue  #  Skip processing this faction to prevent crashes

                #3 Update faction state
                faction.calculate_territory(resource_manager.terrain)
                global_state = faction.aggregate_faction_state()
                
                #4 Task management
                faction.assign_high_level_tasks()
                faction.update_tasks(agents)

                #5 Log faction status
                if LOGGING_ENABLED: logger.debug_log(
                    f"Faction {faction.id} - Gold: {faction.gold_balance}, Food: {faction.food_balance}, "
                    f"Agent Count: {len(faction.agents)}, Threats: {len(faction.threats)}",
                    level=logging.INFO,
                )
        except Exception as e:
            if LOGGING_ENABLED: logger.debug_log(f"[ERROR] An error occurred in FactionManager.update: {e}", level=logging.ERROR)
            import traceback
            if LOGGING_ENABLED: logger.debug_log(traceback.format_exc(), level=logging.ERROR)





    def reset_factions(self, faction_count, resource_manager, agents, game_manager,
                   state_size=DEF_AGENT_STATE_SIZE, action_size=10, 
                   role_size=5, local_state_size=10, global_state_size=15, 
                   network_type="HQNetwork"):
        """
        Fully reset the list of factions and assign agents.
        """
        self.factions.clear()  #  Ensure previous factions are removed
        if LOGGING_ENABLED: logger.debug_log("[INFO] Resetting factions...", level=logging.INFO)

        for i in range(faction_count):
            name = f"Faction {i + 1}"
            colour = generate_random_colour()

            if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] Creating {name} with network type: {network_type}", level=logging.DEBUG)

            try:
                new_faction = Faction(
                    name=name,
                    colour= colour,
                    id=i + 1,
                    resource_manager=resource_manager,
                    game_manager=game_manager,  #  Pass GameManager to Faction
                    agents=[],  #  Agents will be assigned after initialisation
                    state_size=state_size,
                    action_size=action_size,
                    role_size=role_size,
                    local_state_size=local_state_size,
                    global_state_size=global_state_size,
                    network_type=network_type
                )

                if new_faction.network is None:
                    if LOGGING_ENABLED: logger.debug_log(f"[ERROR] Faction {new_faction.id} failed to Initialise network.", level=logging.ERROR)
                    raise RuntimeError(f"[ERROR] Faction {new_faction.id} failed to Initialise network.")

                self.factions.append(new_faction)
                if LOGGING_ENABLED: logger.debug_log(f"[INFO] Successfully created {name} (ID: {new_faction.id})", level=logging.INFO)

            except Exception as e:
                if LOGGING_ENABLED: logger.debug_log(f"[ERROR] Failed to create {name}: {e}", level=logging.ERROR)
                import traceback
                if LOGGING_ENABLED: logger.debug_log(traceback.format_exc(), level=logging.ERROR)

        #  Assign agents to their factions after all factions are created
        for agent in agents:
            for faction in self.factions:
                if agent.faction.id == faction.id:
                    faction.agents.append(agent)

        #  Debug log to verify agents are assigned correctly
        for faction in self.factions:
            if LOGGING_ENABLED: logger.debug_log(f"[DEBUG] {faction.name} Initialised with {len(faction.agents)} agents.", level=logging.INFO)


