import traceback
import random
# Manages factions and their agents.
from utils_config import (
     FACTON_COUNT, 
    INITAL_GATHERER_COUNT, 
    INITAL_PEACEKEEPER_COUNT, 
    CELL_SIZE, 
    TaskState, 
    AgentID, 
    create_task, 
    DEF_AGENT_STATE_SIZE,
    STATE_FEATURES_MAP,
    NETWORK_TYPE_MAPPING)


from utils_helpers import (
    generate_random_color,
    find_closest_actor
)
import torch

import torch.nn as nn
import torch.optim as optim

from Agent_NeuralNetwork import device, PPOModel, DQNModel, HQ_Network 
from env_terrain import Terrain
from env_resources import AppleTree, GoldLump
from agent_communication import CommunicationSystem
 # Adjust this to match the correct file name

import logging
from utils_logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




logger = Logger(log_file="agent_factions.txt", log_level=logging.DEBUG)


class Faction():
    def __init__(self,game_manager, name, colour, id, resource_manager, agents, state_size, action_size, role_size, local_state_size, global_state_size, network_type="HQNetwork"):
        try:
            # Initialise Faction-specific attributes
            self.name = name
            self.color = colour
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
                "color": colour  # Match faction colour
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
                    logger.debug_log(f"[ERROR] Network failed to Initialise for faction {self.id} (Type: {network_type})", level=logging.ERROR)
                    raise AttributeError(f"[ERROR] Network failed to Initialise for faction {self.id}")
                else:
                    logger.debug_log(f"[DEBUG] Faction {self.id}: Successfully initialised {type(self.network).__name__}", level=logging.INFO)

            except Exception as e:
                logger.debug_log(f"[ERROR] Failed to Initialise network for faction {self.id}: {e}", level=logging.ERROR)
                logger.debug_log(traceback.format_exc(), level=logging.ERROR)

            
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
            print(f"{self.name} created with ID {self.id} and colour {self.color}")
        
        except Exception as e:
            print(f"An error occurred in Faction class __init__: {e}")
            import traceback
            traceback.print_exc()  # This will print the full traceback to the console

    def initialise_network(self, network_type, state_size, action_size, role_size, local_state_size, global_state_size):
        """
        Initialise the network model based on the selected network type.
        """
        try:
            logger.debug_log(f"[DEBUG] Faction {self.id}: Attempting to Initialise {network_type}...", level=logging.DEBUG)

            if network_type == "PPOModel":
                logger.debug_log(f"[DEBUG] Initialising PPOModel for faction {self.id} (state_size={state_size}, action_size={action_size})", level=logging.DEBUG)
                return PPOModel(state_size, action_size)
            elif network_type == "DQNModel":
                logger.debug_log(f"[DEBUG] Initialising DQNModel for faction {self.id} (state_size={state_size}, action_size={action_size})", level=logging.DEBUG)
                return DQNModel(state_size, action_size)
            elif network_type == "HQNetwork":
                logger.debug_log(f"[DEBUG] Initialising HQNetwork for faction {self.id} (state_size={state_size}, role_size={role_size}, local_state_size={local_state_size}, global_state_size={global_state_size}, action_size={action_size})", level=logging.DEBUG)
                return HQ_Network(state_size, role_size, local_state_size, global_state_size, action_size)
            else:
                raise ValueError(f"[ERROR] Unsupported network type: {network_type}")

        except Exception as e:
            logger.debug_log(f"[ERROR] Network initialisation failed for faction {self.id} (Type: {network_type}): {e}", level=logging.ERROR)
            
            logger.debug_log(traceback.format_exc(), level=logging.ERROR)
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

    def perform_faction_tasks(self, resource_manager, agents):
        """
        Delegate tasks to agents based on their roles and the faction's priorities.
        """
        for agent in self.agents:
            task = self.assigned_tasks.get(agent)
            if task:
                agent.behavior.execute_task(task, resource_manager, agents)
        logger.debug_log(f"Faction {self.id} assigned task {task} to agent {agent.role}.", level=logging.INFO)


    def receive_experience(self, experience):
        """
        Store experience for centralized training.
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
        logger.debug_log(f"[DEBUG] Faction {self.id} State: {self.global_state}")

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
                    logger.debug_log(f"Faction {self.id} updated threat ID {data['id']} to location {data['location']}.")
            else:
                self.global_state["threats"].append(data)
                logger.debug_log(f"Faction {self.id} added new threat: ID {data['id']} at {data['location']}.")

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
                    logger.debug_log(f"Faction {self.id} added resource: {resource_data}.")
            else:
                logger.debug_log(f"Invalid resource object in report for Faction {self.id}: {data}")

        else:
            logger.debug_log(f"Unknown report type '{report_type}' received by Faction {self.id}: {report}")




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

        logger.debug_log(f"[DEBUG] Cleaning global state for Faction {self.id} BEFORE reset: {self.global_state}")

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

        logger.debug_log(f"[DEBUG] Cleaned global state for Faction {self.id}: {self.global_state}")



    
#       _            _               _            _          _                                 _       
#      / \   ___ ___(_) __ _ _ __   | |_ __ _ ___| | _____  | |_ ___     __ _  __ _  ___ _ __ | |_ ___ 
#     / _ \ / __/ __| |/ _` | '_ \  | __/ _` / __| |/ / __| | __/ _ \   / _` |/ _` |/ _ \ '_ \| __/ __|
#    / ___ \\__ \__ \ | (_| | | | | | || (_| \__ \   <\__ \ | || (_) | | (_| | (_| |  __/ | | | |_\__ \
#   /_/   \_\___/___/_|\__, |_| |_|  \__\__,_|___/_|\_\___/  \__\___/   \__,_|\__, |\___|_| |_|\__|___/
#                      |___/                                                  |___/          

    def assign_high_level_tasks(self):
        """
        Use HQ's neural network to dynamically prioritize and assign tasks to agents.
        """
        try:
            raw_state = self.aggregate_faction_state()
            #print(f"[DEBUG] Raw Global State Before Tensor Conversion: {raw_state}")  #  Debug Print
            state_tensor = self.network.convert_state_to_tensor(raw_state)
            #print(f"[DEBUG] Final State Tensor Size After Conversion: {state_tensor.shape}")  #  Verify new shape
        except Exception as e:
            print(f"[ERROR] Failed to convert state to tensor for Faction {self.id}. Exception: {e}")
            raise

        role_tensor = torch.zeros(1, 5, dtype=torch.float32)
        local_state_tensor = torch.zeros(1, 5, dtype=torch.float32) 
        global_state_tensor = state_tensor.clone()
        logger.debug_log(f"Faction {self.id} - Tensor Shapes:\nRole={role_tensor.shape}\nState={state_tensor.shape}\n Local={local_state_tensor.shape}\n Global={global_state_tensor.shape}")
        #  Automatically update HQ Network if input size changes
        actual_size = state_tensor.shape[1] + role_tensor.shape[1] + local_state_tensor.shape[1] + global_state_tensor.shape[1]

        if self.network.fc1.in_features != actual_size:
            logger.debug_log(f"[INFO] Updating HQ_Network input size from {self.network.fc1.in_features} to {actual_size}")
            self.network.update_network(actual_size)

        logger.debug_log(f"[DEBUG] Actual HQ input size: {actual_size}", level=logging.INFO)

        with torch.no_grad():
            task_priorities, _ = self.network(state_tensor, role_tensor, local_state_tensor, global_state_tensor)

        #  Ensure task_priorities is correctly formatted
        if not isinstance(task_priorities, torch.Tensor):
            raise TypeError(f"[ERROR] Expected tensor from HQ network, but got {type(task_priorities)}")

        if task_priorities.shape[0] < 2:
            raise ValueError(f"[ERROR] Expected at least 2 outputs (threat & resource priority), but got {task_priorities.shape}")

        #  Extract priorities
        threat_priority = task_priorities[0].item()
        resource_priority = task_priorities[1].item()

        logger.debug_log(
            f"Faction {self.id} Task Priorities - Threat: {threat_priority}, Resource: {resource_priority}",
            level=logging.INFO
        )

        #Assign tasks to agents dynamically
        #Skip agents that already have an active task
        for agent in self.agents:
            if agent.current_task is not None:  
                continue  #  Skip assigning new tasks if an agent already has one

            task = self.assign_task(agent)
            if task:
                agent.current_task = task





    def assign_task(self, agent):
        """
        Assigns the most relevant task to an agent using a learned approach.
        Takes agent location into account and validates role-task compatibility.
        """
        role = getattr(agent, "role", None)
        if role not in ["gatherer", "peacekeeper"]:
            logger.debug_log(f"[WARN] Unknown role for agent {agent.agent_id}: {role}", level=logging.WARNING)
            return None

        # Filter unassigned threats/resources
        available_threats = [
                                t for t in self.global_state["threats"]
                                if t["id"] not in self.assigned_tasks
                                and isinstance(t["id"], AgentID)
                                and t["id"].faction_id != self.id
                                and t["id"] != agent.agent_id
                            ]

     
        available_resources = sorted( # Sort resources by distance from agent for task assignment
                                        [r for r in self.resource_manager.resources if not r.is_depleted()],
                                        key=lambda r: ((r.x - agent.x) ** 2 + (r.y - agent.y) ** 2) ** 0.5
                                    )

        # Use the agent's position for distance calculation
        nearest_threat = find_closest_actor(available_threats, entity_type="threat", requester=agent)
        nearest_resource = find_closest_actor(available_resources, entity_type="resource", requester=agent)

        # Build placeholder tensors for the HQ network
        role_tensor = torch.zeros(1, 5, dtype=torch.float32)
        local_state_tensor = torch.zeros(1, 5, dtype=torch.float32)
        global_state_tensor = torch.zeros(1, 5, dtype=torch.float32)
        task_tensor = self.network.convert_state_to_tensor(self.global_state)

        # Let the HQ network recommend task type
        with torch.no_grad():
            task_decision, _ = self.network(task_tensor, role_tensor, local_state_tensor, global_state_tensor)

        # Decide based on HQ prediction
        chosen_task = "eliminate" if task_decision[0].item() > task_decision[1].item() else "gather"

        # Role-task compatibility check
        if role == "gatherer" and chosen_task != "gather":
            logger.debug_log(f"Skipped assigning 'eliminate' task to gatherer {agent.agent_id}", level=logging.INFO)
            return None
        if role == "peacekeeper" and chosen_task != "eliminate":
            logger.debug_log(f"Skipped assigning 'gather' task to peacekeeper {agent.agent_id}", level=logging.INFO)
            return None

        #  Task assignment based on decision
        if chosen_task == "eliminate" and nearest_threat:
            task_id = f"Threat-{nearest_threat['id']}"
            target = {"position": nearest_threat["location"], "id": nearest_threat["id"]}
        elif chosen_task == "gather" and self.global_state["resources"]:
            # Find the closest HQ-known resource
            hq_known_resources = self.global_state["resources"]
            closest = find_closest_actor(hq_known_resources, entity_type="resource", requester=agent)

            if closest:
                # Match it to the actual AppleTree/GoldLump object
                target_obj = next(
                    (r for r in self.resource_manager.resources
                    if not r.is_depleted()
                    and (r.grid_x, r.grid_y) == closest["location"]),
                    None
                )

                if target_obj:
                    task_id = f"Resource-{(target_obj.grid_x, target_obj.grid_y)}"
                    target = {"position": (target_obj.grid_x, target_obj.grid_y)}
                    self.assigned_tasks[task_id] = agent
                    return create_task(self, task_type="gather", target=target, task_id=task_id)

        else:
            return None  # No valid targets

        # Register and return the task
        self.assigned_tasks[task_id] = agent
        return create_task(self, task_type=chosen_task, target=target, task_id=task_id)






#     ____      _            _       _         _            _                   _       _     _       
#    / ___|__ _| | ___ _   _| | __ _| |_ ___  | |_ __ _ ___| | __ __      _____(_) __ _| |__ | |_ ___ 
#   | |   / _` | |/ __| | | | |/ _` | __/ _ \ | __/ _` / __| |/ / \ \ /\ / / _ \ |/ _` | '_ \| __/ __|
#   | |__| (_| | | (__| |_| | | (_| | ||  __/ | || (_| \__ \   <   \ V  V /  __/ | (_| | | | | |_\__ \
#    \____\__,_|_|\___|\__,_|_|\__,_|\__\___|  \__\__,_|___/_|\_\   \_/\_/ \___|_|\__, |_| |_|\__|___/
#                                                                                 |___/  






    def calculate_resource_weight(self, agent, resource):
        """
        Calculate the weight of a resource task based on proximity and quantity.
        Lower weight is better (higher priority).
        """
        distance = ((resource.grid_x - agent.x // CELL_SIZE) ** 2 + (resource.grid_y - agent.y // CELL_SIZE) ** 2) ** 0.5
        weight = distance / (resource.quantity + 1)  # Higher quantity reduces weight
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






    def complete_task(self, task_location, agent, task_state):
        """
        Mark a task as completed and handle the outcome.
        :param task_location: The location of the completed task.
        :param agent: The agent that completed the task.
        :param task_state: The final state of the task (SUCCESS, FAILURE).
        """
        if task_location in self.assigned_tasks:
            logger.debug_log(f"Task at {task_location} completed with state: {task_state}.", level=logging.INFO)
            del self.assigned_tasks[task_location]

            if task_state == TaskState.SUCCESS:
                logger.debug_log(f"Agent {agent.role} successfully completed task at {task_location}.", level=logging.INFO)
                self.clean_global_state()  # Update global state after task completion
            elif task_state == TaskState.FAILURE:
                logger.debug_log(f"Agent {agent.role} failed to complete task at {task_location}. Reassigning task.", level=logging.WARNING)

    def update_tasks(self, agents):
        """
        Update tasks for all agents and handle their TaskState transitions.
        """
        for agent in agents:
            if agent.current_task_state == TaskState.SUCCESS or agent.current_task_state == TaskState.FAILURE:
                # Mark task as completed and clear for reassignment
                task = agent.current_task
                if task:
                    self.complete_task(task.get("target"), agent, agent.current_task_state)
                    agent.current_task = None
                    agent.update_task_state(TaskState.NONE)


    
    
    
    
    
    
    
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











    def choose_action(self):
        """
        HQ decides the best action based on faction state.
        """
        if self.is_under_attack():
            return "DEFEND_HQ"
        
        if self.gold_balance > 15 and len(self.agents) < 10:
            return "RECRUIT_TROOPS"

        if self.gold_balance < 10:
            return "EXPAND_ECONOMY"
        
        if self.should_attack_enemy_hq():
            return "ATTACK_ENEMY_HQ"

        return "SAVE_GOLD"




    def perform_action(self, action):
        """
        HQ executes the chosen action.
        """
        if action == "RECRUIT_TROOPS":
            self.recruit_troops()
        elif action == "DEFEND_HQ":
            self.defend_hq()
        elif action == "EXPAND_ECONOMY":
            self.expand_economy()
        elif action == "ATTACK_ENEMY_HQ":
            self.attack_enemy_hq()
        elif action == "SAVE_GOLD":
            print(f"Faction {self.id} is saving gold for future actions.")


    





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
        logger.debug_log("FactionManager initialised.", level=logging.INFO)

    def update(self, resource_manager, agents):
        """
        Update each faction by delegating tasks to its agents and leveraging the network for high-level decisions.
        """
        try:
            for faction in self.factions:
                if not isinstance(faction, Faction):
                    raise TypeError(f"Invalid faction type: {type(faction)}. Expected Faction instance.")

                if faction.network is None:
                    logger.debug_log(f"[ERROR] Faction {faction.id} has no valid network. Skipping update.", level=logging.ERROR)
                    continue  #  Skip processing this faction to prevent crashes

                faction.calculate_territory(resource_manager.terrain)
                global_state = faction.aggregate_faction_state()
                faction.assign_high_level_tasks()
                faction.update_tasks(agents)

                logger.debug_log(
                    f"Faction {faction.id} - Gold: {faction.gold_balance}, Food: {faction.food_balance}, "
                    f"Agent Count: {len(faction.agents)}, Threats: {len(faction.threats)}",
                    level=logging.INFO,
                )

        except Exception as e:
            logger.debug_log(f"[ERROR] An error occurred in FactionManager.update: {e}", level=logging.ERROR)
            import traceback
            logger.debug_log(traceback.format_exc(), level=logging.ERROR)




    def reset_factions(self, faction_count, resource_manager, agents, game_manager,
                   state_size=DEF_AGENT_STATE_SIZE, action_size=10, 
                   role_size=5, local_state_size=10, global_state_size=15, 
                   network_type="HQNetwork"):
        """
        Fully reset the list of factions and assign agents.
        """
        self.factions.clear()  #  Ensure previous factions are removed
        logger.debug_log("[INFO] Resetting factions...", level=logging.INFO)

        for i in range(faction_count):
            name = f"Faction {i + 1}"
            colour = generate_random_color()

            logger.debug_log(f"[DEBUG] Creating {name} with network type: {network_type}", level=logging.DEBUG)

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
                    logger.debug_log(f"[ERROR] Faction {new_faction.id} failed to Initialise network.", level=logging.ERROR)
                    raise RuntimeError(f"[ERROR] Faction {new_faction.id} failed to Initialise network.")

                self.factions.append(new_faction)
                logger.debug_log(f"[INFO] Successfully created {name} (ID: {new_faction.id})", level=logging.INFO)

            except Exception as e:
                logger.debug_log(f"[ERROR] Failed to create {name}: {e}", level=logging.ERROR)
                import traceback
                logger.debug_log(traceback.format_exc(), level=logging.ERROR)

        #  Assign agents to their factions after all factions are created
        for agent in agents:
            for faction in self.factions:
                if agent.faction.id == faction.id:
                    faction.agents.append(agent)

        #  Debug log to verify agents are assigned correctly
        for faction in self.factions:
            logger.debug_log(f"[DEBUG] {faction.name} Initialised with {len(faction.agents)} agents.", level=logging.INFO)


