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
    LOGGING_ENABLED,
    HQ_STRATEGY_OPTIONS)


from utils_helpers import (
    generate_random_colour,
    find_closest_actor
)
import time
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
            self.current_strategy = None
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
                    if LOGGING_ENABLED: logger.log_msg(f"[ERROR] Network failed to Initialise for faction {self.id} (Type: {network_type})", level=logging.ERROR)
                    raise AttributeError(f"[ERROR] Network failed to Initialise for faction {self.id}")
                else:
                    if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Faction {self.id}: Successfully initialised {type(self.network).__name__}", level=logging.INFO)

            except Exception as e:
                if LOGGING_ENABLED: logger.log_msg(f"[ERROR] Failed to Initialise network for faction {self.id}: {e}", level=logging.ERROR)
                if LOGGING_ENABLED: logger.log_msg(traceback.format_exc(), level=logging.ERROR)

            
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
            if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Faction {self.id}: Attempting to Initialise {network_type}...", level=logging.DEBUG)

            if network_type == "PPOModel":
                if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Initialising PPOModel for faction {self.id} (state_size={state_size}, action_size={action_size})", level=logging.DEBUG)
                return PPOModel(state_size, action_size)
            elif network_type == "DQNModel":
                if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Initialising DQNModel for faction {self.id} (state_size={state_size}, action_size={action_size})", level=logging.DEBUG)
                return DQNModel(state_size, action_size)
            elif network_type == "HQNetwork":
                if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Initialising HQNetwork for faction {self.id} (state_size={state_size}, role_size={role_size}, local_state_size={local_state_size}, global_state_size={global_state_size}, action_size={action_size})", level=logging.DEBUG)
                return HQ_Network(state_size, role_size, local_state_size, global_state_size, action_size)
            else:
                raise ValueError(f"[ERROR] Unsupported network type: {network_type}")

        except Exception as e:
            if LOGGING_ENABLED: logger.log_msg(f"[ERROR] Network initialisation failed for faction {self.id} (Type: {network_type}): {e}", level=logging.ERROR)
            
            if LOGGING_ENABLED: logger.log_msg(traceback.format_exc(), level=logging.ERROR)
            return None  #  Prevent crashes by returning None instead of failing silently

             
    def update(self, resource_manager, agents):
        self.calculate_territory(resource_manager.terrain)
        self.aggregate_faction_state()
        self.choose_HQ_Strategy()
        self.assign_high_level_tasks()
        self.update_tasks(agents)
        self.choose_HQ_Strategy()
        

    





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


    


  

#       _                                    _                                   _          __                                              _       
#      / \   __ _  __ _ _ __ ___  __ _  __ _| |_ ___   _ __ ___ _ __   ___  _ __| |_ ___   / _|_ __ ___  _ __ ___     __ _  __ _  ___ _ __ | |_ ___ 
#     / _ \ / _` |/ _` | '__/ _ \/ _` |/ _` | __/ _ \ | '__/ _ \ '_ \ / _ \| '__| __/ __| | |_| '__/ _ \| '_ ` _ \   / _` |/ _` |/ _ \ '_ \| __/ __|
#    / ___ \ (_| | (_| | | |  __/ (_| | (_| | ||  __/ | | |  __/ |_) | (_) | |  | |_\__ \ |  _| | | (_) | | | | | | | (_| | (_| |  __/ | | | |_\__ \
#   /_/   \_\__, |\__, |_|  \___|\__, |\__,_|\__\___| |_|  \___| .__/ \___/|_|   \__|___/ |_| |_|  \___/|_| |_| |_|  \__,_|\__, |\___|_| |_|\__|___/
#    _      |___/ |___/       _  |___/_           _       _    |_| _                                                       |___/                    
#   (_)_ __ | |_ ___     __ _| | ___ | |__   __ _| |  ___| |_ __ _| |_ ___                                                                          
#   | | '_ \| __/ _ \   / _` | |/ _ \| '_ \ / _` | | / __| __/ _` | __/ _ \                                                                         
#   | | | | | || (_) | | (_| | | (_) | |_) | (_| | | \__ \ || (_| | ||  __/                                                                         
#   |_|_| |_|\__\___/   \__, |_|\___/|_.__/ \__,_|_| |___/\__\__,_|\__\___|                                                                         
#                       |___/                                                                                                                       


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
        #if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Faction {self.id} State: {self.global_state}")

        return self.global_state










    

    def receive_report(self, report):
        """Process reports received from agents."""
        if "type" not in report or "data" not in report:
            logger.warning(f"Invalid report format received by Faction {self.id}: {report}")
            return

        report_type = report["type"]
        data = report["data"]

        if report_type == "threat":
            threat_id = data.get("id")
            location = data.get("location")

            if "threats" not in self.global_state:
                self.global_state["threats"] = []

            # Use string comparison fallback if AgentIDStruc not hashable
            existing_threat = next(
                (t for t in self.global_state["threats"]
                if str(t.get("id")) == str(threat_id)),
                None
            )

            if existing_threat:
                if existing_threat["location"] != location:
                    existing_threat["location"] = location
                    if LOGGING_ENABLED:
                        logger.log_msg(f"Faction {self.id} updated threat ID {threat_id} to location {location}.")
            else:
                self.global_state["threats"].append(data)
                if LOGGING_ENABLED:
                    logger.log_msg(f"Faction {self.id} added new threat: {data['type']} ID {threat_id} at {location}.")


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
                    if LOGGING_ENABLED: logger.log_msg(f"Faction {self.id} added resource: {resource_data}.")
            else:
                if LOGGING_ENABLED: logger.log_msg(f"Invalid resource object in report for Faction {self.id}: {data}")

        else:
            if LOGGING_ENABLED: logger.log_msg(f"Unknown report type '{report_type}' received by Faction {self.id}: {report}")


    def provide_state(self):
        """
        Provide the faction's global state to a requester.
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

        if LOGGING_ENABLED: logger.log_msg(f"[DEBUG] Cleaning global state for Faction {self.id} BEFORE reset: {self.global_state}")

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

        if LOGGING_ENABLED:
            state = self.global_state

            resource_strs = [r.get("location", "?") for r in state.get("resources", [])]
            threat_strs = [f"{t.get('type', '?')}@{t.get('location', '?')}" for t in state.get("threats", [])]

            formatted_state = (
                f"\n[GLOBAL STATE] Faction {self.id}\n"
                f"  Resources     ({state.get('resource_count', 0)}): {resource_strs}\n"
                f"  Threats       ({state.get('threat_count', 0)}): {threat_strs}\n"
                f"  Nearest Resource: {state.get('nearest_resource', {}).get('location')}\n"
                f"  Nearest Threat : {state.get('nearest_threat', {}).get('location')}\n"
                f"  Friendly Agents: {state.get('friendly_agent_count', 0)}\n"
                f"  Enemy Agents   : {state.get('enemy_agent_count', 0)}\n"
                f"  Agent Density  : {state.get('agent_density', 0)} (near HQ)\n"
                f"  Total Agents   : {state.get('total_agents', 0)}\n"
                f"  HQ Health      : {state.get('HQ_health', 100)}\n"
                f"  Gold           : {state.get('gold_balance', 0)}\n"
                f"  Food           : {state.get('food_balance', 0)}\n"
            )
            logger.log_msg(formatted_state, level=logging.DEBUG)




        



    
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
        if LOGGING_ENABLED: logger.log_msg(f"[HQ] Faction {self.id} assigning high-level tasks...", level=logging.INFO)

        # 🧠 Step 1: Choose and execute FACTION strategy
        action = self.choose_HQ_Strategy()
        self.perform_HQ_Strategy(action)
        

        # 🧠 Step 2: Assign tasks to idle agents based on HQ strategy
        for agent in self.agents:
            if agent.current_task:
                continue  # Skip agents that already have a task

            task = self.assign_task(agent)
            if task:
                agent.current_task = task

                #Support multiple agents per task 
                task_id = task["id"]
                if task_id not in self.assigned_tasks:
                    self.assigned_tasks[task_id] = []
                self.assigned_tasks[task_id].append(agent.agent_id)


                # Optional debug print
                if LOGGING_ENABLED:
                    if LOGGING_ENABLED: logger.log_msg(
                        f"[TASK ASSIGNED] {agent.agent_id} => {task['type']} at {task['target'].get('position')}",
                        level=logging.INFO
                    )
                    if LOGGING_ENABLED: logger.log_msg(
                        f"[DEBUG] {agent.agent_id} has task: {agent.current_task}",
                        level=logging.DEBUG
                )


    #This is where tasks are created

    def assign_task(self, agent) -> Optional[dict]:
        role = getattr(agent, "role", None)
        strategy = self.current_strategy or "NO_PRIORITY"

        if role not in ["gatherer", "peacekeeper"]:
            if LOGGING_ENABLED:
                logger.log_msg(f"[WARN] Unknown role {agent.agent_id}: {role}", level=logging.WARNING)
            return None

        if role == "gatherer":
            current = agent.current_task
            current_type = current["type"] if current else None

            is_resource_task = current_type == "gather"

            # If already gathering, don't override
            if is_resource_task:
                return current

            # Otherwise clear task and assign new one
            if current:
                if LOGGING_ENABLED:
                    logger.log_msg(f"[REASSIGN] Gatherer {agent.agent_id} dropping task '{current_type}' for resource assignment.", level=logging.INFO)
                agent.current_task = None

            if strategy == "COLLECT_GOLD":
                return self.assign_mining_task(agent)

            elif strategy == "COLLECT_FOOD":
                return self.assign_forage_task(agent)

            # Fallback: pick best available
            options = list(filter(None, [
                self.assign_forage_task(agent),
                self.assign_mining_task(agent)
            ]))
            if options:
                return random.choice(options)

            return self.assign_explore_task(agent)  # Last fallback


        elif strategy == "ATTACK_THREATS":
            threats = [
                t for t in self.global_state.get("threats", [])
                if t["id"].faction_id != self.id
            ]
            if threats:
                threat = min(threats, key=lambda t: self.calculate_threat_weight(agent, t))

                # Check if the agent already has an eliminate task for this threat
                current = agent.current_task
                already_eliminating = current and current.get("type") == "eliminate" and current.get("target", {}).get("id") == threat["id"]

                if not already_eliminating:
                    if current:
                        if LOGGING_ENABLED:
                            logger.log_msg(f"[REASSIGN] Clearing task '{current['type']}' to reassign eliminate task.", level=logging.INFO)
                        agent.current_task = None  # Clear old task

                    return self.assign_eliminate_task(agent, threat["id"], threat["type"], threat["location"])

        
        logger.log_msg(f"Could not assign task to agent {agent.agent_id}(role={role}, strategy={strategy})", level=logging.WARNING)
        return None

    
    
    def assign_move_to_task(self, agent, position, label=None) -> Optional[dict]:
        """
        Assigns a simple move_to task to the given agent toward a position.
        """
        if not position or not isinstance(position, tuple) or len(position) != 2:
            if LOGGING_ENABLED:
                logger.log_msg(f"[MOVE_TO] Invalid target position for agent {agent.agent_id}: {position}", level=logging.WARNING)
            return None

        task_id = label or f"MoveTo-{position[0]}-{position[1]}-{agent.agent_id}"
        target = {"position": position}

        task = create_task(self, "move_to", target, task_id)

        if LOGGING_ENABLED:
            logger.log_msg(f"[MOVE_TO] Assigned agent {agent.agent_id} to move to {position} (task: {task_id})", level=logging.INFO)

        return task
    
    def assign_eliminate_task(self, agent, target_id, target_type, position) -> Optional[dict]:
        try:
            task_id = f"Eliminate-{target_id}"
            target = {
                "id": target_id,
                "type": target_type,
                "position": position
            }
            logger.log_msg(f"Created task: {task_id} for agent {agent.agent_id}", level=logging.INFO)
            return create_task(self, "eliminate", target, task_id)
        except Exception as e:
            logger.log_msg(f"Failed to create eliminate task: {e}", level=logging.ERROR)
            return None
        
    def assign_forage_task(self, agent) -> Optional[dict]:
        resources = [r for r in self.global_state.get("resources", []) if r["type"] == "AppleTree"]
        if not resources:
            return None
        nearest = min(resources, key=lambda r: self.calculate_resource_weight(agent, r))
        task_id = f"Forage-{nearest['location']}"
        target = {"position": nearest["location"], "type": "AppleTree"}
        logger.log_msg(f"Created task: {task_id} for agent {agent.agent_id}", level=logging.INFO)
        return create_task(self, "gather", target, task_id)

    def assign_mining_task(self, agent) -> Optional[dict]:
        resources = [r for r in self.global_state.get("resources", []) if r["type"] == "GoldLump"]
        if not resources:
            return None
        nearest = min(resources, key=lambda r: self.calculate_resource_weight(agent, r))
        task_id = f"Mine-{nearest['location']}"
        target = {"position": nearest["location"], "type": "GoldLump"}
        logger.log_msg (f"Created task: {task_id} for agent {agent.agent_id}", level=logging.INFO)
        return create_task(self, "gather", target, task_id)

    
    def assign_explore_task(self, agent) -> Optional[dict]:
        terrain = self.resource_manager.terrain
        unexplored_cells = [
            (x, y)
            for x in range(len(terrain.grid))
            for y in range(len(terrain.grid[0]))
            if terrain.grid[x][y]["faction"] != self.id
            and terrain.grid[x][y]["type"] == "land"
            and len(self.assigned_tasks.get(f"Explore-{x}-{y}", [])) < 2
        ]

        if unexplored_cells:
            cell_x, cell_y = random.choice(unexplored_cells)
            return self.assign_move_to_task(agent, (cell_x, cell_y), label=f"Explore-{cell_x}-{cell_y}")

        return None












#     ____      _            _       _         _            _                   _       _     _       
#    / ___|__ _| | ___ _   _| | __ _| |_ ___  | |_ __ _ ___| | __ __      _____(_) __ _| |__ | |_ ___ 
#   | |   / _` | |/ __| | | | |/ _` | __/ _ \ | __/ _` / __| |/ / \ \ /\ / / _ \ |/ _` | '_ \| __/ __|
#   | |__| (_| | | (__| |_| | | (_| | ||  __/ | || (_| \__ \   <   \ V  V /  __/ | (_| | | | | |_\__ \
#    \____\__,_|_|\___|\__,_|_|\__,_|\__\___|  \__\__,_|___/_|\_\   \_/\_/ \___|_|\__, |_| |_|\__|___/
#                                                                                 |___/  






    def calculate_resource_weight(self, agent, resource: dict) -> float:
        """
        Estimates priority weight for a resource (lower is better).
        Expects a dict with 'location' and 'quantity'.
        """
        if resource is None or "location" not in resource:
            return float("inf")

        rx, ry = resource["location"]
        ax, ay = agent.x // CELL_SIZE, agent.y // CELL_SIZE

        distance = ((rx - ax) ** 2 + (ry - ay) ** 2) ** 0.5
        quantity = resource.get("quantity", 0)

        weight = distance / (quantity + 1)
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
                if LOGGING_ENABLED: logger.log_msg(
                    f"[TASK COMPLETE] Agent {agent.agent_id} removed from task {task_id} ({task_state.name})",
                    level=logging.INFO
                )

                if not self.assigned_tasks[task_id]:  # All agents finished
                    del self.assigned_tasks[task_id]
                    if LOGGING_ENABLED: logger.log_msg(f"[TASK CLEARED] Task {task_id} fully cleared.", level=logging.DEBUG)

            except ValueError:
                if LOGGING_ENABLED: logger.log_msg(f"[WARN] Agent {agent.agent_id} not in task {task_id} list.", level=logging.WARNING)



    def update_tasks(self, agents):
        for agent in agents:
            # Skip agents from other factions
            if agent.faction != self:
                continue

            # If task completed, clear it
            if agent.current_task_state in [TaskState.SUCCESS, TaskState.FAILURE]:
                if agent.current_task:
                    self.complete_task(agent.current_task["id"], agent, agent.current_task_state)
                    agent.current_task = None
                    agent.update_task_state(TaskState.NONE)

            
            
        
    
    
    
    
    
    def calculate_territory(self, terrain):
        """Calculate the number of cells owned by this faction."""
        self.territory_count = sum(
            1 for row in terrain.grid for cell in row if cell['faction'] == self.id
        )

    #################################################################################


    #    _  _  ___     _______                 ___ _            _                 
    #   | || |/ _ \   / /_   _|__ __ _ _ __   / __| |_ _ _ __ _| |_ ___ __ _ _  _ 
    #   | __ | (_) | / /  | |/ -_) _` | '  \  \__ \  _| '_/ _` |  _/ -_) _` | || |
    #   |_||_|\__\_\/_/   |_|\___\__,_|_|_|_| |___/\__|_| \__,_|\__\___\__, |\_, |
    #                                                                  |___/ |__/ 

    ################################################################################






    def choose_HQ_Strategy(self):
        """
        HQ chooses a strategy using its neural network (or fallback),
        and stores it as the current_strategy.
        """
        if LOGGING_ENABLED:
            logger.log_msg(f"[HQ STRATEGY] Faction {self.id} requesting decision from HQ network...", level=logging.INFO)

        strategy = "NO_PRIORITY"  # Default

        if hasattr(self, "network") and self.network:
            predicted = self.network.predict_strategy(self.global_state)

            if predicted in HQ_STRATEGY_OPTIONS:
                strategy = predicted
                logger.log_msg(f"[HQ STRATEGY] Faction {self.id} network chose strategy: {strategy}", level=logging.INFO)
            else:
                logger.log_msg(f"[HQ STRATEGY] Invalid strategy returned: {predicted}. Defaulting to NO_PRIORITY.", level=logging.WARNING)
        else:
            logger.log_msg(f"[HQ STRATEGY] No HQ network found. Falling back to NO_PRIORITY.", level=logging.WARNING)

        self.current_strategy = strategy  #SET current strategy
        return strategy



    

    

    def perform_HQ_Strategy(self, action):
        """
        HQ executes the chosen strategic action if valid.
        If invalid, it re-evaluates strategy using the HQ network.
        """
        if LOGGING_ENABLED:
            logger.log_msg(f"[HQ EXECUTE] Faction {self.id} attempting strategy: {action}", level=logging.INFO)

        def retest_strategy():
            new_action = self.choose_HQ_Strategy()
            if new_action != action:
                logger.log_msg(f"[HQ RETEST] Strategy '{action}' invalid. Retesting and switching to '{new_action}'", level=logging.WARNING)
                self.perform_HQ_Strategy(new_action)

        # ========== STRATEGY: Recruit Peacekeeper ==========
        if action == "RECRUIT_PEACEKEEPER":
            if self.gold_balance < 10:
                logger.log_msg(f"[HQ EXECUTE] Not enough gold to recruit peacekeeper.", level=logging.WARNING)
                self.current_strategy = None
                return retest_strategy()
            self.recruit_agent("peacekeeper")

        # ========== STRATEGY: Recruit Gatherer ==========
        elif action == "RECRUIT_GATHERER":
            if self.gold_balance < 10:
                logger.log_msg(f"[HQ EXECUTE] Not enough gold to recruit gatherer.", level=logging.WARNING)
                self.current_strategy = None
                return retest_strategy()
            self.recruit_agent("gatherer")

        # ========== STRATEGY: Defend HQ ==========
        elif action == "DEFEND_HQ":
            hx, hy = self.home_base["position"]
            threats = self.global_state.get("threats", [])
            DEFENSE_RADIUS_SQ = 30 ** 2

            nearby_threat_found = False
            for threat in threats:
                if not isinstance(threat.get("id"), AgentIDStruc):
                    continue
                if threat["id"].faction_id == self.id:
                    continue  # Skip own faction

                tx, ty = threat["location"]
                dist_sq = (tx - hx) ** 2 + (ty - hy) ** 2
                if dist_sq <= DEFENSE_RADIUS_SQ:
                    nearby_threat_found = True
                    break

            if not nearby_threat_found:
                logger.log_msg(f"[HQ STRATEGY] No nearby threats to defend HQ.", level=logging.WARNING)
                return retest_strategy()

            # Strategy is valid
            self.defensive_position = self.home_base["position"]




        # ========== STRATEGY: Attack Threats ==========
        elif action == "ATTACK_THREATS":
            if self.global_state.get("threat_count", 0) == 0:
                logger.log_msg(f"[HQ EXECUTE] No threats to attack.", level=logging.WARNING)
                self.current_strategy = None
                return retest_strategy()
            

        # ========== STRATEGY: Collect Gold ==========
        elif action == "COLLECT_GOLD":
            gold_sources = [r for r in self.global_state.get("resources", []) if r["type"] == "gold"]
            if not gold_sources:
                logger.log_msg(f"[HQ EXECUTE] No gold resources available.", level=logging.WARNING)
                self.current_strategy = None
                return retest_strategy()
            

        # ========== STRATEGY: Collect Food ==========
        elif action == "COLLECT_FOOD":
            food_sources = [r for r in self.global_state.get("resources", []) if r["type"] == "food"]
            if not food_sources:
                logger.log_msg(f"[HQ EXECUTE] No food resources available.", level=logging.WARNING)
                self.current_strategy = None
                return retest_strategy()
            

        # ========== STRATEGY: No Priority ==========
        elif action == "NO_PRIORITY":
            if LOGGING_ENABLED:
                logger.log_msg(f"[HQ EXECUTE] Faction {self.id} conserving resources.", level=logging.INFO)
                time.sleep(5)
                return retest_strategy()
                

        # ========== Unknown Strategy ==========
        else:
            logger.log_msg(f"[HQ EXECUTE] Unknown strategy '{action}'. Retesting...", level=logging.WARNING)
            return retest_strategy()

        # ✅ Set current strategy only if it's valid and successful
        self.current_strategy = action






#    _  _  ___     _______                 ___ _            _                     _  _             _        _      _   _             
#   | || |/ _ \   / /_   _|__ __ _ _ __   / __| |_ _ _ __ _| |_ ___ __ _ _  _    /_\| |_ ___ _ __ (_)__    /_\  __| |_(_)___ _ _  ___
#   | __ | (_) | / /  | |/ -_) _` | '  \  \__ \  _| '_/ _` |  _/ -_) _` | || |  / _ \  _/ _ \ '  \| / _|  / _ \/ _|  _| / _ \ ' \(_-<
#   |_||_|\__\_\/_/   |_|\___\__,_|_|_|_| |___/\__|_| \__,_|\__\___\__, |\_, | /_/ \_\__\___/_|_|_|_\__| /_/ \_\__|\__|_\___/_||_/__/
#                                                                  |___/ |__/                                                        





    def recruit_agent(self, role: str):
        """
        Recruits an agent of the given role if resources allow.
        """
        cost = 10  # Example recruitment cost per agent

        if self.gold_balance < cost:
            if LOGGING_ENABLED: logger.log_msg(f"[HQ RECRUIT] Faction {self.id} lacks gold to recruit {role}.", level=logging.WARNING)
            return

        # Deduct cost and create agent
        self.gold_balance -= cost

        new_agent = self.create_agent(role)
        self.agents.append(new_agent)

        if LOGGING_ENABLED: logger.log_msg(
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

        if LOGGING_ENABLED: logger.log_msg(f"[SPAWN] Created {role} at ({spawn_x}, {spawn_y}) for Faction {self.id}.", level=logging.DEBUG)
        return new_agent
    
    def is_under_attack(self) -> bool:
        """
        Returns True if enemy agents are within field of view range of the faction HQ.
        """
        hq_x, hq_y = self.home_base["position"]
        detection_radius = Agent_Interact_Range

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
                if LOGGING_ENABLED: logger.log_msg(
                    f"[HQ THREAT] Faction {self.id} HQ is under threat from enemy agent {threat_id} at {tx, ty}",
                    level=logging.INFO
                )
                return True

        return False


    




























