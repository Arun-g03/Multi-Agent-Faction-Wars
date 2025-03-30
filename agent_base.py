"""
Base class for agents.
    Contains the basic attributes and methods for all agents and agent specific attributes.
"""


import traceback

import pygame
from Agent_NeuralNetwork import PPOModel, DQNModel  # Import the network models
from agent_behaviours import AgentBehaviour # Import behavior classes
from agent_factions import Faction
from utils_config import (
        SCREEN_WIDTH, 
        SCREEN_HEIGHT, 
        AGENT_SCALE_FACTOR, 
        CELL_SIZE, 
        Agent_field_of_view, 
        WORLD_HEIGHT, 
        WORLD_WIDTH, 
        TaskState,
        DEBUG_MODE,
        AgentIDStruc,
        create_task,
        TASK_TYPE_MAPPING,
        DEF_AGENT_STATE_SIZE,
        ROLE_ACTIONS_MAP,
        NETWORK_TYPE_MAPPING )

from utils_helpers import (
    find_closest_actor
    )
from env_terrain import Terrain
from env_resources import AppleTree, GoldLump
import torch
from torch.distributions import Categorical



import logging
from utils_logger import Logger



#       _                    _                             _ _     _                   _   _                 
#      / \   __ _  ___ _ __ | |_ ___   _ __   ___  ___ ___(_) |__ | | ___    __ _  ___| |_(_) ___  _ __  ___ 
#     / _ \ / _` |/ _ \ '_ \| __/ __| | '_ \ / _ \/ __/ __| | '_ \| |/ _ \  / _` |/ __| __| |/ _ \| '_ \/ __|
#    / ___ \ (_| |  __/ | | | |_\__ \ | |_) | (_) \__ \__ \ | |_) | |  __/ | (_| | (__| |_| | (_) | | | \__ \
#   /_/   \_\__, |\___|_| |_|\__|___/ | .__/ \___/|___/___/_|_.__/|_|\___|  \__,_|\___|\__|_|\___/|_| |_|___/
#           |___/                     |_|                                                                    

#ROLE__ACTIONS denotes the possible actions for each agent role.
#The actions are grouped into categories based on the agent's role.
#The model will be trained to predict the best action to take based on the agent's role and current state.


logger = Logger(log_file="agent_base_log.txt", log_level=logging.DEBUG)



""" Tint the sprite with the faction colour so its easy to identify the agent's faction. """
def tint_sprite(sprite, tint_color):
    """
    Apply a colour tint to a sprite.
    :param sprite: The base sprite image (Surface).
    :param tint_color: The colour to tint the sprite (RGB tuple).
    :return: The tinted sprite (Surface).
    """
    tinted_sprite = sprite.copy()  # Make a copy of the original sprite
    tinted_sprite.fill(tint_color, special_flags=pygame.BLEND_RGB_MULT)  # Apply colour tint
    return tinted_sprite



#       _                    _                                _          _               
#      / \   __ _  ___ _ __ | |_   _ __   __ _ _ __ ___ _ __ | |_    ___| | __ _ ___ ___ 
#     / _ \ / _` |/ _ \ '_ \| __| | '_ \ / _` | '__/ _ \ '_ \| __|  / __| |/ _` / __/ __|
#    / ___ \ (_| |  __/ | | | |_  | |_) | (_| | | |  __/ | | | |_  | (__| | (_| \__ \__ \
#   /_/   \_\__, |\___|_| |_|\__| | .__/ \__,_|_|  \___|_| |_|\__|  \___|_|\__,_|___/___/
#           |___/                 |_|                                                    




class BaseAgent:
    """Base Agent class for agents in the game. Handles agent-specific logic."""
    try:

        #Part of the init is for the agent itself and part is for the agent's network.
        def __init__(
                #Inputs
                self,
                x, #The x-coordinate of the agent's position.
                y,  #The y-coordinate of the agent's position. (Should/Could be converted to a tuple (x, y))
                role,#The role of the agent. Peacekeeper or Gatherer.
                faction, #The faction the agent belongs to.
                terrain,#Reference to the terrain object.
                resource_manager,#Reference to the resource manager object.
                role_actions,#The actions the agent can perform.
                agent_id,#Unique identifier for the agent.
                communication_system,#Reference to the communication system object.
                event_manager : object = None,#Placeholder for the event manager object.
                state_size = DEF_AGENT_STATE_SIZE,
                mode : str = "train",#Default mode is "train".
                network_type : str = "PPOModel"#Default network type is PPOModel.
            ):
            """
            Initialise the agent with a specific network model (e.g., PPO, DQN, etc.).
            """
            # Convert string network type to integer using NETWORK_TYPE_MAPPING
            network_type_int = NETWORK_TYPE_MAPPING.get(network_type, 1)  # Default to "none" if not found

            # Agent-specific initialisation
            self.x : float = x
            self.y : float = y
            self.role = role
            self.faction = faction
            self.terrain = terrain
            self.resource_manager = resource_manager
            self.role_actions = role_actions[role]
            self.agent_id = AgentIDStruc(faction.id, agent_id)
            
            # Ensure a valid state size is always used
            self.state_size = state_size if state_size is not None else DEF_AGENT_STATE_SIZE

            self.communication_system = communication_system
            self.event_manager = event_manager
            self.mode = mode
            self.Health = 100
            self.speed = 1
            self.local_view = Agent_field_of_view
            self.update_detection_range()

            # Initialise the network (model) first
            self.ai = self.initialise_network(
                network_type_int=network_type_int,
                state_size=self.state_size,
                action_size=len(self.role_actions),
                AgentID=self.agent_id
            )


            # Then initialise the behavior object(AgentBehaviour)
            self.behavior = AgentBehaviour(
                agent=self,
                state_size=state_size,
                action_size=len(self.role_actions),
                role_actions=role_actions,
                event_manager=event_manager
            )
            
            self.current_task = None ## Initialise the current task with None
            self.current_task_state = TaskState.NONE ## Initialise the current task state with TaskState.NONE
            
            # Initialise other components
            self.experience_buffer = []  # Temporary experience buffer (for training, works like memory)
            self.create_task = create_task #Initiaises the structure for the task creation/handling



    except Exception as e:
        raise(f"Error in BaseAgent initialisation: {e}")
            
    def initialise_network(self, network_type_int, state_size, action_size, AgentID):
        """
        Initialise the network model based on the selected network type.
        """
        if network_type_int == NETWORK_TYPE_MAPPING["PPOModel"]:
            print(
                "\033[93m"
                + f"Initialising PPOModel with state_size={state_size}, action_size={action_size} for AgentID={AgentID}"
                + "\033[0m"
            )
            return PPOModel(
                AgentID=AgentID,
                state_size=state_size,
                action_size=action_size
            )

        elif network_type_int == NETWORK_TYPE_MAPPING["DQNModel"]:
            print(
                "\033[93m"
                + f"Initialising DQNModel with state_size={state_size}, action_size={action_size} for AgentID={AgentID}"
                + "\033[0m"
            )
            return DQNModel(
                state_size=state_size,
                action_size=action_size
            )

        else:
            raise ValueError(f"Unsupported network type: {network_type_int}")

            
    


    def can_move_to(self, new_x, new_y):
        # Calculate the grid coordinates based on position
        grid_x = new_x // CELL_SIZE
        grid_y = new_y // CELL_SIZE

        # Check bounds
        if 0 <= grid_x < len(self.terrain.grid) and 0 <= grid_y < len(self.terrain.grid[0]):
            return self.terrain.grid[grid_x][grid_y]['type'] == 'land'
        logger.debug_log(f"Attempted to move to invalid position: ({new_x}, {new_y})", level=logging.ERROR)
        return False

    def move(self, dx, dy):
        # Calculate new potential position
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        # Convert pixel coordinates to grid coordinates
        grid_x = new_x // CELL_SIZE
        grid_y = new_y // CELL_SIZE

        # Check if the new position is valid and on land
        if self.can_move_to(new_x, new_y):
            # Mark the current cell as the agent's faction territory
            current_grid_x = self.x // CELL_SIZE
            current_grid_y = self.y // CELL_SIZE
            self.terrain.grid[current_grid_x][current_grid_y]['faction'] = self.faction.id

            # Update the agent's position
            self.x = new_x
            self.y = new_y
            logger.debug_log(f"Agent {self.role} moved to ({new_x}, {new_y})")

            # Update position history
            if not hasattr(self, "recent_positions"):
                self.recent_positions = []  # Initialise history if not present
            self.recent_positions.append((self.x, self.y))
            if len(self.recent_positions) > 10:  # Increase history size for better detection
                self.recent_positions.pop(0)

            # Check for being stuck
            unique_positions = len(set(self.recent_positions))
            if unique_positions <= 2:  # Threshold for "stuck" detection (more sensitive)
                logger.debug_log(
                    f"Agent {self.role} is likely stuck. "
                    f"Recent positions: {self.recent_positions}. Penalising.",
                    level=logging.WARNING
                )
                self.Health -= 2  # Example penalty (health loss)
                logger.debug_log(f"Agent {self.agent_id}{self.role} has been penalised for being stuck.")
                if self.Health <= 0:
                    print(f"Agent {self.role} has died from being stuck.")
            # Mark the new cell as the agent's faction territory
            grid_x = new_x // CELL_SIZE
            grid_y = new_y // CELL_SIZE
            self.terrain.grid[grid_x][grid_y]['faction'] = self.faction.id
        else:
            logger.debug_log(f"Agent {self.role} attempted invalid move to ({new_x}, {new_y}).", level=logging.ERROR)



    
    def is_near(self, target, threshold=3):
        """
        Check if the agent is near a specified target within a given threshold.
        :param target: The target object with `x` and `y` attributes, or a tuple (x, y).
        :param threshold: The maximum distance considered "near". Default is 10.
        :return: True if the agent is within the threshold distance of the target, False otherwise.
        """
        if isinstance(target, tuple):
            target_x, target_y = target
        else:
            target_x, target_y = target.x, target.y

        distance = ((self.x - target_x) ** 2 + (self.y - target_y) ** 2) ** 0.5
        return distance <= threshold





#    ____            __                        _   _            _            _       __         _                   _   _           __      
#   |  _ \ ___ _ __ / _| ___  _ __ _ __ ___   | |_| |__   ___  | |_ __ _ ___| | __  / / __ ___ | | ___    __ _  ___| |_(_) ___  _ __\ \     
#   | |_) / _ \ '__| |_ / _ \| '__| '_ ` _ \  | __| '_ \ / _ \ | __/ _` / __| |/ / | | '__/ _ \| |/ _ \  / _` |/ __| __| |/ _ \| '_ \| |    
#   |  __/  __/ |  |  _| (_) | |  | | | | | | | |_| | | |  __/ | || (_| \__ \   <  | | | | (_) | |  __/ | (_| | (__| |_| | (_) | | | | |    
#   |_|   \___|_|  |_|  \___/|_|  |_| |_| |_|  \__|_| |_|\___|  \__\__,_|___/_|\_\ | |_|  \___/|_|\___|  \__,_|\___|\__|_|\___/|_| |_| |    
#                                                                                   \_\                                             /_/     
#         _                            _             _   _                                       _              _                      _    
#     ___| |__   ___  ___  ___ _ __   | |__  _   _  | |_| |__   ___   _ __   ___ _   _ _ __ __ _| |  _ __   ___| |___      _____  _ __| | __
#    / __| '_ \ / _ \/ __|/ _ \ '_ \  | '_ \| | | | | __| '_ \ / _ \ | '_ \ / _ \ | | | '__/ _` | | | '_ \ / _ \ __\ \ /\ / / _ \| '__| |/ /
#   | (__| | | | (_) \__ \  __/ | | | | |_) | |_| | | |_| | | |  __/ | | | |  __/ |_| | | | (_| | | | | | |  __/ |_ \ V  V / (_) | |  |   < 
#    \___|_| |_|\___/|___/\___|_| |_| |_.__/ \__, |  \__|_| |_|\___| |_| |_|\___|\__,_|_|  \__,_|_| |_| |_|\___|\__| \_/\_/ \___/|_|  |_|\_\
#                                            |___/                                                                                          

    def perform_task(self, state, resource_manager, agents):
        """
        Execute the current task using the behavior component.
        """
        return self.behavior.perform_task(state, resource_manager, agents)





    def update_task_state(self, task_state):
        """
        Update the current task state for the agent.
        :param task_state: The new state of the current task (TaskState).
        """
        self.current_task_state = task_state
        logger.debug_log(f"{self.role} task state updated to {task_state}.", level=logging.DEBUG)

    
    def update(self, resource_manager, agents, hq_state):
        """
        Update the agent's state. This includes:
        - Performing assigned tasks.
        - Observing the environment.
        - Reporting experiences to the faction.
        """
        try:
            # Log the current task before performing it
            logger.debug_log(
                f"[TASK EXECUTION] Agent {self.agent_id} executing task: {self.current_task}",
                level=logging.DEBUG
            )

            # Retrieve the agent's current state based on HQ state
            state = self.get_state(resource_manager, agents, self.faction, hq_state)
            if state is None:
                raise RuntimeError(f"[CRITICAL] Agent {self.agent_id} received a None state from get_state")

            logger.debug_log(f"{self.role} state retrieved: {state}", level=logging.DEBUG)

            # Execute the current task or decide on a new action
            reward, task_state = self.perform_task(state, resource_manager, agents)
            self.update_task_state(task_state)  # Update the task state based on execution

            # Observe the environment and report findings to the faction
            self.observe(agents, {"position": self.faction.home_base["position"]}, resource_manager)

            # Log the task state and reward for centralized learning
            if task_state in [TaskState.SUCCESS, TaskState.FAILURE]:
                next_state = self.get_state(resource_manager, agents, self.faction, hq_state)
                done = task_state in [TaskState.SUCCESS, TaskState.FAILURE]

                # Report the agent's experience to the HQ
                self.report_experience_to_hq(state, self.current_task, reward, next_state, done)

            # Handle health-related conditions
            if self.Health <= 0:
                logger.debug_log(f"{self.role} has died and will be removed from the game.", level=logging.WARNING)

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"An error occurred while updating the agent: {e}")







#     ___  _                               _                 _                   _                                      _   
#    / _ \| |__  ___  ___ _ ____   _____  | | ___   ___ __ _| |   ___ _ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_ 
#   | | | | '_ \/ __|/ _ \ '__\ \ / / _ \ | |/ _ \ / __/ _` | |  / _ \ '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
#   | |_| | |_) \__ \  __/ |   \ V /  __/ | | (_) | (_| (_| | | |  __/ | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_ 
#    \___/|_.__/|___/\___|_|    \_/ \___| |_|\___/ \___\__,_|_|  \___|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|
#                                                                                                                           
#                    _                                                                                                      
#     __ _ _ __   __| |                                                                                                     
#    / _` | '_ \ / _` |                                                                                                     
#   | (_| | | | | (_| |                                                                                                     
#    \__,_|_| |_|\__,_|                                                                                                     
#                                                                                                                           
#                               _      __ _           _ _                                                                   
#     _ __ ___ _ __   ___  _ __| |_   / _(_)_ __   __| (_)_ __   __ _ ___                                                   
#    | '__/ _ \ '_ \ / _ \| '__| __| | |_| | '_ \ / _` | | '_ \ / _` / __|                                                  
#    | | |  __/ |_) | (_) | |  | |_  |  _| | | | | (_| | | | | | (_| \__ \                                                  
#    |_|  \___| .__/ \___/|_|   \__| |_| |_|_| |_|\__,_|_|_| |_|\__, |___/                                                  
#             |_|                                               |___/                                                       



    def observe(self, all_agents, enemy_hq, resource_manager):
        """Observe and report threats and resources."""
        # Detect threats
        observed_threats = self.detect_threats(all_agents, enemy_hq)

        # Log all observed threats
        if observed_threats:
            logger.debug_log(f"Agent {self.agent_id} observed threats: {observed_threats}", level=logging.DEBUG)        

        for threat in observed_threats:
            # Ensure the threat is from a different faction
            if isinstance(threat["id"], AgentIDStruc):
                if threat["id"].faction_id == self.faction.id:
                    continue  # Skip friendly threats
            else:
                if threat["faction"] == self.faction.id:
                    continue  # Skip friendly threats

            # Report threat to HQ using the communication system
            if self.communication_system:
                self.communication_system.agent_to_hq(self, {"type": "threat", "data": threat})

        # Detect resources
        observed_resources = self.detect_resources(resource_manager)

        # Report resources (deduplication logic already exists)
        for resource in observed_resources:
            if self.communication_system:
                self.communication_system.agent_to_hq(self, {"type": "resource", "data": resource})

        # Log reported resources
        logger.debug_log(f"Agent {self.agent_id} observed resources: {observed_resources}", level=logging.DEBUG)



        """ # Debug logs
        if observed_threats:
            print(f"Agent {self.role} detected threats: {observed_threats}\n")
        if observed_resources:
            print(f"Agent {self.role} detected resources: {observed_resources}\n") """



    def detect_resources(self, resource_manager, threshold=Agent_field_of_view):
        """
        Detect resources within the given threshold distance (in grid units).
        """
        detected_resources = []
        agent_grid_x = self.x // CELL_SIZE
        agent_grid_y = self.y // CELL_SIZE
        #print(f"Agent {self.agent_id} is at grid position ({agent_grid_x}, {agent_grid_y})")

        for resource in resource_manager.resources:
            if resource.is_depleted():  # Skip depleted resources
                continue

            # Ensure resource position is in grid coordinates
            resource_grid_x = resource.grid_x
            resource_grid_y = resource.grid_y

            # Calculate Euclidean distance in grid units
            distance = ((resource_grid_x - agent_grid_x) ** 2 + (resource_grid_y - agent_grid_y) ** 2) ** 0.5

            if distance <= threshold:  # Compare with threshold in grid units
                #print(f"Resource at ({resource_grid_x}, {resource_grid_y}) is {distance:.2f} units away from agent at ({agent_grid_x}, {agent_grid_y})")
                detected_resources.append(resource)

        return detected_resources









        
    def detect_threats(self, all_agents, enemy_hq):
        """Detect threats (enemy agents or HQs) within local view."""
        threats = []

        # Detect enemy agents
        for agent in all_agents:
            # Calculate distance to agent
            distance = ((agent.x - self.x) ** 2 + (agent.y - self.y) ** 2) ** 0.5

            # Ensure the agent is within perception radius
            if distance > self.local_view * CELL_SIZE:
                continue

            # Ensure valid IDs and attributes
            if not hasattr(agent, "agent_id") or not isinstance(agent.agent_id, AgentIDStruc):
                logger.warning(
                    f"Invalid agent detected by Agent {self.agent_id}: {agent}. Skipping threat detection."
                )
                continue

            # Skip self-detection
            if agent.agent_id == self.agent_id:
                continue

            # Skip friendly agents
            if agent.agent_id.faction_id == self.faction.id:
                continue

            # Add threat if all conditions are met
            threat = {
                "id": agent.agent_id,  # Use the AgentID namedtuple
                "faction": agent.agent_id.faction_id,  # Include the faction ID for clarity
                "type": f"agent.{agent.role}",
                "location": (agent.x, agent.y),
            }
            threats.append(threat)

            logger.debug_log(
                f"Agent {self.agent_id} detected threat: AgentID {agent.agent_id}, "
                f"Faction {agent.agent_id.faction_id}, at location ({agent.x}, {agent.y}).",
                level=logging.DEBUG
            )

        # Detect enemy HQ
        if "position" in enemy_hq and enemy_hq.get("faction_id") is not None:
            distance_to_hq = ((enemy_hq["position"][0] - self.x) ** 2 + (enemy_hq["position"][1] - self.y) ** 2) ** 0.5
            if distance_to_hq <= self.local_view * CELL_SIZE:
                threat = {
                    "id": AgentIDStruc(faction_id=enemy_hq["faction_id"], agent_id="HQ"),  # Use AgentID for HQ
                    "faction": enemy_hq["faction_id"],  # Use the faction ID from HQ
                    "type": "Faction HQ",
                    "location": enemy_hq["position"],
                }
                threats.append(threat)

                logger.debug_log(
                    f"Agent {self.agent_id} detected enemy HQ at location {enemy_hq['position']}.",
                    level=logging.DEBUG
                )

        # Return all detected threats
        return threats


    








 


    

    #                       _                                     _     _          _   _  ___  
#    ___  ___ _ __   __| |   __ _   _ __ ___ _ __   ___  _ __| |_  | |_ ___   | | | |/ _ \ 
#   / __|/ _ \ '_ \ / _` |  / _` | | '__/ _ \ '_ \ / _ \| '__| __| | __/ _ \  | |_| | | | |
#   \__ \  __/ | | | (_| | | (_| | | | |  __/ |_) | (_) | |  | |_  | || (_) | |  _  | |_| |
#   |___/\___|_| |_|\__,_|  \__,_| |_|  \___| .__/ \___/|_|   \__|  \__\___/  |_| |_|\__\_\
#                                           |_|                                            

    

  



#                                    _                                _         _        _       
#     __ _  ___ _ __   ___ _ __ __ _| |_ ___    __ _  __ _  ___ _ __ | |_   ___| |_ __ _| |_ ___ 
#    / _` |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \  / _` |/ _` |/ _ \ '_ \| __| / __| __/ _` | __/ _ \
#   | (_| |  __/ | | |  __/ | | (_| | ||  __/ | (_| | (_| |  __/ | | | |_  \__ \ || (_| | ||  __/
#    \__, |\___|_| |_|\___|_|  \__,_|\__\___|  \__,_|\__, |\___|_| |_|\__| |___/\__\__,_|\__\___|
#    |___/                                           |___/                                       

    def get_state(self, resource_manager, agents, faction, hq_state=None):
        """
        Generate the agent's state representation based on local perception and HQ state.
        Optimized to reduce redundant `is_within_detection_range` calls.
        """
        if hq_state is None:
            raise ValueError(f"[ERROR] hq_state is None for agent {self.role} in faction {self.faction.id}!")

        #  Convert threats/resources into (x, y) tuples for vectorized processing
        threats_positions = [threat["location"] for threat in hq_state["threats"]]
        resources_positions = [(res["location"][0], res["location"][1]) for res in hq_state["resources"]]

        #  Batch detection instead of calling `is_within_detection_range` multiple times
        perceived_threats = [threat for threat, pos in zip(hq_state["threats"], threats_positions) if self.is_within_detection_range(pos)]
        perceived_resources = [res for res, pos in zip(hq_state["resources"], resources_positions) if self.is_within_detection_range(pos)]

        #  Nearest threat/resource calculation **only using perceived entities**
        nearest_threat = find_closest_actor(perceived_threats, entity_type="threat", requester=self) if perceived_threats else None
        nearest_resource = find_closest_actor(perceived_resources, entity_type="resource", requester=self) if perceived_resources else None

        #  Construct normalized state vector
        core_state = [
            self.x / WORLD_WIDTH,
            self.y / WORLD_HEIGHT,
            self.Health / 100,
            nearest_threat["location"][0] / WORLD_WIDTH if nearest_threat else -1,
            nearest_threat["location"][1] / WORLD_HEIGHT if nearest_threat else -1,
            nearest_resource["location"][0] / WORLD_WIDTH if nearest_resource else -1,
            nearest_resource["location"][1] / WORLD_HEIGHT if nearest_resource else -1,
        ]

        # Append one-hot for task type or role (must match TASK_TYPE_MAPPING)
        one_hot_task = [0] * len(TASK_TYPE_MAPPING)
        # Optional: determine current task type and mark active bit
        if self.current_task:
            task_type_index = TASK_TYPE_MAPPING.get(self.current_task.get("type", "none"), None)
            if task_type_index is not None:
                one_hot_task[task_type_index] = 1
        state = core_state + one_hot_task

        logger.debug_log(f"Agent {self.agent_id} state: {state}", level=logging.DEBUG)
        return state




    def update_detection_range(self):
        # Call this whenever self.local_view or CELL_SIZE changes.
        detection_range = self.local_view * CELL_SIZE
        self.detection_range_squared = detection_range * detection_range

    def is_within_detection_range(self, target_position):
        """
        Determines if a given target is within the agent's field of view using a cached squared threshold.
        :param target_position: Tuple (x, y) of the target.
        :return: True if within range, False otherwise.
        """
        if not target_position:
            return False  # Prevents errors if position is missing

        dx = self.x - target_position[0]
        dy = self.y - target_position[1]
        squared_distance = dx * dx + dy * dy

        return squared_distance <= self.detection_range_squared


    #                                      _             _          _           _              __         _                   _   _               __  
    #    _ __ ___   __ _ _ __    _ __ ___ | | ___  ___  | |_ ___   (_)_ __   __| | _____  __  / / __ ___ | | ___    __ _  ___| |_(_) ___  _ __  __\ \ 
    #   | '_ ` _ \ / _` | '_ \  | '__/ _ \| |/ _ \/ __| | __/ _ \  | | '_ \ / _` |/ _ \ \/ / | | '__/ _ \| |/ _ \  / _` |/ __| __| |/ _ \| '_ \/ __| |
    #   | | | | | | (_| | |_) | | | | (_) | |  __/\__ \ | || (_) | | | | | | (_| |  __/>  <  | | | | (_) | |  __/ | (_| | (__| |_| | (_) | | | \__ \ |
    #   |_| |_| |_|\__,_| .__/  |_|  \___/|_|\___||___/  \__\___/  |_|_| |_|\__,_|\___/_/\_\ | |_|  \___/|_|\___|  \__,_|\___|\__|_|\___/|_| |_|___/ |
    #                   |_|                                                                   \_\                                                 /_/ 



    def role_to_index(self, role):
        """
        Map roles to integer indices. Warn and default to 0 if role is unknown.
        """
        role_mapping = {"gatherer": 0, "peacekeeper": 1}  # Add more roles as necessary
        if role not in role_mapping:
            print(f"Warning: Unknown role '{role}' encountered. Defaulting to 0.")
        return role_mapping.get(role, 0)  # Default to 0 (e.g., gatherer) if unknown


    
    

    def report_experience_to_hq(self, state, action, reward, next_state, done):
        """
        Report the experience to the HQ for centralized training.
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        self.faction.receive_experience(experience)



#    ____                     _                                    _     _ _     _        _               
#   |  _ \ ___  __ _  ___ ___| | _____  ___ _ __   ___ _ __    ___| |__ (_) | __| |   ___| | __ _ ___ ___ 
#   | |_) / _ \/ _` |/ __/ _ \ |/ / _ \/ _ \ '_ \ / _ \ '__|  / __| '_ \| | |/ _` |  / __| |/ _` / __/ __|
#   |  __/  __/ (_| | (_|  __/   <  __/  __/ |_) |  __/ |    | (__| | | | | | (_| | | (__| | (_| \__ \__ \
#   |_|   \___|\__,_|\___\___|_|\_\___|\___| .__/ \___|_|     \___|_| |_|_|_|\__,_|  \___|_|\__,_|___/___/
#                                          |_|                                                            



class Peacekeeper(BaseAgent):
    try:
        
    
        def __init__(self, x, y, faction, base_sprite_path, terrain, agents, resource_manager, agent_id, role_actions, communication_system, state_size=DEF_AGENT_STATE_SIZE, event_manager=None, mode="train", network_type="PPOModel"):
            super().__init__(
                x=x,
                y=y,
                role="peacekeeper",
                faction=faction,
                terrain=terrain,
                resource_manager=resource_manager,
                role_actions=role_actions,
                agent_id=agent_id,
                communication_system=communication_system,
                event_manager=event_manager,
                mode=mode,
                network_type=network_type  
            )
            self.base_sprite = pygame.image.load(base_sprite_path).convert_alpha()
            sprite_size = int(SCREEN_HEIGHT * AGENT_SCALE_FACTOR)
            self.base_sprite = pygame.transform.scale(self.base_sprite, (sprite_size, sprite_size))
            self.sprite = tint_sprite(self.base_sprite, faction.color) if faction and hasattr(faction, 'color') else self.base_sprite

           

            from render_display import get_font
            self.font = get_font(24)

            self.known_threats = []
    except Exception as e:
        raise(f"Error in Initialising Peacekeeper class: {e}")



#     ____       _   _                               _     _ _     _        _               
#    / ___| __ _| |_| |__   ___ _ __ ___ _ __    ___| |__ (_) | __| |   ___| | __ _ ___ ___ 
#   | |  _ / _` | __| '_ \ / _ \ '__/ _ \ '__|  / __| '_ \| | |/ _` |  / __| |/ _` / __/ __|
#   | |_| | (_| | |_| | | |  __/ | |  __/ |    | (__| | | | | | (_| | | (__| | (_| \__ \__ \
#    \____|\__,_|\__|_| |_|\___|_|  \___|_|     \___|_| |_|_|_|\__,_|  \___|_|\__,_|___/___/
#                                                                                           




class Gatherer(BaseAgent):
    try:
    
        def __init__(self, x, y, faction, base_sprite_path, terrain, agents, resource_manager, agent_id, role_actions, communication_system, state_size=DEF_AGENT_STATE_SIZE, event_manager=None, mode="train", network_type="PPOModel"):
            super().__init__(
                x=x,
                y=y,
                role="gatherer",
                faction=faction,
                terrain=terrain,
                resource_manager=resource_manager,
                role_actions=role_actions,
                agent_id=agent_id,
                communication_system=communication_system,
                event_manager=event_manager,
                mode=mode,
                network_type=network_type  
            )
            self.base_sprite = pygame.image.load(base_sprite_path).convert_alpha()
            sprite_size = int(SCREEN_HEIGHT * AGENT_SCALE_FACTOR)
            self.base_sprite = pygame.transform.scale(self.base_sprite, (sprite_size, sprite_size))
            self.sprite = tint_sprite(self.base_sprite, faction.color) if faction and hasattr(faction, 'color') else self.base_sprite

            

            from render_display import get_font
            self.font = get_font(24)

            self.known_resources = []

    except Exception as e:
        raise(f"Error in Initialising Gatherer class: {e}")
    




#       _             _                      _     _ _     _        _               
#      / \   _ __ ___| |__   ___ _ __    ___| |__ (_) | __| |   ___| | __ _ ___ ___ 
#     / _ \ | '__/ __| '_ \ / _ \ '__|  / __| '_ \| | |/ _` |  / __| |/ _` / __/ __|
#    / ___ \| | | (__| | | |  __/ |    | (__| | | | | | (_| | | (__| | (_| \__ \__ \
#   /_/   \_\_|  \___|_| |_|\___|_|     \___|_| |_|_|_|\__,_|  \___|_|\__,_|___/___/
#                                                                                   



""" 
class Archer(BaseAgent):
    
    try:
        def __init__(self, x, y, faction, base_sprite_path, terrain, agents, resource_manager, agent_id, role_actions, communication_system, state_size=DEF_AGENT_STATE_SIZE, event_manager=None, mode="train", network_type="PPOModel"):
            super().__init__(
                x=x,
                y=y,
                role="archer",
                faction=faction,
                terrain=terrain,
                resource_manager=resource_manager,
                role_actions=role_actions,
                agent_id=agent_id,
                communication_system=communication_system,
                event_manager=event_manager,
                mode=mode,
                network_type=network_type
            )
            self.base_sprite = pygame.image.load(base_sprite_path).convert_alpha()
            sprite_size = int(SCREEN_HEIGHT * AGENT_SCALE_FACTOR)
            self.base_sprite = pygame.transform.scale(self.base_sprite, (sprite_size, sprite_size))

            self.sprite = tint_sprite(self.base_sprite, faction.color) if faction and hasattr(faction, 'color') else self.base_sprite
    except Exception as e:
        raise(f"Error in Initialising Archer class: {e}") """

   
    