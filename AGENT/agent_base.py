"""
Base class for agents.
    Contains the basic attributes and methods for all agents and agent specific attributes.
"""
"""Common Imports"""
from SHARED.core_imports import *
import UTILITIES.utils_config as utils_config



"""File Specific Imports"""
from NEURAL_NETWORK.PPO_Agent_Network import PPOModel
from NEURAL_NETWORK.DQN_Model import DQNModel
from AGENT.agent_behaviours import AgentBehaviour




#       _                    _                             _ _     _                   _   _
#      / \   __ _  ___ _ __ | |_ ___   _ __   ___  ___ ___(_) |__ | | ___    __ _  ___| |_(_) ___  _ __  ___
#     / _ \ / _` |/ _ \ '_ \| __/ __| | '_ \ / _ \/ __/ __| | '_ \| |/ _ \  / _` |/ __| __| |/ _ \| '_ \/ __|
#    / ___ \ (_| |  __/ | | | |_\__ \ | |_) | (_) \__ \__ \ | |_) | |  __/ | (_| | (__| |_| | (_) | | | \__ \
#   /_/   \_\__, |\___|_| |_|\__|___/ | .__/ \___/|___/___/_|_.__/|_|\___|  \__,_|\___|\__|_|\___/|_| |_|___/
#           |___/                     |_|

# ROLE__ACTIONS denotes the possible actions for each agent role.
# The actions are grouped into categories based on the agent's role.
# The model will be trained to predict the best action to take based on
# the agent's role and current state.


logger = Logger(log_file="agent_base_log.txt", log_level=logging.DEBUG)


""" Tint the sprite with the faction colour so its easy to identify the agent's faction. """


def tint_sprite(sprite, tint_colour):
    """
    Apply a colour tint to a sprite.
    :param sprite: The base sprite image (Surface).
    :param tint_colour: The colour to tint the sprite (RGB tuple).
    :return: The tinted sprite (Surface).
    """
    tinted_sprite = sprite.copy()  # Make a copy of the original sprite
    # Apply colour tint
    tinted_sprite.fill(tint_colour, special_flags=pygame.BLEND_RGB_MULT)
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

        # Part of the init is for the agent itself and part is for the agent's
        # network.
        def __init__(
            # Inputs
            self,
            x,  # The x-coordinate of the agent's position.
            # The y-coordinate of the agent's position. (Should/Could be
            # converted to a tuple (x, y))
            y,
            role,  # The role of the agent. Peacekeeper or Gatherer.
            faction,  # The faction the agent belongs to.
            terrain,  # Reference to the terrain object.
            # Reference to the resource manager object.
            resource_manager,
            role_actions,  # The actions the agent can perform.
            agent_id,  # Unique identifier for the agent.
            # Reference to the communication system object.
            communication_system,
            # Placeholder for the event manager object.
            event_manager: object = None,
            state_size=utils_config.DEF_AGENT_STATE_SIZE,
            mode: str = "train",  # Default mode is "train".
            # Default network type is PPOModel.
            network_type: str = "PPOModel"
        ):
            """
            Initialise the agent with a specific network model (e.g., PPO, DQN, etc.).
            """
            # Convert string network type to integer using
            # utils_config.NETWORK_TYPE_MAPPING
            network_type_int = utils_config.NETWORK_TYPE_MAPPING.get(
                network_type, 1)  # Default to "none" if not found

            # Agent-specific initialisation
            self.x: float = x
            self.y: float = y
            self.role = role
            self.faction = faction
            self.terrain = terrain
            self.resource_manager = resource_manager
            self.role_actions = role_actions[role]
            self.agent_id = utils_config.AgentIDStruc(faction.id, agent_id)

            # Ensure a valid state size is always used
            self.state_size = state_size if state_size is not None else utils_config.DEF_AGENT_STATE_SIZE

            self.communication_system = communication_system
            self.event_manager = event_manager
            self.mode = mode
            self.Health = 100
            self.speed = 1
            self.local_view = utils_config.Agent_field_of_view
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

            self.current_task = None  # Initialise the current task with None
            # Initialise the current task state with
            # utils_config.TaskState.NONE
            self.current_task_state = utils_config.TaskState.NONE

            # Initialise other components
            # Temporary experience buffer (for training, works like memory)
            self.experience_buffer = []
            # Initiaises the structure for the task creation/handling
            self.create_task = utils_config.create_task

    except Exception as e:
        raise (f"Error in BaseAgent initialisation: {e}")

    def initialise_network(
            self,
            network_type_int,
            state_size,
            action_size,
            AgentID):
        """
        Initialise the network model based on the selected network type.
        """
        if network_type_int == utils_config.NETWORK_TYPE_MAPPING["PPOModel"]:
            print(
                "\033[93m" +
                f"Initialising PPOModel with state_size={state_size}, action_size={action_size} for AgentID={AgentID}" +
                "\033[0m")
            return PPOModel(
                AgentID=AgentID,
                state_size=state_size,
                action_size=action_size,
                training_mode=self.mode
            )

        elif network_type_int == utils_config.NETWORK_TYPE_MAPPING["DQNModel"]:
            print(
                "\033[93m" +
                f"Initialising DQNModel with state_size={state_size}, action_size={action_size} for AgentID={AgentID}" +
                "\033[0m")
            return DQNModel(
                state_size=state_size,
                action_size=action_size
            )

        else:
            raise ValueError(f"Unsupported network type: {network_type_int}")

    def can_move_to(self, new_x, new_y):
        try:
            grid_x = int(new_x // utils_config.CELL_SIZE)
            grid_y = int(new_y // utils_config.CELL_SIZE)

            if 0 <= grid_x < len(
                    self.terrain.grid) and 0 <= grid_y < len(
                    self.terrain.grid[0]):
                tile = self.terrain.grid[grid_x][grid_y]

                tile_type = None
                if isinstance(tile, dict):
                    tile_type = tile.get('type')
                elif hasattr(tile, 'dtype') and 'type' in tile.dtype.names:
                    tile_type = tile['type']

                return tile_type == 'land'

            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"[WARN] Out-of-bounds grid access: ({grid_x}, {grid_y})",
                        level=logging.WARNING)
                return False

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[ERROR] Exception in can_move_to({new_x}, {new_y}): {repr(e)}",
                    level=logging.ERROR)
            return False

    def move(self, dx, dy):
        try:
            # Calculate new potential position
            new_x = self.x + dx * self.speed
            new_y = self.y + dy * self.speed

            # Convert pixel coordinates to grid coordinates
            grid_x = int(new_x // utils_config.CELL_SIZE)
            grid_y = int(new_y // utils_config.CELL_SIZE)
            current_grid_x = int(self.x // utils_config.CELL_SIZE)
            current_grid_y = int(self.y // utils_config.CELL_SIZE)

            # Check if the new position is valid and on land
            if self.can_move_to(new_x, new_y):
                # Mark the current cell as faction territory (with bounds
                # check)
                if 0 <= current_grid_x < len(
                        self.terrain.grid) and 0 <= current_grid_y < len(
                        self.terrain.grid[0]):
                    self.terrain.grid[current_grid_x][current_grid_y]['faction'] = self.faction.id

                # Update agent position
                self.x = new_x
                self.y = new_y
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Agent {self.agent_id} ({self.role}) moved to ({new_x}, {new_y})")

                # Update position history
                self.recent_positions = getattr(self, "recent_positions", [])
                self.recent_positions.append((self.x, self.y))
                if len(self.recent_positions) > 10:
                    self.recent_positions.pop(0)

                # Check if stuck
                if len(set(self.recent_positions)) <= 2:
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Agent {self.agent_id} ({self.role}) is likely stuck. "
                            f"Recent positions: {self.recent_positions}. Penalising.",
                            level=logging.WARNING)
                    self.Health -= 2
                    if utils_config.ENABLE_LOGGING:
                        logger.log_msg(
                            f"Agent {self.agent_id} ({self.role}) has been penalised for being stuck.")
                    if self.Health <= 0:
                        print(
                            f"Agent {self.agent_id} ({self.role}) has died from being stuck.")

                # Mark new cell as faction territory (with bounds check)
                if 0 <= grid_x < len(
                        self.terrain.grid) and 0 <= grid_y < len(
                        self.terrain.grid[0]):
                    self.terrain.grid[grid_x][grid_y]['faction'] = self.faction.id
            else:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Agent {self.agent_id} ({self.role}) attempted invalid move to ({new_x}, {new_y}).",
                        level=logging.ERROR)

        except Exception as e:
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[ERROR] Exception during move(): {repr(e)}",
                    level=logging.ERROR)

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
        :param task_state: The new state of the current task (utils_config.TaskState).
        """
        self.current_task_state = task_state
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"{self.role} task state updated to {task_state}.",
                level=logging.DEBUG)

    def update(self, resource_manager, agents, hq_state):
        """
        Update the agent's state. This includes:
        - Performing assigned tasks.
        - Observing the environment.
        - Reporting experiences to the faction.

        """
        self._perception_cache = None
        try:
            # Log the current task before performing it
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"[TASK EXECUTION] Agent {self.agent_id} executing task: {self.current_task}",
                    level=logging.DEBUG)

            # Retrieve the agent's current state based on HQ state
            state = self.get_state(
                resource_manager, agents, self.faction, hq_state)
            if state is None:
                raise RuntimeError(
                    f"[CRITICAL] Agent {self.agent_id} received a None state from get_state")

            if utils_config.ENABLE_LOGGING:
                # Split state for readability
                core = state[:7]  # position, health, nearest threat/resource
                task_one_hot = state[7:7 + len(utils_config.TASK_TYPE_MAPPING)]
                # e.g., target_x, target_y, current_action_norm
                task_info = state[7 + len(utils_config.TASK_TYPE_MAPPING):]

                # Map task type index back to name (if any bit is 1)
                if 1 not in task_one_hot:
                    task_type_name = "none"
                else:
                    try:
                        task_type_index = task_one_hot.index(1)
                        task_type_name = list(utils_config.TASK_TYPE_MAPPING.keys())[task_type_index]
                    except (ValueError, IndexError):
                        task_type_name = "none"


                logger.log_msg(
                    f"\n[STATE DEBUG] Agent {self.agent_id} ({self.role}) State:\n"
                    f" - Position: ({core[0]*(utils_config.WORLD_WIDTH):.1f}, {core[1]*utils_config.WORLD_HEIGHT:.1f})\n"
                    f" - Health: {core[2]*100:.0f}\n"
                    f" - Nearest Threat: ({core[3]*(utils_config.WORLD_WIDTH):.1f}, {core[4]*utils_config.WORLD_HEIGHT:.1f})\n"
                    f" - Nearest Resource: ({core[5]*utils_config.WORLD_WIDTH:.1f}, {core[6]*utils_config.WORLD_HEIGHT:.1f})\n"
                    f" - Task Type: {task_type_name}\n"
                    f" - Task Target: ({task_info[0]*utils_config.WORLD_WIDTH:.1f}, {task_info[1]*utils_config.WORLD_HEIGHT:.1f})\n"
                    f" - Current Action Index: {int(task_info[2]*len(self.role_actions)) if task_info[2] >= 0 else 'None'}\n",
                    level=logging.DEBUG)

            # Execute the current task or decide on a new action
            reward, task_state = self.perform_task(
                state, resource_manager, agents)
            # Update the task state based on execution
            self.update_task_state(task_state)

            # Observe the environment and report findings to the faction
            self.observe(agents,
                         {"position": self.faction.home_base["position"]},
                         resource_manager)

            # Log the task state and reward for centralised learning
            if task_state in [
                    utils_config.TaskState.SUCCESS,
                    utils_config.TaskState.FAILURE]:
                next_state = self.get_state(
                    resource_manager, agents, self.faction, hq_state)
                done = task_state in [
                    utils_config.TaskState.SUCCESS,
                    utils_config.TaskState.FAILURE]

                # Report the agent's experience to the HQ
                self.report_experience_to_hq(
                    state, self.current_task, reward, next_state, done)

            # Handle health-related conditions
            if self.Health <= 0:
                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"{self.role} has died and will be removed from the game.",
                        level=logging.WARNING)
                print(f"{self.role} has died and will be removed from the game.")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"An error occurred while updating the agent: {e}")


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
            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Agent {self.agent_id} observed threats: {observed_threats}",
                    level=logging.DEBUG)

        for threat in observed_threats:
            # Ensure the threat is from a different faction
            if isinstance(threat["id"], utils_config.AgentIDStruc):
                if threat["id"].faction_id == self.faction.id:
                    continue  # Skip friendly threats
            else:
                if threat["faction"] == self.faction.id:
                    continue  # Skip friendly threats

            # Report threat to HQ using the communication system
            if self.communication_system:
                self.communication_system.agent_to_hq(
                    self, {"type": "threat", "data": threat})

        # Detect resources
        observed_resources = self.detect_resources(resource_manager)

        # Report resources (deduplication logic already exists)
        for resource in observed_resources:
            if self.communication_system:
                self.communication_system.agent_to_hq(
                    self, {"type": "resource", "data": resource})

        # Log reported resources
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"Agent {self.agent_id} observed resources: {observed_resources}",
                level=logging.DEBUG)

        """ # Debug logs
        if observed_threats:
            print(f"Agent {self.role} detected threats: {observed_threats}\n")
        if observed_resources:
            print(f"Agent {self.role} detected resources: {observed_resources}\n") """

    def detect_resources(
            self,
            resource_manager,
            threshold=utils_config.Agent_field_of_view):
        """
        Detect resources within the given threshold distance (in grid units).
        """
        detected_resources = []
        agent_grid_x = self.x // utils_config.CELL_SIZE
        agent_grid_y = self.y // utils_config.CELL_SIZE
        # print(f"Agent {self.agent_id} is at grid position ({agent_grid_x}, {agent_grid_y})")

        for resource in resource_manager.resources:
            if resource.is_depleted():  # Skip depleted resources
                continue

            # Ensure resource position is in grid coordinates
            resource_grid_x = resource.grid_x
            resource_grid_y = resource.grid_y

            # Calculate Euclidean distance in grid units
            distance = ((resource_grid_x - agent_grid_x) ** 2 +
                        (resource_grid_y - agent_grid_y) ** 2) ** 0.5

            if distance <= threshold:  # Compare with threshold in grid units
                # print(f"Resource at ({resource_grid_x}, {resource_grid_y}) is {distance:.2f} units away from agent at ({agent_grid_x}, {agent_grid_y})")
                detected_resources.append(resource)

        return detected_resources

    def detect_threats(self, all_agents, enemy_hq):
        """Detect threats (enemy agents or HQs) within local view."""
        threats = []

        # Detect enemy agents
        for agent in all_agents:
            # Calculate distance to agent
            distance = ((agent.x - self.x) ** 2 +
                        (agent.y - self.y) ** 2) ** 0.5

            # Ensure the agent is within perception radius
            if distance > self.local_view * utils_config.CELL_SIZE:
                continue

            # Ensure valid IDs and attributes
            if not hasattr(
                    agent,
                    "agent_id") or not isinstance(
                    agent.agent_id,
                    utils_config.AgentIDStruc):
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

            if utils_config.ENABLE_LOGGING:
                logger.log_msg(
                    f"Agent {self.agent_id} detected threat: AgentID {agent.agent_id}, "
                    f"Faction {agent.agent_id.faction_id}, at location ({agent.x}, {agent.y}).",
                    level=logging.DEBUG)

        # Detect enemy HQ
        if "position" in enemy_hq and enemy_hq.get("faction_id") is not None:
            distance_to_hq = ((enemy_hq["position"][0] - self.x) **
                              2 + (enemy_hq["position"][1] - self.y) ** 2) ** 0.5
            if distance_to_hq <= self.local_view * utils_config.CELL_SIZE:
                threat = {
                    # Use AgentID for HQ
                    "id": utils_config.AgentIDStruc(faction_id=enemy_hq["faction_id"], agent_id="HQ"),
                    # Use the faction ID from HQ
                    "faction": enemy_hq["faction_id"],
                    "type": "Faction HQ",
                    "location": enemy_hq["position"],
                }
                threats.append(threat)

                if utils_config.ENABLE_LOGGING:
                    logger.log_msg(
                        f"Agent {self.agent_id} detected enemy HQ at location {enemy_hq['position']}.",
                        level=logging.DEBUG)

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
        if hq_state is None:
            raise ValueError(
                f"[ERROR] hq_state is None for agent {self.role} in faction {self.faction.id}!")

        # ✅ Use actual agent perception functions
        perceived_threats = self.detect_threats(agents, hq_state)
        perceived_resources = self.detect_resources(resource_manager)

        # 🔍 Find closest ones
        nearest_threat = find_closest_actor(
            perceived_threats,
            entity_type="threat",
            requester=self) if perceived_threats else None
        nearest_resource = find_closest_actor(
            perceived_resources,
            entity_type="resource",
            requester=self) if perceived_resources else None

        # 🧠 Build base state vector
        core_state = [
            self.x /
            utils_config.WORLD_WIDTH,
            self.y /
            utils_config.WORLD_HEIGHT,
            self.Health /
            100,
            nearest_threat["location"][0] /
            utils_config.WORLD_WIDTH if nearest_threat else -
            1,
            nearest_threat["location"][1] /
            utils_config.WORLD_HEIGHT if nearest_threat else -
            1,
            nearest_resource.x /
            utils_config.WORLD_WIDTH if nearest_resource else -
            1,
            nearest_resource.y /
            utils_config.WORLD_HEIGHT if nearest_resource else -
            1,
        ]

        # ✅ Task encoding
        one_hot_task = [0] * len(utils_config.TASK_TYPE_MAPPING)
        if self.current_task:
            task_type_index = utils_config.TASK_TYPE_MAPPING.get(
                self.current_task.get("type", "none"))
            if task_type_index is not None:
                one_hot_task[task_type_index] = 1

        # ✅ Task-specific info
        task_target = self.current_task.get("target", {}).get(
            "position", (-1, -1)) if self.current_task else (-1, -1)
        task_target_x = task_target[0] / utils_config.WORLD_WIDTH
        task_target_y = task_target[1] / utils_config.WORLD_HEIGHT
        current_action_norm = self.current_action / \
            len(self.role_actions) if getattr(
                self, "current_action", -1) >= 0 else -1

        task_info = [task_target_x, task_target_y, current_action_norm]

        state = core_state + one_hot_task + task_info

        # 🪵 Log for debugging
        if utils_config.ENABLE_LOGGING:
            logger.log_msg(
                f"[STATE DEBUG] Agent {self.agent_id} ({self.role}) state generated. Core={core_state}, Task={one_hot_task}, Target={task_info}",
                level=logging.DEBUG)

        return state

    def update_detection_range(self):
        # Call this whenever self.local_view or utils_config.CELL_SIZE changes.
        detection_range = self.local_view * utils_config.CELL_SIZE
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
        role_mapping = {"gatherer": 0,
                        "peacekeeper": 1}  # Add more roles as necessary
        if role not in role_mapping:
            print(
                f"Warning: Unknown role '{role}' encountered. Defaulting to 0.")
        # Default to 0 (e.g., gatherer) if unknown
        return role_mapping.get(role, 0)

    def report_experience_to_hq(self, state, action, reward, next_state, done):
        """
        Report the experience to the HQ for centralised training.
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

        def __init__(
                self,
                x,
                y,
                faction,
                base_sprite_path,
                terrain,
                agents,
                resource_manager,
                agent_id,
                role_actions,
                communication_system,
                state_size=utils_config.DEF_AGENT_STATE_SIZE,
                event_manager=None,
                mode="train",
                network_type="PPOModel"):
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
            self.base_sprite = pygame.image.load(
                base_sprite_path).convert_alpha()
            sprite_size = int(utils_config.SCREEN_HEIGHT *
                              utils_config.AGENT_SCALE_FACTOR)
            self.base_sprite = pygame.transform.scale(
                self.base_sprite, (sprite_size, sprite_size))
            self.sprite = tint_sprite(
                self.base_sprite, faction.colour) if faction and hasattr(
                faction, 'colour') else self.base_sprite

            from RENDER.Game_Renderer import get_font
            self.font = get_font(24)

            self.known_threats = []
    except Exception as e:
        raise (f"Error in Initialising Peacekeeper class: {e}")


#     ____       _   _                               _     _ _     _        _
#    / ___| __ _| |_| |__   ___ _ __ ___ _ __    ___| |__ (_) | __| |   ___| | __ _ ___ ___
#   | |  _ / _` | __| '_ \ / _ \ '__/ _ \ '__|  / __| '_ \| | |/ _` |  / __| |/ _` / __/ __|
#   | |_| | (_| | |_| | | |  __/ | |  __/ |    | (__| | | | | | (_| | | (__| | (_| \__ \__ \
#    \____|\__,_|\__|_| |_|\___|_|  \___|_|     \___|_| |_|_|_|\__,_|  \___|_|\__,_|___/___/
#


class Gatherer(BaseAgent):
    try:

        def __init__(
                self,
                x,
                y,
                faction,
                base_sprite_path,
                terrain,
                agents,
                resource_manager,
                agent_id,
                role_actions,
                communication_system,
                state_size=utils_config.DEF_AGENT_STATE_SIZE,
                event_manager=None,
                mode="train",
                network_type="PPOModel"):
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
            self.base_sprite = pygame.image.load(
                base_sprite_path).convert_alpha()
            sprite_size = int(utils_config.SCREEN_HEIGHT *
                              utils_config.AGENT_SCALE_FACTOR)
            self.base_sprite = pygame.transform.scale(
                self.base_sprite, (sprite_size, sprite_size))
            self.sprite = tint_sprite(
                self.base_sprite, faction.colour) if faction and hasattr(
                faction, 'colour') else self.base_sprite
            from RENDER.Game_Renderer import get_font

            self.font = get_font(24)

            self.known_resources = []

    except Exception as e:
        raise (f"Error in Initialising Gatherer class: {e}")
